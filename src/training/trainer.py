from pathlib import Path
from typing import Any, Dict, Union, List

import os
import cv2
import einops
import imageio
import numpy as np
import pyrallis
import torch
import torch.nn.functional as F
from PIL import Image
from loguru import logger
from matplotlib import cm
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
import kornia

from src import utils
from src.configs.train_config import TrainConfig
from src.models.textured_mesh import TexturedMeshModel
from src.stable_diffusion_depth import StableDiffusion #Base TEXTure - NOT in USE !
from src.sdxl_depth import SDXL #SDXL base 1.0 + SDXL inpaint 1.0
from src.training.views_dataset import ViewsDataset, MultiviewDataset
from src.utils import make_path, tensor2numpy
import time # texture generation time calculation
import os
from pathlib import Path

import torchvision.utils as vutils

class TEXTure:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.paint_step = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ncount = 0
        self.ncount_1 = 0
        self.texturecount = 0
        self.image_count = 0  # Counter for valid images
        self.initialized_count = 0
        self.initial_uvmap = []

        utils.seed_everything(self.cfg.optim.seed)

        # Make view_dirs
        self.exp_path = make_path(self.cfg.log.exp_dir)
        self.ckpt_path = make_path(self.exp_path / 'checkpoints')
        self.train_renders_path = make_path(self.exp_path / 'vis' / 'train')
        self.eval_renders_path = make_path(self.exp_path / 'vis' / 'eval')
        self.final_renders_path = make_path(self.exp_path / 'results')

        self.init_logger()
        pyrallis.dump(self.cfg, (self.exp_path / 'config.yaml').open('w'))

        self.view_dirs = ['front', 'left', 'back', 'right', 'overhead', 'bottom']
        # Mesh 불러오는 과정
        self.mesh_model = self.init_mesh_model()
        self.mask_model = self.init_mesh_model()
        # Diffusion Initialization
        self.diffusion = self.init_diffusion()
        # Text_embeddings initialization
        self.text_z, self.text_string ,self.text_z_origin, self.text_string_origin= self.calc_text_embeddings()
        self.dataloaders = self.init_dataloaders()
        self.back_im = torch.Tensor(np.array(Image.open(self.cfg.guide.background_img).convert('RGB'))).to(
            self.device).permute(2, 0,
                                 1) / 255.0

        logger.info(f'Successfully initialized {self.cfg.log.exp_name}')

    def init_mesh_model(self) -> nn.Module:
        cache_path = Path('cache') / Path(self.cfg.guide.shape_path).stem
        cache_path.mkdir(parents=True, exist_ok=True)
        model = TexturedMeshModel(self.cfg.guide, device=self.device,
                                  render_grid_size=self.cfg.render.train_grid_size,
                                  cache_path=cache_path,
                                  texture_resolution=self.cfg.guide.texture_resolution,
                                  augmentations=False)

        model = model.to(self.device)
        logger.info(
            f'Loaded Mesh, #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
        logger.info(model)
        return model

    def init_diffusion(self) -> Any:
        diffusion_model = SDXL(self.device, model_name=self.cfg.guide.diffusion_name,
                                          concept_name=self.cfg.guide.concept_name,
                                          concept_path=self.cfg.guide.concept_path,
                                          latent_mode=False,
                                          min_timestep=self.cfg.optim.min_timestep,
                                          max_timestep=self.cfg.optim.max_timestep,
                                          no_noise=self.cfg.optim.no_noise,
                                          use_inpaint=True, use_autodepth=False)

        for p in diffusion_model.parameters():
            p.requires_grad = False
        return diffusion_model

    def calc_text_embeddings(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        ref_text = self.cfg.guide.text
        
        if not self.cfg.guide.append_direction:
            text_z = self.diffusion.get_text_embeds([ref_text])
            text_string = ref_text
        else:
            text_z = []
            text_string = []
            text_string_origin = ref_text.split('{', 1)[0].strip()
            text_z_origin = self.diffusion.get_text_embeds([ref_text.split('{', 1)[0].strip()])
            for d in self.view_dirs:
                text = ref_text.format(d)
                text_string.append(text)
                logger.info(text)
                negative_prompt = None
                logger.info(negative_prompt)
                text_z.append(self.diffusion.get_text_embeds([text], negative_prompt=negative_prompt))
        return text_z, text_string, text_z_origin, text_string_origin

    def init_dataloaders(self) -> Dict[str, DataLoader]:
        init_train_dataloader = MultiviewDataset(self.cfg.render, device=self.device).dataloader()

        val_loader = ViewsDataset(self.cfg.render, device=self.device,
                                  size=self.cfg.log.eval_size).dataloader()
        # Will be used for creating the final video
        val_large_loader = ViewsDataset(self.cfg.render, device=self.device,
                                        size=self.cfg.log.full_eval_size).dataloader()
        dataloaders = {'train': init_train_dataloader, 'val': val_loader,
                       'val_large': val_large_loader}
        return dataloaders

    def init_logger(self):
        logger.remove()  # Remove default logger
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format)
        logger.add(self.exp_path / 'log.txt', colorize=False, format=log_format)

    def paint(self):
        logger.info('Starting training ^_^')
        # Evaluate the initialization
        self.evaluate(self.dataloaders['val'], self.eval_renders_path)
        # TexturedMeshModle to training Mode <-> self.mesh_model.eval()
        self.mesh_model.train()

        pbar = tqdm(total=len(self.dataloaders['train']), initial=self.paint_step,
                    bar_format='{desc}: {percentage:3.0f}% painting step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        #self.dataloaders : train, val, val_large dict
        #train 에는 Mesh, Phi, theta등 카메라 뷰 parameter 포함됨
        for data in self.dataloaders['train']:
            if self.paint_step == 0:
                self.paint_step += 1
                pbar.update(1)
                self.paint_viewpoint_initial(data, UV_MAP=False, initial=False)
                self.evaluate(self.dataloaders['val'], self.eval_renders_path)
                self.mesh_model.train()

            else :
                self.paint_step += 1
                pbar.update(1)
                self.paint_viewpoint(data)
                self.evaluate(self.dataloaders['val'], self.eval_renders_path)
                self.mesh_model.train()

        self.mesh_model.change_default_to_median()
        logger.info('Finished Painting ^_^')
        logger.info('Saving the last result...')
        self.full_eval()
        logger.info('\tDone!')

    # Dataloader을 이용해 mesh_model의 현재상태 평가,save picture, video 생성
    # Dataloader['val'] 에는 viewpoint 정보 들어있다.
    def evaluate(self, dataloader: DataLoader, save_path: Path, save_as_video: bool = False):
        logger.info(f'Evaluating and saving model, painting iteration #{self.paint_step}...')
        self.mesh_model.eval()
        save_path.mkdir(exist_ok=True)

        if save_as_video:
            all_preds = []
        for i, data in enumerate(dataloader):
            preds, textures, depths, normals = self.eval_render(data)

            pred = tensor2numpy(preds[0])

            if save_as_video:
                all_preds.append(pred)
            else:
                # Image.fromarray(pred).save(save_path / f"step_{self.paint_step:05d}_{i:04d}_rgb.jpg")
                # Image.fromarray((cm.seismic(normals[0, 0].cpu().numpy())[:, :, :3] * 255).astype(np.uint8)).save(
                #   save_path / f'{self.paint_step:04d}_{i:04d}_normals_cache.jpg')
                if self.paint_step == 0:
                    # Also save depths for debugging
                    torch.save(depths[0], save_path / f"{i:04d}_depth.pt")

        # Texture map is the same, so just take the last result
        texture = tensor2numpy(textures[0])
        Image.fromarray(texture).save(save_path / f"step_{self.paint_step:05d}_texture.png")

        if save_as_video:
            all_preds = np.stack(all_preds, axis=0)

            dump_vid = lambda video, name: imageio.mimsave(save_path / f"step_{self.paint_step:05d}_{name}.mp4", video,
                                                           fps=25,
                                                           quality=8, macro_block_size=1)

            dump_vid(all_preds, 'rgb')
        logger.info('Done!')

    def full_eval(self, output_dir: Path = None):
        if output_dir is None:
            output_dir = self.final_renders_path
        self.evaluate(self.dataloaders['val_large'], output_dir, save_as_video=True)
        # except:
        #     logger.error('failed to save result video')

        if self.cfg.log.save_mesh:
            save_path = make_path(self.exp_path / 'mesh')
            logger.info(f"Saving mesh to {save_path}")

            self.mesh_model.export_mesh(save_path)

            logger.info(f"\tDone!")

    def paint_viewpoint(self, data: Dict[str, Any]):
        logger.info(f'--- Painting step #{self.paint_step} ---')
        theta, phi, radius = data['theta'], data['phi'], data['radius']
        # If offset of phi was set from code

        phi = phi - np.deg2rad(self.cfg.render.front_offset)
        phi = float(phi + 2 * np.pi if phi < 0 else phi)
        logger.info(f'Painting from theta: {theta}, phi: {phi}, radius: {radius}')

        # Set background image
        if self.cfg.guide.use_background_color:
            background = torch.Tensor([0, 0.8, 0]).to(self.device)
        else:
            #background image를 grid size로 맞춤
            #self.back_im.unsqueeze(0) : [1, 3, 1024, 1024]
            background = F.interpolate(self.back_im.unsqueeze(0),
                                       (self.cfg.render.train_grid_size, self.cfg.render.train_grid_size),
                                       mode='bilinear', align_corners=False)

        # Render from viewpoint
        # 여기서 정면 시작, Rgb 이미지, depth를 뽑는다.
        # outputs[Tensor] : image, mask, background, foreground, depth, normals, render_cache(uv_features, face_normals, face_idx, depth_map), texture_map
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, background=background)
        render_cache = outputs['render_cache']
        rgb_render_raw = outputs['image']  # Render where missing values have special color
        depth_render = outputs['depth']
        # Render again with the median value to use as rgb, we shouldn't have color leakage, but just in case
        outputs = self.mesh_model.render(background=background,
                                         render_cache=render_cache, use_median=self.paint_step > 1)
        rgb_render = outputs['image']
        # Render meta texture map
        meta_output = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                             use_meta_texture=True, render_cache=render_cache)

        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        z_normals_cache = meta_output['image'].clamp(0, 1)
        edited_mask = meta_output['image'].clamp(0, 1)[:, 1:2]

        self.log_train_image(rgb_render, 'rendered_input')
        self.log_train_image(depth_render[0, 0], 'depth', colormap=True)
        self.log_train_image(z_normals[0, 0], 'z_normals', colormap=True)
        self.log_train_image(z_normals_cache[0, 0], 'z_normals_cache', colormap=True)

        # text embeddings
        if self.cfg.guide.append_direction:
            dirs = data['dir']  # [B,]
            text_z = self.text_z[dirs]
            text_string = self.text_string[dirs]
        else:
            text_z = self.text_z
            text_string = self.text_string
        logger.info(f'text: {text_string}')
        
        update_mask, generate_mask, refine_mask = self.calculate_trimap(rgb_render_raw=rgb_render_raw,
                                                                        depth_render=depth_render,
                                                                        z_normals=z_normals,
                                                                        z_normals_cache=z_normals_cache,
                                                                        edited_mask=edited_mask,
                                                                        mask=outputs['mask'])

        update_ratio = float(update_mask.sum() / (update_mask.shape[2] * update_mask.shape[3]))
        if self.cfg.guide.reference_texture is not None and update_ratio < 0.01:
            logger.info(f'Update ratio {update_ratio:.5f} is small for an editing step, skipping')
            return

        self.log_train_image(rgb_render * (1 - update_mask), name='masked_input')
        self.log_train_image(rgb_render * refine_mask, name='refine_regions')

        # Crop to inner region based on object mask
        min_h, min_w, max_h, max_w = utils.get_nonzero_region(outputs['mask'][0, 0])
        crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]
        cropped_rgb_render = crop(rgb_render)
        cropped_depth_render = crop(depth_render)
        cropped_update_mask = crop(update_mask)
        self.log_train_image(cropped_rgb_render, name='cropped_input')

        checker_mask = None
        if self.paint_step > 1 or self.cfg.guide.initial_texture is not None:
            checker_mask = self.generate_checkerboard(crop(update_mask), crop(refine_mask),
                                                      crop(generate_mask))
            self.log_train_image(F.interpolate(cropped_rgb_render, (1024, 1024)) * (1 - checker_mask),
                                 'checkerboard_input')
        self.diffusion.use_inpaint = self.cfg.guide.use_inpainting and self.paint_step > 1

        cropped_rgb_output, steps_vis = self.diffusion.img2img_step(
                text_z, 
                cropped_rgb_render.detach(),
                cropped_depth_render.detach(),
                guidance_scale=self.cfg.guide.guidance_scale,
                strength=1.0, update_mask=cropped_update_mask,
                fixed_seed=self.cfg.optim.seed,
                check_mask=checker_mask,
                intermediate_vis=self.cfg.log.vis_diffusion_steps)
        self.log_train_image(cropped_rgb_output, name='direct_output')
        self.log_diffusion_steps(steps_vis)

        cropped_rgb_output = F.interpolate(cropped_rgb_output,
                                           (cropped_rgb_render.shape[2], cropped_rgb_render.shape[3]),
                                           mode='bilinear', align_corners=False)

        # Extend rgb_output to full image size
        rgb_output = rgb_render.clone()
        rgb_output[:, :, min_h:max_h, min_w:max_w] = cropped_rgb_output
        self.log_train_image(rgb_output, name='full_output')

        # Project back
        object_mask = outputs['mask']
        fitted_pred_rgb, _ = self.project_back(render_cache=render_cache, background=background, rgb_output=rgb_output,
                                               object_mask=object_mask, update_mask=update_mask, z_normals=z_normals,
                                               z_normals_cache=z_normals_cache, initial=False)
        self.log_train_image(fitted_pred_rgb, name='fitted')

        return

    def paint_viewpoint_initial(self, data: Dict[str, Any], UV_MAP=False, initial=False):
        logger.info(f'--- Painting step #{self.paint_step} ---')
        theta, phi, radius = data['theta'], data['phi'], data['radius']
        # phi_angles = [np.pi, np.pi/2, 3*np.pi/2, 0]
        phi_angles = [0, np.pi/2, 3*np.pi/2, np.pi]
        # phi_angles = [np.pi/4, 2*np.pi/4, 6*np.pi/4, 7*np.pi/4,]
        # phi_angles = [4 * np.pi/4, 2 * np.pi/4, 6 * np.pi/4, 0 * np.pi/4]
        cropped_renders = []
        cropped_depths = []
        cropped_masks = []
        render_caches = []
        object_masks = []
        update_masks = []
        z_normals_list = []
        z_normals_caches = []
        rgb_renders = []
        min_hs = []
        min_ws = []
        max_hs = []
        max_ws = []

        for idx, phi in enumerate(phi_angles):
            self.idx = idx
            phi = phi - np.deg2rad(self.cfg.render.front_offset)
            phi = float(phi + 2 * np.pi if phi < 0 else phi)
            logger.info(f'Painting from theta: {theta}, phi: {phi}, radius: {radius}')

            # Set background image
            if self.cfg.guide.use_background_color:
                background = torch.Tensor([0, 0.8, 0]).to(self.device)
            else:
                #background image를 grid size로 맞춤
                #self.back_im.unsqueeze(0) : [1, 3, 1024, 1024]
                background = F.interpolate(self.back_im.unsqueeze(0),
                                        (self.cfg.render.train_grid_size, self.cfg.render.train_grid_size),
                                        mode='bilinear', align_corners=False)

            # Rendering Process : Kaolin을 이용해 depthMap, Rendered image를 얻음
            # outputs[Tensor] : image, mask, background, foreground, depth, normals, render_cache(uv_features, face_normals, face_idx, depth_map), texture_map
            outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, background=background)
            render_cache = outputs['render_cache']
            rgb_render_raw = outputs['image']  # Render where missing values have special color
            depth_render = outputs['depth']
            # Render again with the median value to use as rgb, we shouldn't have color leakage, but just in case
            outputs = self.mesh_model.render(background=background,
                                            render_cache=render_cache, use_median=self.paint_step > 1)
            rgb_render = outputs['image']
            # Render meta texture map - Kaolin 사용
            meta_output = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                                use_meta_texture=True, render_cache=render_cache)

            z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
            z_normals_cache = meta_output['image'].clamp(0, 1)
            edited_mask = meta_output['image'].clamp(0, 1)[:, 1:2]

            self.log_train_image(rgb_render, 'rendered_input_initial')
            self.log_train_image(depth_render[0, 0], 'depth_initial', colormap=True)
            self.log_train_image(z_normals[0, 0], 'z_normals_initial', colormap=True)
            self.log_train_image(z_normals_cache[0, 0], 'z_normals_cache_initial', colormap=True)

            # text embeddings
            if self.cfg.guide.append_direction:
                dirs = data['dir']  # [B,]
                text_z = self.text_z
                text_z_origin = self.text_z_origin
                text_string = self.text_string
                text_string_origin = self.text_string_origin
            else:
                text_z = self.text_z
                text_z_origin = self.text_z_origin
                text_string = self.text_string
                text_string_origin = self.text_string_origin
            logger.info(f'text: {text_string_origin}')
            
            #Making Trimap_original
            update_mask, generate_mask, refine_mask = self.calculate_trimap(rgb_render_raw=rgb_render_raw,
                depth_render=depth_render,
                z_normals=z_normals,
                z_normals_cache=z_normals_cache,
                edited_mask=edited_mask,
                mask=outputs['mask'])

            update_ratio = float(update_mask.sum() / (update_mask.shape[2] * update_mask.shape[3]))
            if self.cfg.guide.reference_texture is not None and update_ratio < 0.01:
                logger.info(f'Update ratio {update_ratio:.5f} is small for an editing step, skipping')
                return

            self.log_train_image(rgb_render * (1 - update_mask), name='masked_input')
            self.log_train_image(rgb_render * refine_mask, name='refine_regions')
            
            # Crop to inner region based on object mask
            min_h, min_w, max_h, max_w = utils.get_nonzero_region(outputs['mask'][0, 0])
            crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]
            cropped_rgb_render = crop(rgb_render)
            cropped_depth_render = crop(depth_render)
            cropped_update_mask = crop(update_mask)

            #Add to list for concatenate
            cropped_renders.append(cropped_rgb_render)
            cropped_depths.append(cropped_depth_render)
            cropped_masks.append(cropped_update_mask)
            
            # Save the required tensors for each view
            rgb_renders.append(rgb_render)
            render_caches.append(render_cache)
            object_masks.append(outputs['mask'])
            update_masks.append(update_mask)
            z_normals_list.append(z_normals)
            z_normals_caches.append(z_normals_cache)

            min_hs.append(min_h)
            min_ws.append(min_w)
            max_hs.append(max_h)
            max_ws.append(max_w)

            self.log_train_image(cropped_rgb_render, name='cropped_input')

        # Find the minimum height and width among the cropped images
        min_height = min([img.shape[2] for img in cropped_renders])
        min_width = min([img.shape[3] for img in cropped_renders])

        # Resize all cropped images to the minimum height and width
        cropped_renders_r = [F.interpolate(img, size=(min_height, min_width), mode='bilinear', align_corners=False) for img in cropped_renders]
        cropped_depths_r = [F.interpolate(img, size=(min_height, min_width), mode='bilinear', align_corners=False) for img in cropped_depths]
        cropped_masks_r = [F.interpolate(img, size=(min_height, min_width), mode='bilinear', align_corners=False) for img in cropped_masks]

        # Concatenate the cropped images into a 2x2 grid
        cropped_rgb_render_2x2 = torch.cat([
            torch.cat([cropped_renders_r[0], cropped_renders_r[1]], dim=3),
            torch.cat([cropped_renders_r[2], cropped_renders_r[3]], dim=3)
        ], dim=2)
        cropped_depth_render_2x2 = torch.cat([
            torch.cat([cropped_depths_r[0], cropped_depths_r[1]], dim=3),
            torch.cat([cropped_depths_r[2], cropped_depths_r[3]], dim=3)
        ], dim=2)
        cropped_update_mask_2x2 = torch.cat([
            torch.cat([cropped_masks_r[0], cropped_masks_r[1]], dim=3),
            torch.cat([cropped_masks_r[2], cropped_masks_r[3]], dim=3)
        ], dim=2)

        # Resize the concatenated image to the required size for the diffusion process
        cropped_rgb_render_2x2 = F.interpolate(cropped_rgb_render_2x2, (1024, 1024), mode='bilinear', align_corners=False)
        cropped_depth_render_2x2 = F.interpolate(cropped_depth_render_2x2, (1024, 1024), mode='bilinear', align_corners=False)
        cropped_update_mask_2x2 = F.interpolate(cropped_update_mask_2x2, (1024, 1024), mode='bilinear', align_corners=False)

        # self.save_vu_image(cropped_depth_render_2x2, 'cropped_depth_render_2x2')
        # self.save_vu_image(cropped_rgb_render_2x2, 'cropped_rgb_render_2x2')
        # self.save_vu_image(cropped_update_mask_2x2, 'cropped_update_mask_2x2')

        checker_mask = None
        if self.paint_step > 1 or self.cfg.guide.initial_texture is not None:
            checker_mask = self.generate_checkerboard(cropped_update_mask_2x2, cropped_update_mask_2x2,
                                                    cropped_update_mask_2x2)
            self.log_train_image(F.interpolate(cropped_rgb_render_2x2, (1024, 1024)) * (1 - checker_mask),
                                'checkerboard_input')
        self.diffusion.use_inpaint = self.cfg.guide.use_inpainting and self.paint_step > 1

        # Diffusion Process with 2x2 grid
        cropped_rgb_output, steps_vis = self.diffusion.img2img_step(
            text_z_origin, 
            cropped_rgb_render_2x2.detach(),
            cropped_depth_render_2x2.detach(),
            guidance_scale=self.cfg.guide.guidance_scale,
            strength=1.0, update_mask=cropped_update_mask_2x2,
            fixed_seed=self.cfg.optim.seed,
            check_mask=checker_mask,
            intermediate_vis=self.cfg.log.vis_diffusion_steps)
        
        self.log_train_image(cropped_rgb_output, name='direct_output_initial')
        self.log_diffusion_steps(steps_vis)

        # Split the 2x2 grid into four separate images
        split_images = torch.split(cropped_rgb_output, 512, dim=2)
        top_left = torch.split(split_images[0], 512, dim=3)[0]
        top_right = torch.split(split_images[0], 512, dim=3)[1]
        bottom_left = torch.split(split_images[1], 512, dim=3)[0]
        bottom_right = torch.split(split_images[1], 512, dim=3)[1]

        # Resize each image to match the size of the corresponding cropped render
        resized_top_left = F.interpolate(top_left, size=(cropped_renders[0].shape[2], cropped_renders[0].shape[3]), mode='bilinear', align_corners=False)
        resized_top_right = F.interpolate(top_right, size=(cropped_renders[1].shape[2], cropped_renders[1].shape[3]), mode='bilinear', align_corners=False)
        resized_bottom_left = F.interpolate(bottom_left, size=(cropped_renders[2].shape[2], cropped_renders[2].shape[3]), mode='bilinear', align_corners=False)
        resized_bottom_right = F.interpolate(bottom_right, size=(cropped_renders[3].shape[2], cropped_renders[3].shape[3]), mode='bilinear', align_corners=False)

        resized_images = [resized_top_left, resized_top_right, resized_bottom_left, resized_bottom_right]

        if UV_MAP:
            image_path = "UVmap_Checker.png"  # Replace with your image file path
            image = Image.open(image_path).convert("RGB")  # Ensure it's RGB
            image_resized = image.resize((1024, 1024), Image.LANCZOS)  # High-quality resizing

            # Convert the resized image to a tensor
            img_array = np.array(image_resized, dtype=np.float32) / 255.0  # Normalize to [0, 1]
            img_array = np.transpose(img_array, (2, 0, 1))  # Shape: [3, 1024, 1024]
            cropped_rgb_output = torch.tensor(img_array).unsqueeze(0)  # Shape: [1, 3, 1024, 1024]

            # Split the 2x2 grid into four separate images (512x512 each)
            split_images = torch.split(cropped_rgb_output, 512, dim=2)  # Split along height: [1, 3, 512, 1024] x 2
            top_left = torch.split(split_images[0], 512, dim=3)[0]      # [1, 3, 512, 512]
            top_right = torch.split(split_images[0], 512, dim=3)[1]     # [1, 3, 512, 512]
            bottom_left = torch.split(split_images[1], 512, dim=3)[0]   # [1, 3, 512, 512]
            bottom_right = torch.split(split_images[1], 512, dim=3)[1]  # [1, 3, 512, 512]

            # Resize each image to match the size of the corresponding cropped render
            resized_top_left = F.interpolate(top_left, size=(cropped_renders[0].shape[2], cropped_renders[0].shape[3]), mode='bilinear', align_corners=False)
            resized_top_right = F.interpolate(top_right, size=(cropped_renders[1].shape[2], cropped_renders[1].shape[3]), mode='bilinear', align_corners=False)
            resized_bottom_left = F.interpolate(bottom_left, size=(cropped_renders[2].shape[2], cropped_renders[2].shape[3]), mode='bilinear', align_corners=False)
            resized_bottom_right = F.interpolate(bottom_right, size=(cropped_renders[3].shape[2], cropped_renders[3].shape[3]), mode='bilinear', align_corners=False)

            # Combine resized tensors into a list
            resized_images = [resized_top_left, resized_top_right, resized_bottom_left, resized_bottom_right]

        # Project back
        # 만들어진 이미지 I_0를 texture atlas T_0 에 project 시켜 보이는 부분을 색칠한다.
        # Extend rgb_output to full image size
        for i, cropped_rgb_out in enumerate(resized_images):
            rgb_output = rgb_renders[i].clone()
            rgb_output[:, :, min_hs[i]:max_hs[i], min_ws[i]:max_ws[i]] = cropped_rgb_out

            fitted_pred_rgb, fitted_z_normals = self.project_back(
                render_cache=render_caches[i], 
                background=background, 
                rgb_output=rgb_output,
                object_mask=object_masks[i], 
                update_mask=update_masks[i], 
                z_normals=z_normals_list[i],
                z_normals_cache=z_normals_caches[i],
                initial=initial,
                index = i)
            # self.save_vu_image(fitted_z_normals, f'project_back_output_{i}_z_normals')
            # self.save_vu_image(fitted_pred_rgb, f'project_back_output_{i}_rgb')

            # TODO: masked_normals 뽑았는데 다시 projection 시키기.
        # self.save_uv_map(self.dataloaders['val'], self.eval_renders_path, 'project_back_output_UV')
        return

    # eval_render dataloader을 입력받아, preds, textures, depths, normals = self.eval_render(data)
    # 여기서 render 
    def eval_render(self, data):
        theta = data['theta']
        phi = data['phi']
        radius = data['radius']
        phi = phi - np.deg2rad(self.cfg.render.front_offset)
        phi = float(phi + 2 * np.pi if phi < 0 else phi)
        dim = self.cfg.render.eval_grid_size
        # data로 이미지 뽑는것 (분홍색, initial)
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                         dims=(dim, dim), background='white')
        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        rgb_render = outputs['image']  # .permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        
        diff = (rgb_render.detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
            self.device)).abs().sum(axis=1)
        # self.save_vu_image(diff, 'eval_render_rgb_(1)_diff')

        uncolored_mask = (diff < 0.1).float().unsqueeze(0)
        # self.save_vu_image(uncolored_mask, 'eval_render_rgb_(2)_uncolored_mask')

        rgb_render = rgb_render * (1 - uncolored_mask) + utils.color_with_shade([0.85, 0.85, 0.85],
        z_normals=z_normals, light_coef=0.3) * uncolored_mask
        # self.save_vu_image(rgb_render, 'eval_render_rgb_(3)_colored')

        outputs_with_median = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
            dims=(dim, dim), use_median=True,
            render_cache=outputs['render_cache'])
        # self.save_vu_image(outputs_with_median['image'], 'eval_render_rgb_(4)_median')

        meta_output = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
            background=torch.Tensor([0, 0, 0]).to(self.device),
            use_meta_texture=True, render_cache=outputs['render_cache'])
        # self.save_vu_image(meta_output['image'], 'eval_render_rgb_(5)_meta')


        pred_z_normals = meta_output['image'][:, :1].detach()
        rgb_render = rgb_render.permute(0, 2, 3, 1).contiguous().clamp(0, 1).detach()
        texture_rgb = outputs_with_median['texture_map'].permute(0, 2, 3, 1).contiguous().clamp(0, 1).detach()
        depth_render = outputs['depth'].permute(0, 2, 3, 1).contiguous().detach()

        return rgb_render, texture_rgb, depth_render, pred_z_normals

    def calculate_trimap(self, rgb_render_raw: torch.Tensor,
        depth_render: torch.Tensor,
        z_normals: torch.Tensor, z_normals_cache: torch.Tensor, edited_mask: torch.Tensor,
        mask: torch.Tensor):
        diff = (rgb_render_raw.detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
            self.device)).abs().sum(axis=1)
        exact_generate_mask = (diff < 0.1).float().unsqueeze(0)

        # Extend mask
        generate_mask = torch.from_numpy(
            cv2.dilate(exact_generate_mask[0, 0].detach().cpu().numpy(), np.ones((19, 19), np.uint8))).to(
            exact_generate_mask.device).unsqueeze(0).unsqueeze(0)

        update_mask = generate_mask.clone()

        object_mask = torch.ones_like(update_mask)
        object_mask[depth_render == 0] = 0
        object_mask = torch.from_numpy(
            cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((7, 7), np.uint8))).to(
            object_mask.device).unsqueeze(0).unsqueeze(0)

        # Generate the refine mask based on the z normals, and the edited mask

        #update mask 기반 refine_mask shape 일치하게
        refine_mask = torch.zeros_like(update_mask)
        # z_normal 부분이 cache + thr 보다 큰 부분만 refine_mask에 1로 채움)(차이가 큰부분)
        refine_mask[z_normals > z_normals_cache[:, :1, :, :] + self.cfg.guide.z_update_thr] = 1
        # initial texture이 없는 부분은 refine하지 않는다.
        if self.cfg.guide.initial_texture is None:
            refine_mask[z_normals_cache[:, :1, :, :] == 0] = 0
        elif self.cfg.guide.reference_texture is not None:
            # edited_mask 부분만 refinement 되도록
            refine_mask[edited_mask == 0] = 0
            refine_mask = torch.from_numpy(
                cv2.dilate(refine_mask[0, 0].detach().cpu().numpy(), np.ones((31, 31), np.uint8))).to(
                mask.device).unsqueeze(0).unsqueeze(0)
            refine_mask[mask == 0] = 0
            # z_normal이 작은 부분(bad angle)은 refinement하지 않는다
            refine_mask[z_normals < 0.4] = 0
        else:
            # Update all regions inside the object
            refine_mask[mask == 0] = 0

        refine_mask = torch.from_numpy(
            cv2.erode(refine_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
            mask.device).unsqueeze(0).unsqueeze(0)
        refine_mask = torch.from_numpy(
            cv2.dilate(refine_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
            mask.device).unsqueeze(0).unsqueeze(0)
        update_mask[refine_mask == 1] = 1

        update_mask[torch.bitwise_and(object_mask == 0, generate_mask == 0)] = 0

        # Visualize trimap
        if self.cfg.log.log_images:
            trimap_vis = utils.color_with_shade(color=[112 / 255.0, 173 / 255.0, 71 / 255.0], z_normals=z_normals)
            trimap_vis[mask.repeat(1, 3, 1, 1) == 0] = 1
            trimap_vis = trimap_vis * (1 - exact_generate_mask) + utils.color_with_shade(
                [255 / 255.0, 22 / 255.0, 67 / 255.0],
                z_normals=z_normals,
                light_coef=0.7) * exact_generate_mask

            shaded_rgb_vis = rgb_render_raw.detach()
            shaded_rgb_vis = shaded_rgb_vis * (1 - exact_generate_mask) + utils.color_with_shade([0.85, 0.85, 0.85],
                                                                                                 z_normals=z_normals,
                                                                                                 light_coef=0.7) * exact_generate_mask

            if self.paint_step > 1 or self.cfg.guide.initial_texture is not None:
                refinement_color_shaded = utils.color_with_shade(color=[91 / 255.0, 155 / 255.0, 213 / 255.0],
                                                                 z_normals=z_normals)
                only_old_mask_for_vis = torch.bitwise_and(refine_mask == 1, exact_generate_mask == 0).float().detach()
                trimap_vis = trimap_vis * 0 + 1.0 * (trimap_vis * (
                        1 - only_old_mask_for_vis) + refinement_color_shaded * only_old_mask_for_vis)
            self.log_train_image(shaded_rgb_vis, 'shaded_input')
            self.log_train_image(trimap_vis, 'trimap')

        return update_mask, generate_mask, refine_mask

    def generate_checkerboard(self, update_mask_inner, improve_z_mask_inner, update_mask_base_inner):
        checkerboard = torch.ones((1, 1, 64 // 2, 64 // 2)).to(self.device)
        # Create a checkerboard grid
        checkerboard[:, :, ::2, ::2] = 0
        checkerboard[:, :, 1::2, 1::2] = 0
        checkerboard = F.interpolate(checkerboard,
                                     (1024, 1024))
        checker_mask = F.interpolate(update_mask_inner, (1024, 1024))
        only_old_mask = F.interpolate(torch.bitwise_and(improve_z_mask_inner == 1,
                                                        update_mask_base_inner == 0).float(), (1024, 1024))
        checker_mask[only_old_mask == 1] = checkerboard[only_old_mask == 1]
        return checker_mask

    # 마지막 Mesh Projection
    def project_back(self, render_cache: Dict[str, Any], background: Any, rgb_output: torch.Tensor,
        object_mask: torch.Tensor, update_mask: torch.Tensor, z_normals: torch.Tensor,
        z_normals_cache: torch.Tensor,
        initial: False, index = 0):
        #cv2.erode : 침식연산 깎으면서, noise 제거, mask가 잘 fit하도록
        object_mask = torch.from_numpy(
            cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
            object_mask.device).unsqueeze(0).unsqueeze(0)
        #initialize render_update_mask with object mask
        render_update_mask = object_mask.clone()

        # update mask가 0인 부분을 render_update_mask에 0으로 채움
        render_update_mask[update_mask == 0] = 0

        # smooth transition between the updated, non-updated regions -> slim했던 mask를 다시 확장(거의 기존 update mask와 비슷)
        blurred_render_update_mask = torch.from_numpy(
            cv2.dilate(render_update_mask[0, 0].detach().cpu().numpy(), np.ones((25, 25), np.uint8))).to(
            render_update_mask.device).unsqueeze(0).unsqueeze(0)

        # 전체 Gaussian blur
        blurred_render_update_mask = utils.gaussian_blur(blurred_render_update_mask, 21, 16)

        # Do not get out of the object(mask setting)-> 다시 슬림해짐
        blurred_render_update_mask[object_mask == 0] = 0

        #strict constraint
        if self.cfg.guide.strict_projection:
            blurred_render_update_mask[blurred_render_update_mask < 0.5] = 0
            # Do not use bad normals
            z_was_better = z_normals + self.cfg.guide.z_update_thr < z_normals_cache[:, :1, :, :]
            blurred_render_update_mask[z_was_better] = 0
        #update
        render_update_mask = blurred_render_update_mask
        # self.save_vu_image(rgb_output, 'rgb_output(project_back_input)')
        self.log_train_image(rgb_output * render_update_mask, 'project_back_input')

        # Update the normals (max value with two)
        z_normals_cache[:, 0, :, :] = torch.max(z_normals_cache[:, 0, :, :], z_normals[:, 0, :, :])
        # self.save_vu_image(z_normals_cache, 'z_normals_cache_updated(project_back_input)')
        # self.save_vu_image(z_normals, 'z_normals(project_back_input)')

        # 현재 상태의 bad z-normal 부분 mask
        threshold = 0.85
        # 1) Background mask: where all 3 channels == 0
        bg_mask = z_normals[:, 0, :, :] == 0 # shape: [1, H, W], dtype=bool

        # 2) Masked part (white where edge normal is bad(edge)
        masked_part = (z_normals[:, 0, :, :] < threshold) & ~bg_mask  # shape [1, 1200, 1200]
        # self.save_tensor_image(bg_mask, 'bg_mask')
        # self.save_tensor_image(masked_part, 'masked_part')

        # 3) Get Extract the masked region from the rgb_output
        masked_part_3ch = masked_part.unsqueeze(1).expand_as(rgb_output)
        combined_mask = torch.where(masked_part_3ch.bool(), rgb_output, background)

        #Adam optimizer for updating model parameter
        optimizer = torch.optim.Adam(self.mesh_model.get_params(), lr=self.cfg.optim.lr, betas=(0.9, 0.99), eps=1e-15)

        for i in tqdm(range(200), desc='fitting mesh colors'):
            optimizer.zero_grad()
            outputs = self.mesh_model.render(background=background,
                                             render_cache=render_cache,
                                             )
            rgb_render = outputs['image']
            if i == 0 and index > 0:
                self.prev_mask = rgb_render.clone()
                color_tensor = torch.tensor(self.mesh_model.default_color, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
                color_image = color_tensor.expand(1, 3, 1200, 1200)
                self.save_tensor_image(color_image, 'mask_filtering')
                self.save_tensor_image(rgb_render, 'rgb_render_mask(학습)')
                self.save_tensor_image(combined_mask, 'rgb_output_mask(BaseImage)')

                diff = torch.abs(self.prev_mask - color_image)  # shape: [1, 3, H, W]
                self.save_tensor_image(diff, '1. diff')
                mask_non_bg = (diff > 0.005).float().sum(dim=1, keepdim=True) > 0
                mask_non_bg = mask_non_bg.float()
                self.save_tensor_image(mask_non_bg, '2. mask_non_bg') # Mask
                self.save_tensor_image(self.prev_mask, 'current_rendered') 
                extracted_with_mask_black = self.prev_mask * mask_non_bg
                self.save_tensor_image(extracted_with_mask_black, '3. extracted_with_mask_black') # Extracted Foreground
                extracted_with_mask = mask_non_bg * self.prev_mask + (1 - mask_non_bg) * 1.0
                self.save_tensor_image(extracted_with_mask, '4. extracted_with_mask_white') # Extracted Foreground
                # -------------------------------BLENDING--------------------------------
                self.save_tensor_image(rgb_output, '3. img_1')
                self.save_tensor_image(extracted_with_mask, '4. img_2')
                # blended_texture = self.blend_texture_patches(rgb_output, extracted_with_mask, mask_non_bg)
                blended_texture = self.alpha_blend_texture(rgb_output, extracted_with_mask, mask_non_bg)
                self.save_tensor_image(blended_texture, '5. blended_texture')
                # --------------------------------------------------------------------------------
                # kernel = torch.ones(9, 9, device=mask_non_bg.device)
                # eroded_mask = kornia.morphology.erosion(mask_non_bg, kernel)
                # self.save_tensor_image(eroded_mask, '3. eroded_mask')
                # eroded_mask = kornia.filters.median_blur(eroded_mask, (5, 5))
                # self.save_tensor_image(eroded_mask, '4. eroded_mask_filtered')
                # extracted_foreground = self.prev_mask * eroded_mask
                # self.save_tensor_image(extracted_foreground, '5. extracted_foreground')
                # # extracted_foreground_filtered = extracted_foreground
                # extracted_foreground_filtered = kornia.filters.median_blur(extracted_foreground, (5, 5))
                # self.save_tensor_image(extracted_foreground_filtered, '6. extracted_foreground_filtered')
                # # combined = extracted_foreground_filtered * eroded_mask * 0.5 + rgb_output * (1 - eroded_mask * 0.5)
                # combined = extracted_foreground_filtered * eroded_mask + rgb_output * (1 - eroded_mask)
                # self.save_tensor_image(combined, '6. combined')
                # rgb_output = combined
                # --------------------------------------------------------------------------------

                # rgb_output = blended_texture.to(self.device)
            
            mask = render_update_mask.flatten()
            masked_pred = rgb_render.reshape(1, rgb_render.shape[1], -1)[:, :, mask > 0]
            if i == 100:
                KERNEL_SIZE = (3, 3)
                rgb_render = kornia.filters.median_blur(rgb_render, KERNEL_SIZE)

            masked_target = rgb_output.reshape(1, rgb_output.shape[1], -1)[:, :, mask > 0]
            masked_mask = mask[mask > 0]

            # L2 loss
            loss = ((masked_pred - masked_target.detach()).pow(2) * masked_mask).mean()

            # current_z_normals [1,3,1200,1200]
            # meta_outputs['mask'] [1,1,1200,1200]
            # z_normals_cache [1,3,1200,1200]
            meta_outputs = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                                  use_meta_texture=True, render_cache=render_cache)
            current_z_normals = meta_outputs['image']
            if i == 150 :
                self.save_tensor_image(current_z_normals, 'meta_outputs(학습)')
                self.save_tensor_image(z_normals_cache, 'z_normals_cache(BaseImage)')
            current_z_mask = meta_outputs['mask'].flatten()
            masked_current_z_normals = current_z_normals.reshape(1, current_z_normals.shape[1], -1)[:, :,
                                       current_z_mask == 1][:, :1] # -> 얘를 학습하는거
            masked_last_z_normals = z_normals_cache.reshape(1, z_normals_cache.shape[1], -1)[:, :,
                                    current_z_mask == 1][:, :1]
            

            loss += (masked_current_z_normals - masked_last_z_normals.detach()).pow(2).mean()

            loss.backward()
            optimizer.step()
        self.save_vu_image(rgb_render, 'rgb_render(project_back_output)')
        # self.save_uv_map(self.dataloaders['val'], self.eval_renders_path, 'UV_map(project_back_output)')

        #optimizer로 업데이트 -> uv_map 저장 -> param 초기화
        if initial == True:
            self.initial_uvmap.append(self.mesh_model.texture_img.detach().cpu())
            self.mesh_model.initialize_params()

            if len(self.initial_uvmap) == 4:
                initial = False

        return rgb_render, current_z_normals

    def log_train_image(self, tensor: torch.Tensor, name: str, colormap=False):
        if self.cfg.log.log_images:
            self.ncount += 1
            if colormap:
                tensor = cm.seismic(tensor.detach().cpu().numpy())[:, :, :3]
            else:
                tensor = einops.rearrange(tensor, '(1) c h w -> h w c').detach().cpu().numpy()
            Image.fromarray((tensor * 255).astype(np.uint8)).save(
                #self.train_renders_path / f'{self.paint_step:04d}_{name}.jpg')
                self.train_renders_path / f'{self.ncount:04d}_{self.paint_step:02d}_{name}.jpg')

    def log_diffusion_steps(self, intermediate_vis: List[Image.Image]):
        if len(intermediate_vis) > 0:
            step_folder = self.train_renders_path / f'{self.paint_step:04d}_diffusion_steps'
            step_folder.mkdir(exist_ok=True)
            for k, intermedia_res in enumerate(intermediate_vis):
                intermedia_res.save(
                    step_folder / f'{k:02d}_diffusion_step.jpg')
                
    def save_tensor_image(self, tensor: torch.Tensor, name: str = None, filepath: str = None):
        """
        Save a float Tensor (values in [0,1]) of shape [1,H,W], [1,1,H,W] or [1,3,H,W] to disk as a PNG.
        """
        self.ncount += 1
        filepath = self.train_renders_path
        name = f"{self.ncount:04d}_{self.paint_step:02d}_{name}.png"
        t = tensor.detach().cpu().squeeze(0)  # → [H,W] or [C,H,W]

        if t.ndim == 2:
            # Tensor is of shape [H, W]
            arr = (t.numpy() * 255).astype(np.uint8)
            img = Image.fromarray(arr, mode="L")
        elif t.ndim == 3:
            if t.shape[0] == 3:
                # Tensor is of shape [3, H, W]
                arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                img = Image.fromarray(arr, mode="RGB")
            elif t.shape[0] == 1:
                # Tensor is of shape [1, H, W] which we treat as grayscale
                arr = (t.squeeze(0).numpy() * 255).astype(np.uint8)
                img = Image.fromarray(arr, mode="L")
            else:
                raise ValueError(f"Unsupported tensor shape {tensor.shape}")
        else:
            raise ValueError(f"Unsupported tensor shape {tensor.shape}")

        img.save(os.path.join(filepath, name))

    def save_numpy_image(self, array: np.ndarray, name: str = None, filepath: str = None):
        """
        Save a float NumPy array (values in [0,1]) of shape [H,W], [1,H,W], [H,W,1], [H,W,3] or [3,H,W] to disk as a PNG.
        """
        self.ncount += 1
        filepath = self.train_renders_path
        name = f"{self.ncount:04d}_{self.paint_step:02d}_{name}.png"

        if array.ndim == 2:
            # Grayscale image of shape [H, W]
            arr = (array * 255).astype(np.uint8)
            img = Image.fromarray(arr, mode="L")
        elif array.ndim == 3:
            if array.shape[0] == 3:
                # RGB image with shape [3, H, W]
                arr = (np.transpose(array, (1, 2, 0)) * 255).astype(np.uint8)
                img = Image.fromarray(arr, mode="RGB")
            elif array.shape[2] == 3:
                # RGB image with shape [H, W, 3]
                arr = (array * 255).astype(np.uint8)
                img = Image.fromarray(arr, mode="RGB")
            elif array.shape[0] == 1:
                # Grayscale image with shape [1, H, W]
                arr = (array.squeeze(0) * 255).astype(np.uint8)
                img = Image.fromarray(arr, mode="L")
            elif array.shape[2] == 1:
                # Grayscale image with shape [H, W, 1]
                arr = (array.squeeze(2) * 255).astype(np.uint8)
                img = Image.fromarray(arr, mode="L")
            else:
                raise ValueError(f"Unsupported array shape {array.shape}")
        else:
            raise ValueError(f"Unsupported array shape {array.shape}")

        img.save(os.path.join(filepath, name))



    def save_vu_image(self, tensor: torch.Tensor, name: str):
        self.ncount += 1
        # Save the image with the new naming format
        vutils.save_image(tensor, self.train_renders_path / f'{self.ncount:04d}_{self.paint_step:02d}_{name}.png')

    def save_uv_map(self, dataloader: DataLoader, save_path: Path, name: str = 'collapsed'):
        self.texturecount += 1
        logger.info(f'Saving UV maps to {save_path}')
        # evel render 을 통해 UV map 뽑아냄 
        _, textures, _, _ = self.eval_render(next(iter(dataloader)))
        texture = tensor2numpy(textures[0])
        if name == 'collapsed':
            Image.fromarray(texture).save(save_path / f"step_{self.paint_step:02d}_{self.texturecount:03d}_collapsed_texture.png")
        elif name == 'initial':
            Image.fromarray(texture).save(save_path / f"step_{self.paint_step:02d}_{self.texturecount:03d}_initial_texture.png")
        else :
            Image.fromarray(texture).save(save_path / f"step_{self.paint_step:02d}_{self.texturecount:03d}_{name}_texture.png")

    def blend_texture_patches(self,
        img1_tensor: torch.Tensor,
        img2_tensor: torch.Tensor,
        patch_mask_tensor: torch.Tensor,
        smooth_kernel_size=15, # Kernel size for smoothing img2 high frequencies (odd number)
        feather_kernel_size=31, # Kernel size for feathering the patch mask (odd number)
        levels=6 # Pyramid levels
        ):
        """
        Blends texture patches from img2 onto img1 naturally.
        Args:
            img1_tensor: Base image (e.g., full rabbit) [1, 3, H, W], float [0,1]
            img2_tensor: Image with texture patches (and black elsewhere on rabbit) [1, 3, H, W], float [0,1]
            patch_mask_tensor: Mask defining patches (1=patch, 0=rabbit, 1=background) [1, 1, H, W], float [0,1]
            smooth_kernel_size: Size of Gaussian kernel to smooth img2.
            feather_kernel_size: Size of Gaussian kernel to feather patch mask.
            levels: Number of Laplacian pyramid levels.
        Returns:
            Blended image tensor [1, 3, H, W], float [0,1]
        """
        # --- Input Validation ---
        if not (isinstance(img1_tensor, torch.Tensor) and
                isinstance(img2_tensor, torch.Tensor) and
                isinstance(patch_mask_tensor, torch.Tensor)):
            raise TypeError("Inputs must be PyTorch tensors.")

        if not (img1_tensor.ndim == 4 and img1_tensor.shape[0] == 1 and img1_tensor.shape[1] == 3 and
                img2_tensor.ndim == 4 and img2_tensor.shape[0] == 1 and img2_tensor.shape[1] == 3 and
                patch_mask_tensor.ndim == 4 and patch_mask_tensor.shape[0] == 1 and patch_mask_tensor.shape[1] == 1):
            raise ValueError("Input tensor dimensions are incorrect. Expecting [1,3,H,W] for images, [1,1,H,W] for mask.")

        if not (img1_tensor.shape[2:] == img2_tensor.shape[2:] == patch_mask_tensor.shape[2:]):
            raise ValueError("Input tensors must have the same Height and Width.")

        if not (smooth_kernel_size % 2 == 1 and feather_kernel_size % 2 == 1):
            raise ValueError("Kernel sizes must be odd")

        # --- Processing ---
        # Use CPU and detach for NumPy conversion
        device = img1_tensor.device # Remember original device
        img1_tensor = img1_tensor.cpu().detach()
        img2_tensor = img2_tensor.cpu().detach()
        patch_mask_tensor = patch_mask_tensor.cpu().detach()


        # 1. Preprocessing: Tensor to NumPy (HxWx3 or HxW, float32, 0-1)
        img1_np = img1_tensor.squeeze(0).permute(1, 2, 0).numpy().astype(np.float32)
        img2_np = img2_tensor.squeeze(0).permute(1, 2, 0).numpy().astype(np.float32)
        # Mask needs special handling: HxW, float32, 0-1
        patch_mask_raw_np = patch_mask_tensor.squeeze(0).squeeze(0).numpy().astype(np.float32)

        H, W = img1_np.shape[:2]
        print(f"NumPy shapes: img1={img1_np.shape}, img2={img2_np.shape}, patch_mask_raw={patch_mask_raw_np.shape}")

        # 2. Smooth img2 to remove high frequencies
        img2_smoothed_np = cv2.GaussianBlur(img2_np, (smooth_kernel_size, smooth_kernel_size), 0)
        # Ensure result didn't go out of bounds (blurring can slightly exceed 0/1)
        img2_smoothed_np = np.clip(img2_smoothed_np, 0, 1)
        print(f"img2 smoothed shape: {img2_smoothed_np.shape}")

        # 3. Prepare Blend Mask (Isolate Patches + Feather)
        # 3a. Create silhouette mask (rabbit=1, background=0)
        #     Assuming white background is approx 1.0 in all channels
        # Use img1's shape to define the silhouette reliably
        img1_gray = cv2.cvtColor(img1_np, cv2.COLOR_BGR2GRAY) # Or use any channel
        # Threshold: Where image is NOT white (e.g., < 0.98), it's part of the rabbit
        # Adjust threshold (0.98) if background isn't perfectly white (1.0)
        silhouette_mask_np = (img1_gray < 0.98).astype(np.float32)
        print(f"Silhouette mask shape: {silhouette_mask_np.shape}, min={silhouette_mask_np.min()}, max={silhouette_mask_np.max()}")

        # 3b. Isolate patches (mask = 1 only for patches, 0 elsewhere)
        #     Multiply raw patch mask (patch=1, rabbit=0, bg=1) by silhouette (rabbit=1, bg=0)
        patch_mask_isolated_np = patch_mask_raw_np * silhouette_mask_np
        print(f"Isolated patch mask shape: {patch_mask_isolated_np.shape}, min={patch_mask_isolated_np.min()}, max={patch_mask_isolated_np.max()}")
        # Check if isolation worked (max should be <= 1)
        if patch_mask_isolated_np.max() > 1.001: # Allow for slight floating point inaccuracies
            print(f"Warning: Isolated patch mask max {patch_mask_isolated_np.max()} > 1.0, clipping.")
            patch_mask_isolated_np = np.clip(patch_mask_isolated_np, 0, 1)


        # 3c. Feather the isolated patch mask
        patch_mask_feathered_np = cv2.GaussianBlur(patch_mask_isolated_np, (feather_kernel_size, feather_kernel_size), 0)
        # Add channel dim: HxWx1
        patch_mask_feathered_np = patch_mask_feathered_np[:, :, np.newaxis]
        # Ensure mask stays in [0, 1] after blur
        patch_mask_feathered_np = np.clip(patch_mask_feathered_np, 0, 1)
        print(f"Feathered patch mask shape: {patch_mask_feathered_np.shape}, min={patch_mask_feathered_np.min()}, max={patch_mask_feathered_np.max()}")


        # 4. Laplacian Blending
        # --- Build Gaussian Pyramids ---
        gpA = [img1_np] # Base image
        gpB = [img2_smoothed_np] # Smoothed patch image
        gpM = [patch_mask_feathered_np] # Feathered mask (1=use img2)

        for i in range(levels):
            # Check dimensions before pyrDown
            if gpA[i].shape[0] < 2 or gpA[i].shape[1] < 2:
                print(f"Stopping pyramid generation at level {i} due to small dimensions.")
                levels = i # Adjust effective levels
                break

            gpA.append(cv2.pyrDown(gpA[i]))
            gpB.append(cv2.pyrDown(gpB[i]))

            # Need to handle mask pyrDown carefully
            prev_mask = gpM[i]
            next_mask = cv2.pyrDown(prev_mask)
            if next_mask.ndim == 2:
                next_mask = next_mask[:, :, np.newaxis]

            # Handle potential shape mismatch if pyrDown rounds differently
            expected_h, expected_w = gpA[-1].shape[:2]
            current_h, current_w = next_mask.shape[:2]
            if current_h != expected_h or current_w != expected_w:
                print(f"Resizing mask at level {i+1} from {next_mask.shape[:2]} to {(expected_h, expected_w)}")
                # Resize needs target size as (width, height)
                next_mask = cv2.resize(next_mask, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)
                if next_mask.ndim == 2: next_mask = next_mask[:,:,np.newaxis] # Add dim back if lost

            gpM.append(next_mask)

        effective_levels = len(gpA) - 1 # Actual levels generated

        # --- Build Laplacian Pyramids ---
        lpA = []
        lpB = []
        for i in range(effective_levels):
            target_h, target_w = gpA[i].shape[:2]
            try:
                GE_A = cv2.pyrUp(gpA[i+1], dstsize=(target_w, target_h))
                GE_B = cv2.pyrUp(gpB[i+1], dstsize=(target_w, target_h))
            except cv2.error as e:
                print(f"pyrUp failed at level {i}. Shapes: gpA[i]={gpA[i].shape}, gpA[i+1]={gpA[i+1].shape}. Error: {e}")
                # Fallback to resize if pyrUp constraint fails
                GE_A = cv2.resize(gpA[i+1], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                GE_B = cv2.resize(gpB[i+1], (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            # Ensure shapes match exactly after potential resize/pyrUp issues
            if GE_A.shape != gpA[i].shape: GE_A = cv2.resize(GE_A, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            if GE_B.shape != gpB[i].shape: GE_B = cv2.resize(GE_B, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            L_A = cv2.subtract(gpA[i], GE_A)
            L_B = cv2.subtract(gpB[i], GE_B)
            lpA.append(L_A)
            lpB.append(L_B)

        # Add the coarsest Gaussian level
        lpA.append(gpA[effective_levels])
        lpB.append(gpB[effective_levels])

        # --- Blend Laplacian Pyramids ---
        LS = []
        print(f"\n--- Blending {effective_levels+1} Laplacian Levels ---")
        for i in range(effective_levels + 1):
            la = lpA[i]
            lb = lpB[i]
            gm = gpM[i] # Mask corresponds directly to level i

            # Ensure mask is broadcastable (e.g., HxWx1 -> HxWx3)
            if gm.ndim == 3 and gm.shape[2] == 1 and la.ndim == 3 and la.shape[2] == 3:
                gm = np.tile(gm, (1, 1, 3))
            # Ensure spatial dimensions match, resize mask if necessary (shouldn't happen ideally)
            elif gm.shape[:2] != la.shape[:2]:
                print(f"Warning: Resizing mask at blend level {i} from {gm.shape[:2]} to {la.shape[:2]}")
                gm = cv2.resize(gm, (la.shape[1], la.shape[0]), interpolation=cv2.INTER_LINEAR)
                if gm.ndim == 2: gm = gm[:,:,np.newaxis]
                if gm.ndim == 3 and gm.shape[2] == 1 and la.ndim == 3 and la.shape[2] == 3: gm = np.tile(gm, (1,1,3))


            # Standard formula: Use img1 (la) where mask=0, use smoothed img2 (lb) where mask=1
            ls = la * (1.0 - gm) + lb * gm
            LS.append(ls)

        # --- Reconstruct ---
        blended_np = LS[effective_levels] # Start with coarsest level
        for i in range(effective_levels - 1, -1, -1): # Loop down to finest level index 0
            target_h, target_w = LS[i].shape[:2]
            try:
                upsampled_level = cv2.pyrUp(blended_np, dstsize=(target_w, target_h))
            except cv2.error as e:
                print(f"pyrUp failed during reconstruction at level {i}. Current shape: {blended_np.shape}, Target shape: {LS[i].shape[:2]}. Error: {e}")
                # Fallback to resize
                upsampled_level = cv2.resize(blended_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            # Ensure exact shape match before adding
            if upsampled_level.shape != LS[i].shape:
                print(f"Resizing upsampled level at reconstruction level {i} from {upsampled_level.shape[:2]} to {LS[i].shape[:2]}")
                upsampled_level = cv2.resize(upsampled_level, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            blended_np = cv2.add(upsampled_level, LS[i])

        # Final clipping
        blended_np = np.clip(blended_np, 0, 1)
        print(f"\nFinal Blended NumPy shape: {blended_np.shape}")

        # 5. Postprocessing: NumPy to Tensor
        # Ensure output dtype matches input tensor dtype if needed
        output_dtype = img1_tensor.dtype
        blended_tensor_out = torch.from_numpy(blended_np).permute(2, 0, 1).unsqueeze(0).to(dtype=output_dtype, device=device)


        # Optional: Apply silhouette mask again for clean background if needed
        # silhouette_mask_np_hwc = silhouette_mask_np[:, :, np.newaxis] # Ensure HxWx1
        # silhouette_mask_tensor = torch.from_numpy(silhouette_mask_np_hwc).permute(2, 0, 1).unsqueeze(0).to(dtype=output_dtype, device=device) # Shape [1,1,H,W]
        # black_background = torch.zeros_like(blended_tensor_out, device=device)
        # blended_tensor_out = blended_tensor_out * silhouette_mask_tensor + black_background * (1.0 - silhouette_mask_tensor)
        return blended_tensor_out
    
    def alpha_blend_texture(
            self,
            img1_tensor: "torch.Tensor", # Type hint requires torch
            img2_tensor: "torch.Tensor", # Type hint requires torch
            patch_mask_tensor: "torch.Tensor", # Type hint requires torch
            blend_weight=0.5, # Weight/Opacity of img2 texture (0.0 to 1.0)
            feather_kernel_size=31 # Kernel size for feathering the patch mask (odd number)
        ):
        """
        Adds texture from img2 onto img1 using alpha blending.
        Args:
            img1_tensor: Base image [1, 3, H, W], float [0,1]
            img2_tensor: Texture image (patches=texture, rabbit=white, bg=white) [1, 3, H, W], float [0,1]
            patch_mask_tensor: Mask (1=patch, 0=rabbit, 1=background) [1, 1, H, W], float [0,1]
            blend_weight: How strongly to blend img2's patches (0=img1 only, 1=img2 replaces img1 in patches).
            feather_kernel_size: Size of Gaussian kernel to feather patch mask edges.
        Returns:
            Blended image tensor [1, 3, H, W], float [0,1]
        """
        # --- Input Validation ---
        # (Add tensor type/shape/dimension checks as in previous functions if running with PyTorch)
        if not (0.0 <= blend_weight <= 1.0):
            raise ValueError("blend_weight must be between 0.0 and 1.0")
        if not (feather_kernel_size % 2 == 1):
            raise ValueError("feather_kernel_size must be odd")

        # --- Processing ---
        # Use CPU and detach for NumPy conversion
        # device = img1_tensor.device # Remember original device if using torch
        img1_tensor_cpu = img1_tensor.cpu().detach()
        img2_tensor_cpu = img2_tensor.cpu().detach()
        patch_mask_tensor_cpu = patch_mask_tensor.cpu().detach()

        # 1. Preprocessing: Tensor to NumPy
        img1_np = img1_tensor_cpu.squeeze(0).permute(1, 2, 0).numpy().astype(np.float32)
        img2_np = img2_tensor_cpu.squeeze(0).permute(1, 2, 0).numpy().astype(np.float32)
        patch_mask_raw_np = patch_mask_tensor_cpu.squeeze(0).squeeze(0).numpy().astype(np.float32)
        H, W = img1_np.shape[:2]
        print(f"NumPy shapes: img1={img1_np.shape}, img2={img2_np.shape}, patch_mask_raw={patch_mask_raw_np.shape}")

        # 2. Prepare Alpha Mask
        # 2a. Create silhouette mask (rabbit=1, background=0) from img1
        img1_gray = cv2.cvtColor(img1_np, cv2.COLOR_BGR2GRAY)
        silhouette_mask_np = (img1_gray < 0.98).astype(np.float32) # Adjust 0.98 if needed

        # 2b. Isolate patches (mask = 1 only for patches, 0 elsewhere)
        patch_mask_isolated_np = patch_mask_raw_np * silhouette_mask_np
        if patch_mask_isolated_np.max() > 1.001:
            patch_mask_isolated_np = np.clip(patch_mask_isolated_np, 0, 1)
        print(f"Isolated patch mask shape: {patch_mask_isolated_np.shape}, min={patch_mask_isolated_np.min()}, max={patch_mask_isolated_np.max()}")

        # 2c. Feather the isolated patch mask
        patch_mask_feathered_np = cv2.GaussianBlur(patch_mask_isolated_np, (feather_kernel_size, feather_kernel_size), 0)
        patch_mask_feathered_np = np.clip(patch_mask_feathered_np, 0, 1)
        print(f"Feathered patch mask shape: {patch_mask_feathered_np.shape}, min={patch_mask_feathered_np.min()}, max={patch_mask_feathered_np.max()}")

        # 2d. Scale by blend_weight and prepare for broadcasting (add channel dim HxWx1 -> HxWx3)
        alpha_np = patch_mask_feathered_np * blend_weight
        alpha_np_3channel = np.tile(alpha_np[:, :, np.newaxis], (1, 1, 3)) # Create HxWx3 alpha mask
        print(f"Final Alpha mask shape: {alpha_np_3channel.shape}, min={alpha_np_3channel.min()}, max={alpha_np_3channel.max()}")


        # 3. Alpha Blend
        # Result = img1 * (1 - alpha) + img2 * alpha
        blended_np = img1_np * (1.0 - alpha_np_3channel) + img2_np * alpha_np_3channel

        # 4. Clip
        blended_np = np.clip(blended_np, 0, 1)
        print(f"\nFinal Blended NumPy shape: {blended_np.shape}")

        # 5. Postprocessing: NumPy to Tensor
        output_dtype = img1_tensor.dtype # Match input dtype
        blended_tensor_out = torch.from_numpy(blended_np).permute(2, 0, 1).unsqueeze(0).to(dtype=output_dtype, device=self.device)

        # --- Since torch is not available, return the NumPy result for inspection ---
        return blended_tensor_out
