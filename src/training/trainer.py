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

from src import utils
from src.configs.train_config import TrainConfig
from src.models.textured_mesh import TexturedMeshModel
from src.basemodels.stable_diffusion_depth import BlendedLatentDiffusionSDXL

from src.training.views_dataset import ViewsDataset, MultiviewDataset
from src.utils import make_path, tensor2numpy
import time # texture generation time calculation
import os
from pathlib import Path

import torchvision.utils as vutils
from typing import Optional

class TEXTure:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.paint_step = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ncount = 0
        self.texturecount = 0
        self.image_count = 0  # Counter for valid images
        self.initialized_count = 0
        self.initial_uvmap = []
        self.z_normals_cache_list = [None]
        

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
        self.mesh_model = self.init_mesh_model()
        self.diffusion = self.init_diffusion()
        self.text_z, self.text_string, self.text_z_origin, self.text_string_origin = self.calc_text_embeddings()
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
        diffusion_model = BlendedLatentDiffusionSDXL(model_name = "stabilityai/stable-diffusion-xl-base-1.0", device=self.device)

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
            text_string_origin = ref_text.split('{', 1)[0].strip().rstrip(',')
            text_z_origin = None
            for d in self.view_dirs:
                text = ref_text.format(d)
                text_string.append(text)
                text_z = text_string
                logger.info(text)
                negative_prompt = None
                logger.info(negative_prompt)
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
        self.mesh_model.train()

        pbar = tqdm(total=len(self.dataloaders['train']), initial=self.paint_step,
                    bar_format='{desc}: {percentage:3.0f}% painting step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for data in self.dataloaders['train']:
            if self.paint_step == 0:
                self.paint_step += 1
                pbar.update(1)
                self.paint_viewpoint_initial(data)
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
                Image.fromarray(pred).save(save_path / f"step_{self.paint_step:05d}_{i:04d}_rgb.jpg")
                Image.fromarray((cm.seismic(normals[0, 0].cpu().numpy())[:, :, :3] * 255).astype(np.uint8)).save(
                    save_path / f'{self.paint_step:04d}_{i:04d}_normals_cache.jpg')
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

    def paint_viewpoint(self, data: Dict[str, Any], initial=False):
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
            background = F.interpolate(self.back_im.unsqueeze(0),
                                       (self.cfg.render.train_grid_size, self.cfg.render.train_grid_size),
                                       mode='bilinear', align_corners=False)

        # Render from viewpoint
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, background='white') # background -> 'white'
        # TODO: 기존 : background=backgroud
        render_cache = outputs['render_cache']
        rgb_render_raw = outputs['image']  # Render where missing values have special color
        depth_render = outputs['depth']
        # Render again with the median value to use as rgb, we shouldn't have color leakage, but just in case
        outputs = self.mesh_model.render(background='white',
                                         render_cache=render_cache, use_median=self.paint_step > 1) # background -> 'white'
        rgb_render = outputs['image']
        # Render meta texture map
        meta_output = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                             use_meta_texture=True, render_cache=render_cache)

        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        normals_all = outputs['normals'].clamp(0, 1)
        z_normals_cache = meta_output['image'].clamp(0, 1) # [1, 3, 1200, 1200]
        edited_mask = meta_output['image'].clamp(0, 1)[:, 1:2] # [1, 1, 1200, 1200]

        #self.log_train_image(rgb_render_raw, '첫번째 렌더링 결과 이미지')
        #self.log_train_image(rgb_render, '두번째 렌더링 결과 이미지') # [1, 3, 1200, 1200]
        
        direction = None
        # text embeddings
        if self.cfg.guide.append_direction:
            dirs = data['dir']  # [B,]
            print(self.text_z[dirs])
            direction = self.text_z[dirs].split(',')[-1].strip()
            #text_z = self.text_z[dirs]
            #text_string = self.text_string[dirs]
            text_z = self.text_string_origin
            text_string = self.text_string
        else:
            text_z = self.text_z
            text_string = self.text_string
        text_z += ", photography, realistic, "
        if direction is not None:
            text_z += direction
        logger.info(f'text: {text_z}')

        update_mask, generate_mask, refine_mask = self.calculate_trimap(rgb_render_raw=rgb_render_raw,
                                                                        depth_render=depth_render,
                                                                        z_normals=z_normals,
                                                                        z_normals_cache=z_normals_cache,
                                                                        edited_mask=edited_mask,
                                                                        mask=outputs['mask']) # [1, 1, 1200, 1200]
        uv_features = render_cache['uv_features']
        refine_mask = self.make_render_mask(uv_features)
        self.log_train_image(refine_mask, name='Refine mask gray contour')
        self.log_train_image(generate_mask, name='Generate mask')

        update_mask = refine_mask.clone()
        update_mask[generate_mask == 1] = 1
        """update_mask = torch.from_numpy(
            cv2.dilate(update_mask[0, 0].detach().cpu().numpy(), np.ones((10, 10), np.uint8))).to(
            update_mask.device).unsqueeze(0).unsqueeze(0)"""
        
        update_ratio = float(update_mask.sum() / (update_mask.shape[2] * update_mask.shape[3]))
        if self.cfg.guide.reference_texture is not None and update_ratio < 0.01:
            logger.info(f'Update ratio {update_ratio:.5f} is small for an editing step, skipping')
            return
        
        self.log_train_image(update_mask, name='Update mask gray')
        self.log_train_image(rgb_render, name='RGB render')
        self.log_train_image(rgb_render * (update_mask), name='Update mask')

        # Crop to inner region based on object mask
        min_h, min_w, max_h, max_w = utils.get_nonzero_region(outputs['mask'][0, 0])
        crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]
        cropped_rgb_render = crop(rgb_render)
        cropped_depth_render = crop(depth_render) # [1, 3, 849, 849] -> 시점마다 바뀜뀜
        cropped_update_mask = crop(update_mask)
        cropped_refine_mask = crop(refine_mask)
        cropped_generate_mask = crop(generate_mask)
        cropped_normal = crop(normals_all)
        self.log_train_image(cropped_rgb_render, name='Diffusion 입력값')
        self.log_train_image(cropped_normal, name='crop normal')

        checker_mask = None
        """if self.paint_step > 1 or self.cfg.guide.initial_texture is not None:
            checker_mask = self.generate_checkerboard(crop(update_mask), crop(refine_mask), # [1, 1, 512, 512]
                                                      crop(generate_mask))
            self.log_train_image(F.interpolate(cropped_rgb_render, (1024, 1024)) * (1 - checker_mask),
                                 'Crop된 이미지에서 checker_mask 적용')"""
        self.diffusion.use_inpaint = self.cfg.guide.use_inpainting and self.paint_step > 1

        cropped_rgb_output, steps_vis = self.diffusion.img2img_step(text_z, cropped_rgb_render.detach(),
                                                                    cropped_depth_render.detach(),
                                                                    guidance_scale=self.cfg.guide.guidance_scale,
                                                                    strength=1.0, update_mask=cropped_update_mask,
                                                                    fixed_seed=self.cfg.optim.seed,
                                                                    check_mask=checker_mask,
                                                                    intermediate_vis=self.cfg.log.vis_diffusion_steps, 
                                                                    generate_mask=cropped_generate_mask,
                                                                    paint_step=self.paint_step, path=self.train_renders_path,
                                                                    z_normal = cropped_normal, refine_mask = cropped_refine_mask)
        self.log_train_image(cropped_rgb_output, name='difusion latent 결과')
        self.log_diffusion_steps(steps_vis)

        cropped_rgb_output = F.interpolate(cropped_rgb_output,
                                           (cropped_rgb_render.shape[2], cropped_rgb_render.shape[3]),
                                           mode='bilinear', align_corners=False)

        # Extend rgb_output to full image size
        rgb_output = rgb_render.clone()
        rgb_output[:, :, min_h:max_h, min_w:max_w] = cropped_rgb_output

        # Project back
        object_mask = outputs['mask']
        fitted_pred_rgb, _ = self.project_back(render_cache=render_cache, background='white', rgb_output=rgb_output,
                                               object_mask=object_mask, update_mask=update_mask, z_normals=z_normals,
                                               z_normals_cache=z_normals_cache, initial=initial) # background -> 'white'

        return
    def paint_viewpoint_initial(self, data: Dict[str, Any], UV_MAP=False, initial=False):
        logger.info(f'--- Painting step #{self.paint_step} ---')
        theta, phi, radius = data['theta'], data['phi'], data['radius']
        phi_angles = [0, np.pi/2, 3*np.pi/2, np.pi]
        
        # Diffusion의 입력으로 들어가는 리스트
        cropped_renders = []
        cropped_depths = []
        cropped_masks = []
        cropped_generate_masks = []
        cropped_normals = []
        cropped_normals_modify = []

        # Project back에 필요한 리스트
        render_caches = []
        object_masks = []
        update_masks = []
        z_normals_list = []
        z_normals_caches = []
        rgb_renders = []

        # Resize 할 때 필요한 리스트
        min_hs = []
        min_ws = []
        max_hs = []
        max_ws = []      

        for phi in phi_angles:
            phi = phi - np.deg2rad(self.cfg.render.front_offset)
            phi = float(phi + 2 * np.pi if phi < 0 else phi)
            logger.info(f'Painting from theta: {theta}, phi: {phi}, radius: {radius}')

            # Set background image
            if self.cfg.guide.use_background_color:
                background = torch.Tensor([0, 0.8, 0]).to(self.device)
            else:
                background = F.interpolate(self.back_im.unsqueeze(0),
                                        (self.cfg.render.train_grid_size, self.cfg.render.train_grid_size),
                                        mode='bilinear', align_corners=False)

            # Render from viewpoint
            outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius, background='white')
            render_cache = outputs['render_cache']
            rgb_render_raw = outputs['image'] 
            depth_render = outputs['depth']
            
            outputs = self.mesh_model.render(background='white',
                                            render_cache=render_cache, use_median=self.paint_step > 1)
            rgb_render = outputs['image']
            meta_output = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                                use_meta_texture=True, render_cache=render_cache)

            z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
            normals_all = outputs['normals']
            normals_all = (normals_all * 0.5 + 0.5)
            normals_all_modify = normals_all.clamp(0.0, 1.0)
            z_normals_cache = meta_output['image'].clamp(0, 1)
            #self.log_train_image(z_normals_cache, name=f'z_normals_cache_{phi}', colormap=True)

            self.update_meta_texture_normals(render_cache, z_normals, z_normals_cache)
            edited_mask = meta_output['image'].clamp(0, 1)[:, 1:2]

            # text embeddings
            if self.cfg.guide.append_direction:
                dirs = data['dir']  # [B,]
                text_z = self.text_z[dirs]
                text_string = self.text_string[dirs]
            else:
                text_z = self.text_z
                text_string = self.text_string
            text_z = text_z.split(',')[0].strip()
            text_z += ", photography, realistic"
            logger.info(f'text: {text_z}')
            
            #Making Trimap_original
            update_mask, generate_mask, refine_mask  = self.calculate_trimap(rgb_render_raw=rgb_render_raw,
                depth_render=depth_render,
                z_normals=z_normals,
                z_normals_cache=z_normals_cache,
                edited_mask=edited_mask,
                mask=outputs['mask'])

            update_ratio = float(update_mask.sum() / (update_mask.shape[2] * update_mask.shape[3]))
            if self.cfg.guide.reference_texture is not None and update_ratio < 0.01:
                logger.info(f'Update ratio {update_ratio:.5f} is small for an editing step, skipping')
                return

            # Crop 
            min_h, min_w, max_h, max_w = utils.get_nonzero_region(outputs['mask'][0, 0])
            crop = lambda x: x[:, :, min_h:max_h, min_w:max_w]

            # Diffusion의 입력으로 들어가는 리스트
            cropped_renders.append(crop(rgb_render))
            cropped_depths.append(crop(depth_render))
            8
            cropped_masks.append(crop(update_mask))
            cropped_generate_masks.append(crop(generate_mask))
            cropped_normals.append(crop(normals_all))
            cropped_normals_modify.append(crop(normals_all_modify))
            
            # Project back에 필요한 리스트
            render_caches.append(render_cache)            
            object_masks.append(outputs['mask'])
            update_masks.append(update_mask)
            z_normals_list.append(z_normals)
            z_normals_caches.append(z_normals_cache)
            rgb_renders.append(rgb_render)
            
            # Reshape 할 때 필요한 리스트
            min_hs.append(min_h)
            min_ws.append(min_w)
            max_hs.append(max_h)
            max_ws.append(max_w)


        # Crop된 이미지의 높이와 너비 찾기기
        max_height = max([img.shape[2] for img in cropped_renders])
        max_width = max([img.shape[3] for img in cropped_renders])

        cropped_rgb_render_2x2 = self.crop_and_grid(max_height, max_width, cropped_renders)
        cropped_depth_render_2x2 = self.crop_and_grid(max_height, max_width, cropped_depths)
        cropped_update_mask_2x2 = self.crop_and_grid(max_height, max_width, cropped_masks)
        cropped_generate_mask_2x2 = self.crop_and_grid(max_height, max_width, cropped_generate_masks)
        cropped_all_normal_2x2 = self.crop_and_grid(max_height, max_width, cropped_normals)
        cropped_all_normal_2x2_modify = self.crop_and_grid(max_height, max_width, cropped_normals_modify)

        mn = cropped_all_normal_2x2.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        mx = cropped_all_normal_2x2.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        cropped_all_normal_2x2 = (cropped_all_normal_2x2 - mn) / (mx - mn + 1e-8)

        self.log_train_image(cropped_all_normal_2x2, 'cropped_all_normal_2x2')
        self.log_train_image(cropped_all_normal_2x2_modify, 'cropped_all_normal_2x2_modify')
        self.log_train_image(cropped_depth_render_2x2, 'cropped_depth_render_2x2')
        self.diffusion.use_inpaint = self.cfg.guide.use_inpainting and self.paint_step > 1

        # Diffusion Process with 2x2 grid
        # TODO: img2img_step mask 사용하는것 BLD 수정 필요 - 295번째 줄 참조
        cropped_rgb_output, steps_vis = self.diffusion.img2img_step(
            text_z, 
            cropped_rgb_render_2x2.detach(),
            cropped_depth_render_2x2.detach(),
            guidance_scale=self.cfg.guide.guidance_scale,
            strength=1.0, update_mask=cropped_update_mask_2x2,
            fixed_seed=self.cfg.optim.seed,
            check_mask=None,
            intermediate_vis=self.cfg.log.vis_diffusion_steps,
            generate_mask=cropped_generate_mask_2x2,
            paint_step=self.paint_step, path=self.train_renders_path, z_normal=cropped_all_normal_2x2)
        
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
        
        prev_uv_map = None
        prev_uv_map_mask = None
        overlap_final = None
        for i, cropped_rgb_out in enumerate([
            resized_top_left, resized_top_right,
            resized_bottom_left, resized_bottom_right
        ]):
            # RGB output 구성
            rgb_output = rgb_renders[i].clone()
            rgb_output[:, :, min_hs[i]:max_hs[i], min_ws[i]:max_ws[i]] = cropped_rgb_out

            # 이전까지의 텍스처로 현재 view를 렌더링 → trimap 계산용
            rendered_after_texture = self.mesh_model.render(
                theta=theta, phi=phi_angles[i], radius=radius,
                background='white', use_meta_texture=False
            )

            fitted_pred_rgb, _ = self.project_back(
                render_cache=render_caches[i],
                background='white',
                rgb_output=rgb_output,
                object_mask=object_masks[i],
                update_mask=update_masks[i],
                z_normals=z_normals_list[i],
                z_normals_cache=z_normals_caches[i],
                initial=initial,
                check_value=i
            )

            self.save_uv_map(self.dataloaders['val'], self.eval_renders_path, 'collapsed')

            data_batch = next(iter(self.dataloaders['val']))
            _, textures, _, _ = self.eval_render(data_batch)
            current_uv_map = textures[0]  

            if i == 0:
                prev_uv_map = current_uv_map
                prev_uv_map_mask = self.mesh_model.create_basecolor_exclusion_mask_from_tensor(current_uv_map)
                #self.log_train_image(prev_uv_map_mask, name=f'prev_uv_map_mask_{i}')
                #self.log_train_image(prev_uv_map, name=f'prev_uv_map_{i}')
            else:
                current_uv_map_mask = self.mesh_model.create_basecolor_exclusion_mask_from_tensor(current_uv_map)
                #self.log_train_image(current_uv_map_mask, name=f'current_uv_map_mask_{i}')

                diff = torch.abs(current_uv_map - prev_uv_map)
                diff_mean = diff.mean(dim=2, keepdim=True)
                binary_mask = (diff_mean > 0.01).float()
                change_mask = binary_mask.permute(2, 0, 1).unsqueeze(0).cpu()
                self.log_train_image(change_mask, name=f'change_mask_{i}')

                overlap_uv_map = prev_uv_map_mask * change_mask
                #self.log_train_image(overlap_uv_map, name=f'overlap_uv_map_{i}')

                if overlap_final is None:
                    overlap_final = overlap_uv_map
                else:
                    overlap_final = ((overlap_final + overlap_uv_map) > 0)
                self.log_train_image(overlap_final, name=f'overlap_final_{i}')

                uv_features = render_caches[i]['uv_features']  # [1, C, H_uv, W_uv]
                project_mask = self.mesh_model.project_uv_mask_to_view(uv_features, overlap_uv_map.to(self.device))
                self.log_train_image(project_mask, name=f'project_mask_{i}')
                #self.log_train_image(rgb_render_raw *(1 - project_mask), name=f'project_mask_rgb_{i}')

                prev_uv_map_mask = ((current_uv_map_mask + prev_uv_map_mask) > 0)
                prev_uv_map = current_uv_map
                #self.log_train_image(prev_uv_map_mask, name=f'prev_uv_map_mask_{i}')

        return

    def eval_render(self, data):
        theta = data['theta']
        phi = data['phi']
        radius = data['radius']
        phi = phi - np.deg2rad(self.cfg.render.front_offset)
        phi = float(phi + 2 * np.pi if phi < 0 else phi)
        dim = self.cfg.render.eval_grid_size
        outputs = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                         dims=(dim, dim), background='white')
        z_normals = outputs['normals'][:, -1:, :, :].clamp(0, 1)
        rgb_render = outputs['image']  # .permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        diff = (rgb_render.detach() - torch.tensor(self.mesh_model.default_color).view(1, 3, 1, 1).to(
            self.device)).abs().sum(axis=1)
        uncolored_mask = (diff < 0.1).float().unsqueeze(0)
        rgb_render = rgb_render * (1 - uncolored_mask) + utils.color_with_shade([0.85, 0.85, 0.85], z_normals=z_normals,
                                                                                light_coef=0.3) * uncolored_mask

        outputs_with_median = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                                     dims=(dim, dim), use_median=True,
                                                     render_cache=outputs['render_cache'])

        meta_output = self.mesh_model.render(theta=theta, phi=phi, radius=radius,
                                             background=torch.Tensor([0, 0, 0]).to(self.device),
                                             use_meta_texture=True, render_cache=outputs['render_cache'])
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
        exact_generate_mask = (diff < 0.1).float().unsqueeze(0) # exact_generate_mask: 색칠이 거의 안된 부분분
    
        # 마스크 확장장
        generate_mask = exact_generate_mask.clone()
        """generate_mask = torch.from_numpy(
            cv2.dilate(exact_generate_mask[0, 0].detach().cpu().numpy(), np.ones((19, 19), np.uint8))).to(
            exact_generate_mask.device).unsqueeze(0).unsqueeze(0)"""

        update_mask = generate_mask.clone()

        object_mask = torch.ones_like(update_mask)
        object_mask[depth_render == 0] = 0
        object_mask = torch.from_numpy(
            cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((7, 7), np.uint8))).to(
            object_mask.device).unsqueeze(0).unsqueeze(0)
        
        # background mask 생성
        background_mask = 1 - object_mask
        background_mask = torch.from_numpy(
            cv2.dilate(background_mask[0, 0].detach().cpu().numpy(), np.ones((15, 15), np.uint8))
            ).to(object_mask.device).unsqueeze(0).unsqueeze(0)  # Dilate를 통해 배경 확장
        
        # Generate the refine mask based on the z normals, and the edited mask

        refine_mask = torch.zeros_like(update_mask)
        refine_mask[z_normals > z_normals_cache[:, :1, :, :] + self.cfg.guide.z_update_thr] = 1
        if self.cfg.guide.initial_texture is None:
            refine_mask[z_normals_cache[:, :1, :, :] == 0] = 0
        elif self.cfg.guide.reference_texture is not None:
            refine_mask[edited_mask == 0] = 0
            refine_mask = torch.from_numpy(
                cv2.dilate(refine_mask[0, 0].detach().cpu().numpy(), np.ones((31, 31), np.uint8))).to(
                mask.device).unsqueeze(0).unsqueeze(0)
            refine_mask[mask == 0] = 0
            # Don't use bad angles here
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
            #self.log_train_image(shaded_rgb_vis, 'shaded_input')
            #self.log_train_image(trimap_vis, 'trimap')

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

    def project_back(self, render_cache: Dict[str, Any], background: Any, rgb_output: torch.Tensor,
                     object_mask: torch.Tensor, update_mask: torch.Tensor, z_normals: torch.Tensor,
                     z_normals_cache: torch.Tensor, initial: False, check_value=0):
        object_mask = torch.from_numpy(
            cv2.erode(object_mask[0, 0].detach().cpu().numpy(), np.ones((5, 5), np.uint8))).to(
            object_mask.device).unsqueeze(0).unsqueeze(0)
        render_update_mask = object_mask.clone()

        render_update_mask[update_mask == 0] = 0

        blurred_render_update_mask = torch.from_numpy(
            cv2.dilate(render_update_mask[0, 0].detach().cpu().numpy(), np.ones((25, 25), np.uint8))).to(
            render_update_mask.device).unsqueeze(0).unsqueeze(0)
        blurred_render_update_mask = utils.gaussian_blur(blurred_render_update_mask, 21, 16)

        # Do not get out of the object
        blurred_render_update_mask[object_mask == 0] = 0

        if self.cfg.guide.strict_projection:
            blurred_render_update_mask[blurred_render_update_mask < 0.5] = 0
            # Do not use bad normals
            #z_was_better = z_normals + self.cfg.guide.z_update_thr < z_normals_cache[:, :1, :, :]
            z_was_better = z_normals + 0 < z_normals_cache[:, :1, :, :]
            blurred_render_update_mask[z_was_better] = 0

        render_update_mask = blurred_render_update_mask
        self.log_train_image(rgb_output * render_update_mask, f'project_back_input_{check_value}')

        # Update the normals
        z_normals_cache[:, 0, :, :] = torch.max(z_normals_cache[:, 0, :, :], z_normals[:, 0, :, :])

        optimizer = torch.optim.Adam(self.mesh_model.get_params(), lr=self.cfg.optim.lr, betas=(0.9, 0.99),
                                     eps=1e-15)
        for _ in tqdm(range(200), desc='fitting mesh colors'):
            optimizer.zero_grad()
            outputs = self.mesh_model.render(background='white',
                                             render_cache=render_cache) # background -> 'white'
            rgb_render = outputs['image']

            mask = render_update_mask.flatten()
            masked_pred = rgb_render.reshape(1, rgb_render.shape[1], -1)[:, :, mask > 0]
            masked_target = rgb_output.reshape(1, rgb_output.shape[1], -1)[:, :, mask > 0]
            masked_mask = mask[mask > 0]
            loss = ((masked_pred - masked_target.detach()).pow(2) * masked_mask).mean()

            meta_outputs = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                                  use_meta_texture=True, render_cache=render_cache)
            current_z_normals = meta_outputs['image']
            current_z_mask = meta_outputs['mask'].flatten()
            masked_current_z_normals = current_z_normals.reshape(1, current_z_normals.shape[1], -1)[:, :,
                                       current_z_mask == 1][:, :1]
            masked_last_z_normals = z_normals_cache.reshape(1, z_normals_cache.shape[1], -1)[:, :,
                                    current_z_mask == 1][:, :1]
            loss += (masked_current_z_normals - masked_last_z_normals.detach()).pow(2).mean()
            loss.backward()
            optimizer.step()
        self.save_uv_map(self.dataloaders['val'], self.eval_renders_path, 'UV_map(project_back_output)')

        return rgb_render, current_z_normals

    def update_meta_texture_normals(self,
                                    render_cache: Dict[str, Any],
                                    target_z_normals: torch.Tensor,
                                    z_normals_cache: torch.Tensor):
        # 2) optimizer: meta_texture_img 만
        z_normals_cache[:, 0, :, :] = torch.max(z_normals_cache[:, 0, :, :], target_z_normals[:, 0, :, :])
        optimizer = torch.optim.Adam([self.mesh_model.meta_texture_img], lr=self.cfg.optim.lr, betas=(0.9,0.99), eps=1e-15)

        for _ in range(200):
            optimizer.zero_grad()
            meta_outputs = self.mesh_model.render(background=torch.Tensor([0, 0, 0]).to(self.device),
                                                  use_meta_texture=True, render_cache=render_cache)
            current_z_normals = meta_outputs['image']
            current_z_mask = meta_outputs['mask'].flatten()

            masked_current_z_normals = current_z_normals.reshape(1, current_z_normals.shape[1], -1)[:, :,
                                       current_z_mask == 1][:, :1]
            masked_last_z_normals = z_normals_cache.reshape(1, z_normals_cache.shape[1], -1)[:, :,
                                    current_z_mask == 1][:, :1]
            loss = (masked_current_z_normals - masked_last_z_normals.detach()).pow(2).mean()
            loss.backward()
            optimizer.step()

    def log_train_image(self, tensor: torch.Tensor, name: str, colormap=False):
        if self.cfg.log.log_images:
            # 텐서가 4차원([1, C, H, W])인 경우
            if tensor.dim() == 4:
                np_img = einops.rearrange(tensor, '(1) c h w -> h w c').detach().cpu().numpy()
            # 3차원([H, W, C])인 경우
            elif tensor.dim() == 3:
                np_img = tensor.detach().cpu().numpy()
            else:
                raise ValueError("Unsupported tensor shape.")
                
            # colormap 옵션이 True면 colormap 적용 (이미지에 3채널 결과로 변환)
            if colormap:
                # 그레이스케일 2D로 만들기
                if np_img.ndim == 3 and np_img.shape[-1] == 1:
                    gray = np.squeeze(np_img, axis=-1)
                else:
                    gray = np.mean(np_img, axis=-1)
                # colormap 적용 → RGBA
                colored = cm.seismic(gray)
                # RGB만 취하기
                np_img = colored[:, :, :3]
            
            # 만약 결과 이미지가 그레이스케일 (채널이 1)이라면 채널 차원을 제거합니다.
            if np_img.ndim == 3 and np_img.shape[-1] == 1:
                np_img = np.squeeze(np_img, axis=-1)  # shape: [H, W]
            
            # 0~1 범위의 값을 [0,255]로 스케일링 후 정수형으로 변환
            np_img = np.clip(np_img, 0, 1) * 255
            np_img = np_img.astype(np.uint8)
            
            # 만약 np_img가 2차원이면 grayscale, 그렇지 않으면 RGB
            mode = "L" if np_img.ndim == 2 else "RGB"
            Image.fromarray(np_img, mode=mode).save(
                self.train_renders_path / f'{self.paint_step:04d}_{name}.jpg')

    def save_uv_map(self, dataloader: DataLoader, save_path: Path, name: str = 'collapsed'):
        self.texturecount += 1
        logger.info(f'Saving UV maps to {save_path}')
        _, textures, _, _ = self.eval_render(next(iter(dataloader)))
        texture = tensor2numpy(textures[0])
        if name == 'collapsed':
            Image.fromarray(texture).save(save_path / f"step_{self.paint_step:02d}_{self.texturecount:03d}_collapsed_texture.png")
        elif name == 'initial':
            Image.fromarray(texture).save(save_path / f"step_{self.paint_step:02d}_{self.texturecount:03d}_initial_texture.png")
        else :
            Image.fromarray(texture).save(save_path / f"step_{self.paint_step:02d}_{self.texturecount:03d}_{name}_texture.png")          

    def log_diffusion_steps(self, intermediate_vis: List[Image.Image]):
        if len(intermediate_vis) > 0:
            step_folder = self.train_renders_path / f'{self.paint_step:04d}_diffusion_steps'
            step_folder.mkdir(exist_ok=True)
            for k, intermedia_res in enumerate(intermediate_vis):
                intermedia_res.save(
                    step_folder / f'{k:02d}_diffusion_step.jpg')
            
    def load_binary_mask(self, image_path, threshold=0.5):
        # 이미지 불러오기 및 grayscale 변환
        img = Image.open(image_path).convert("L")  # "L" 모드는 grayscale
        # NumPy 배열로 변환 (값은 0~255)
        arr = np.array(img, dtype=np.float32)
        # 0~1 범위로 정규화
        arr = arr / 255.0
        # 임계값으로 이진화: threshold보다 큰 값은 1, 아니면 0
        # binary_arr = (arr > threshold).astype(np.float32)
        # 4차원 텐서로 변환: [1, 1, H, W]
        mask_tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        return mask_tensor

    def crop_and_grid(self, max_height, max_width, value):
        value = [F.interpolate(img, size=(max_height, max_width), mode='bilinear', align_corners=False) for img in value]

        output = torch.cat([
            torch.cat([value[0], value[1]], dim=3),
            torch.cat([value[2], value[3]], dim=3)
        ], dim=2)  

        output = F.interpolate(output, (1024,1024), mode='bilinear', align_corners=False)   
        return output

    def make_render_mask(self, uv_features):
        refine_mask_path = os.path.join(self.train_renders_path / f'0001_overlap_final_3.jpg')
        refine_mask = self.load_binary_mask(refine_mask_path)
        refine_mask = self.mesh_model.project_uv_mask_to_view(uv_features, refine_mask.to(self.device))
        self.log_train_image(refine_mask, name='Refine mask gray')
        image_path = f'/home/mmai6k_01/TEXTurePaper/experiments/rabbit_2/vis/train/000{self.paint_step}_Refine mask gray.jpg'
        image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(image_gray, dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, color=1, thickness=10)
        mask_visual = mask 
        mask_visual_expanded = mask_visual[np.newaxis, np.newaxis, :, :]
        refine_mask = torch.from_numpy(mask_visual_expanded).to(refine_mask.device).float()
        self.log_train_image(refine_mask, name='Refine mask gray contour')
        
        return refine_mask