from diffusers import (AutoencoderKL, UNet2DConditionModel, ControlNetModel, 
                       StableDiffusionXLControlNetPipeline, StableDiffusionXLPipeline,
                       PNDMScheduler, LMSDiscreteScheduler, DDIMScheduler, StableDiffusionXLImg2ImgPipeline)
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from huggingface_hub import hf_hub_download
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    logging,
    DPTFeatureExtractor, DPTForDepthEstimation,
)

######################################################################

# suppress partial model loading warning
from src import utils
from src.utils import seed_everything
logging.set_verbosity_error()

######################################################################

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tqdm.auto import tqdm
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from diffusers.utils import load_image
import einops
import sys
sys.path.append("/workspace/TEXTurePaper/src")
from src.basemodels.controlnet_union import ControlNetModel_Union

class BlendedLatentDiffusionSDXL(StableDiffusionXLPipeline):
    def __init__(self, model_name, device="cuda"):
        # StableDiffusionXLPipeline의 모든 모델을 직접 로드
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to(device)
        pipeline.vae = pipeline.vae.float()
        # 부모 클래스 초기화 (vae, unet, text_encoder 등의 파라미터 상속)
        super().__init__(
            vae=pipeline.vae,
            text_encoder=pipeline.text_encoder,
            text_encoder_2=pipeline.text_encoder_2,
            tokenizer=pipeline.tokenizer,
            tokenizer_2=pipeline.tokenizer_2,
            unet=pipeline.unet,
            scheduler=pipeline.scheduler,
            feature_extractor=pipeline.feature_extractor,
        )

        # ControlNet 추가
        self.controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16
        ).to(device)
        self.normal_controlnet = ControlNetModel_Union.from_pretrained("xinsir/controlnet-union-sdxl-1.0", torch_dtype=torch.float16, use_safetensors=True).to(device)
        self.inpaint_unet =  UNet2DConditionModel.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", subfolder='unet').to(device)
        

    def parameters(self):
        """ StableDiffusionXLPipeline 내부 모듈들의 parameters() 호출 """
        params = []
        if hasattr(self, "unet"):
            params += list(self.unet.parameters())
        if hasattr(self, "vae"):
            params += list(self.vae.parameters())
        if hasattr(self, "text_encoder"):
            params += list(self.text_encoder.parameters())
        if hasattr(self, "text_encoder_2"):
            params += list(self.text_encoder_2.parameters())
        if hasattr(self, "controlnet"):
            params += list(self.controlnet.parameters())

        return params
    
    
    def img2img_step(self, text, inputs, depth_mask, guidance_scale=100, strength=1.0,
                     num_inference_steps=50, update_mask=None, latent_mode=False, check_mask=None,
                     fixed_seed=None, check_mask_iters=0.5, intermediate_vis=False,
                     generate_mask=None, paint_step=None, path=None, z_normal=None, refine_mask=None):
        intermediate_results = []
        device="cuda"

        # print(depth_mask.shape) #[1 1 H W]
        # print(inputs.shape) #[1 3 H W]
        def sample(latents, depth_mask, controlnet_depth_mask, strength, num_inference_steps, update_mask=None, check_mask=None,
                   masked_latents=None, prompt_embeds=None, pooled_prompt_embeds=None, negative_prompt_embeds=None,
                   negative_pooled_prompt_embeds=None, generate_mask=None, prompt=None, normal=None,
                   blending_percentage=0.0, height=1024, width=1024, guidance_scale=5.0):
            
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            # Noise to target latent
            max_noise_timestep = int(len(timesteps) * blending_percentage)
            init_image = latents
            latents = self.scheduler.add_noise(
                latents, torch.randn_like(latents), self.scheduler.timesteps[max_noise_timestep].unsqueeze(0)
            )
            depth_mask = torch.cat([depth_mask] * 2) # [2 1 128 128]

            # 0. Default height and width to unet
            height = height or self.default_sample_size * self.vae_scale_factor
            width = width or self.default_sample_size * self.vae_scale_factor

            original_size = (height, width)
            target_size = (height, width)
            crops_coords_top_left = (0,0)
            
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim
            add_text_embeds = pooled_prompt_embeds
            add_time_ids = self._get_add_time_ids(
                original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype, text_encoder_projection_dim=text_encoder_projection_dim
            )

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

            prompt_embeds = prompt_embeds.to(device)
            add_text_embeds = add_text_embeds.to(device)
            add_time_ids = add_time_ids.to(device).repeat(1, 1)

            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

            with torch.autocast('cuda'):
                for i, t in tqdm(enumerate(timesteps[max_noise_timestep:])):
                    use_bld = i >= num_inference_steps * 0.0 and i <= num_inference_steps * 1.0
                    use_inpaint_step = i >= num_inference_steps * 0.0 and i <= num_inference_steps * 1.0


                    latent_model_input = torch.cat([latents] * 2) 
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    added_cond_kwargs_original = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                    if self.use_multi_controlnet:
                        union_control_type = torch.Tensor([0, 1, 0, 0, 1, 0])
                    else:
                        union_control_type = torch.Tensor([0, 0, 0, 0, 1, 0])
                    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids, \
                    "control_type":union_control_type.reshape(1, -1).to(device, dtype=prompt_embeds.dtype).repeat(1 * 1 * 2, 1)}

                    cond_depth_mask = torch.cat([controlnet_depth_mask] * 3, dim=1)
                    cond_depth_mask = torch.cat([cond_depth_mask] * 2, dim=0)
                    input_normal = torch.cat([normal] * 2)
                    #cond_normal = torch.cat([normal] * 3, dim=1)

                    if self.use_multi_controlnet:
                        controlnet_down_features, controlnet_mid_features = self.normal_controlnet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            controlnet_cond_list=[0, cond_depth_mask, 0, 0, input_normal, 0],
                            conditioning_scale=0.5,
                            #guess_mode=False,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )
                    else:
                        controlnet_down_features, controlnet_mid_features = self.normal_controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        controlnet_cond_list=[0, 0, 0, 0, input_normal, 0],
                        conditioning_scale=0.5,
                        #guess_mode=False,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )

                    """controlnet_latent = self.controlnet(
                                latent_model_input,  # 현재 Latent 입력
                                t,  # 현재 timestep
                                encoder_hidden_states=prompt_embeds,  # Text Condition
                                controlnet_cond=cond_depth_mask,  # ControlNet에 Depth 정보를 입력
                                added_cond_kwargs=added_cond_kwargs,
                                return_dict=True  # NOTE : v2(250303) Dict 로 반환해서 down block, mid block 반환
                            )  # ControlNet이 변형 한 latent 출력

                    controlnet_scale = 0.5  
                    controlnet_down_features = [
                        feature * controlnet_scale for feature in controlnet_latent['down_block_res_samples']
                    ]
                    controlnet_mid_features = controlnet_latent['mid_block_res_sample'] * controlnet_scale"""

                    if not self.use_inpaint:
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=None,
                            added_cond_kwargs=added_cond_kwargs,
                            down_block_additional_residuals=controlnet_down_features,  # ControlNet Downblock feature
                            mid_block_additional_residual=controlnet_mid_features,  # ControlNet Midblock feature
                            return_dict=False,
                        )[0]
                    else:
                        if self.use_inpaint_unet and use_inpaint_step:
                            latent_mask = torch.cat([update_mask] * 2)
                            latent_image = torch.cat([masked_latents] * 2)
                            latent_model_input_inpaint = torch.cat([latent_model_input, latent_mask, latent_image], dim=1)
                            noise_pred_inpaint = self.inpaint_unet(
                                latent_model_input_inpaint,
                                t,
                                encoder_hidden_states=prompt_embeds,
                                cross_attention_kwargs=None,
                                added_cond_kwargs=added_cond_kwargs_original,
                                #down_block_additional_residuals=controlnet_down_features,  # ControlNet Downblock feature
                                #mid_block_additional_residual=controlnet_mid_features,  # ControlNet Midblock feature
                                return_dict=False,
                            )[0]
                            noise_pred = noise_pred_inpaint
                        else:
                            noise_pred = self.unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=prompt_embeds,
                                cross_attention_kwargs=None,
                                added_cond_kwargs=added_cond_kwargs,
                                down_block_additional_residuals=controlnet_down_features,  # ControlNet Downblock feature
                                mid_block_additional_residual=controlnet_mid_features,  # ControlNet Midblock feature
                                return_dict=False,
                            )[0]
                    # perform guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    extra_step_kwargs = self.prepare_extra_step_kwargs(None, 0.0)
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                    # BLD 시작
                    # Source latents와의 블렌딩
                    noise_source_latents = self.scheduler.add_noise(
                        init_image, torch.randn_like(latents), t.unsqueeze(0)
                    ) # [2, 4, 128, 128]

                    if self.use_inpaint_unet:
                        if use_bld and self.use_inpaint and not use_inpaint_step:
                            latents = latents * (update_mask) + noise_source_latents * (1 - update_mask)
                    else:
                        if use_bld and self.use_inpaint:
                            latents = latents * (update_mask) + noise_source_latents * (1 - update_mask)

                    latents = latents.to(torch.float32)
            
            return latents

        controlnet_depth_mask = F.interpolate(depth_mask, size=(1024, 1024), mode='bicubic',
                                   align_corners=False)
        depth_mask = F.interpolate(depth_mask, size=(128, 128), mode='bicubic',
                                   align_corners=False)
        controlnet_normal = None
        if z_normal is not None:
            controlnet_normal =  F.interpolate(z_normal, size=(1024, 1024), mode='bicubic',
                                    align_corners=False)                             
        self.use_inpaint_unet = False
        self.use_multi_controlnet = False
        masked_latents = None
        if inputs is None:
            latents = None
        elif latent_mode:
            latents = inputs
        else:         
            inputs_1024 = F.interpolate(inputs, (1024, 1024), mode='bilinear', align_corners=False)
            if refine_mask is not None:
                refine_1024 = F.interpolate(refine_mask, (1024, 1024), mode='bilinear', align_corners=False)

                inputs_blurred_1024 = self.apply_gaussian_blur(inputs_1024, kernel_size=61, sigma=10.0)
                inputs_1024 = refine_1024 * inputs_blurred_1024 + (1 - refine_1024) * inputs_1024
                self.log_train_image(refine_1024, name='refine_mask', path=path, paint_step=paint_step)
                self.log_train_image(inputs_1024, name='refine_image', path=path, paint_step=paint_step)

            if self.use_inpaint and self.use_inpaint_unet:
                update_mask_1024 = F.interpolate(update_mask, (1024, 1024))
                #masked_latents = inputs_1024 * (update_mask_1024 < 0.5) + 0.5 * (update_mask_1024 >= 0.5)
                update_mask_1024 = (update_mask_1024 > 0.5).float()
                keep_mask        = 1 - update_mask_1024
                masked_latents = inputs_1024 * keep_mask
                self.log_train_image(masked_latents, name='masked_latents', path=path, paint_step=paint_step)
                masked_latents = self.encode_imgs(masked_latents)
            latents = self.encode_imgs(inputs_1024)


        if update_mask is not None:
            update_mask = F.interpolate(update_mask, (128, 128), mode='nearest')
        if check_mask is not None:
            check_mask = F.interpolate(check_mask, (128, 128), mode='nearest')
        if generate_mask is not None:
            generate_mask = F.interpolate(generate_mask, (128, 128), mode='nearest')

        depth_mask = 2.0 * (depth_mask - depth_mask.min()) / (depth_mask.max() - depth_mask.min()) - 1.0
        #controlnet_depth_mask = 2.0 * (controlnet_depth_mask - controlnet_depth_mask.min()) / (controlnet_depth_mask.max() - controlnet_depth_mask.min()) - 1.0
        mn = controlnet_depth_mask.min()
        mx = controlnet_depth_mask.max()
        controlnet_depth_mask = (controlnet_depth_mask - mn) / (mx - mn + 1e-8)
        controlnet_depth_mask = controlnet_depth_mask.clamp(0.0, 1.0)
        self.config.force_zeros_for_empty_prompt = False # get_text_embeds와 동일하게 작동시키기 위해서
        # 3. Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=text,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=None,
            #negative_prompt='Unrealistic, cropped, blurry, low quality, bad anatomy, cartoon, anime, 3D, painting, NSFW.',
            negative_prompt_2=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=None,
        )

        with torch.no_grad():
            target_latents = sample(latents, depth_mask, controlnet_depth_mask, strength=strength, num_inference_steps=num_inference_steps,
                                    update_mask=update_mask, check_mask=check_mask, masked_latents=masked_latents, prompt_embeds=prompt_embeds,
                                    pooled_prompt_embeds=pooled_prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds, generate_mask=generate_mask,
                                    prompt=text, normal=controlnet_normal)
            target_rgb = self.decode_latents(target_latents)
            """if self.use_inpaint:
                self.log_train_image(target_rgb, name='decode_image', path=path, paint_step=paint_step)
                pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
                pipe = pipe.to("cuda")
                init_image = load_image(f"{path}/{paint_step:04d}_decode_image.jpg").convert('RGB')
                #image = pipe(prompt=text, image=inputs_1024).images[0]
                image = pipe(prompt=text, image=init_image).images[0]
                target_rgb = TF.to_tensor(image).unsqueeze(0).to("cuda") """

        if latent_mode:
            return target_rgb, target_latents
        else:
            return target_rgb, intermediate_results

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.13025
        return latents
    
    def decode_latents(self, latents):
        latents = 1 / 0.13025 * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def log_train_image(self, tensor: torch.Tensor, name: str, path=None, paint_step=None, colormap=False):
        if tensor.dim() == 4:
            np_img = einops.rearrange(tensor, '(1) c h w -> h w c').detach().cpu().numpy()
        # 3차원([H, W, C])인 경우
        elif tensor.dim() == 3:
            np_img = tensor.detach().cpu().numpy()
        else:
            raise ValueError("Unsupported tensor shape.")
            
        # colormap 옵션이 True면 colormap 적용 (이미지에 3채널 결과로 변환)
        if colormap:
            np_img = cm.seismic(np_img)[:, :, :3]
        
        # 만약 결과 이미지가 그레이스케일 (채널이 1)이라면 채널 차원을 제거합니다.
        if np_img.ndim == 3 and np_img.shape[-1] == 1:
            np_img = np.squeeze(np_img, axis=-1)  # shape: [H, W]
        
        # 0~1 범위의 값을 [0,255]로 스케일링 후 정수형으로 변환
        np_img = np.clip(np_img, 0, 1) * 255
        np_img = np_img.astype(np.uint8)
        
        # 만약 np_img가 2차원이면 grayscale, 그렇지 않으면 RGB
        mode = "L" if np_img.ndim == 2 else "RGB"
        Image.fromarray(np_img, mode=mode).save(
            path / f'{paint_step:04d}_{name}.jpg')

    def apply_gaussian_blur(self, x, kernel_size=5, sigma=1.0):
        """
        x: [B, C, H, W] 텐서
        kernel_size: 홀수 정수 (예: 5)
        sigma: 가우시안 분포의 표준편차
        """
        channels = x.shape[1]
        # Gaussian kernel 생성: 2D 가우시안
        coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - kernel_size // 2
        x_coords = coords.view(1, -1)  # shape: (1, kernel_size)
        y_coords = coords.view(-1, 1)  # shape: (kernel_size, 1)
        kernel = torch.exp(-(x_coords**2 + y_coords**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()  # 정규화
        
        # 커널 모양을 [1, 1, kernel_size, kernel_size]로 만들고, 채널 수 만큼 복제
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
        
        # convolution에 사용할 padding 계산
        padding = kernel_size // 2
        blurred = F.conv2d(x, weight=kernel, padding=padding, groups=channels)
        return blurred