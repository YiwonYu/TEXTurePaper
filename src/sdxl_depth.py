#=========================Need to be modified========================#
# Class SDXL(nn.Module):
# def __init__
# def load_concept
# def get_text_embeds
# def img2img_step
# def produce_latent
# def decode_latent
# def encode_imgs
# def get_timesteps
# def prompt_to_img
# img2img_single_step
# def train_step
# def produce_latent
#====================================================================#

from diffusers import (AutoencoderKL, UNet2DConditionModel, ControlNetModel, 
                       StableDiffusionXLControlNetPipeline,
                       )
from diffusers.schedulers import (PNDMScheduler, DDIMScheduler, 
                                  LMSDiscreteScheduler, DPMSolverMultistepScheduler)
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tqdm.auto import tqdm
import cv2
from PIL import Image

######################################################################
# NOTE : SDXL 1,024 image -> 128 latent (64 pixel per latent)
######################################################################

class SDXL(nn.Module):
    def __init__(self, device, model_name="stabilityai/stable-diffusion-xl-base-1.0", concept_name=None, concept_path=None,
                 latent_mode=True, min_timestep=0.02, max_timestep=0.98, no_noise=False,
                 use_inpaint=False, use_autodepth=False):
        super().__init__()

        try:
            with open('./TOKEN', 'r') as f:
                self.token = f.read().replace('\n', '')  # remove the last \n!
                logger.info(f'loaded hugging face access token from ./TOKEN!')
        except FileNotFoundError as e:
            self.token = True
            logger.warning(
                f'try to load hugging face access token from the default place, make sure you have run `huggingface-cli login`.')

        self.device = device
        self.latent_mode = latent_mode
        self.no_noise = no_noise
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * min_timestep)
        self.max_step = int(self.num_train_timesteps * max_timestep)
        self.use_inpaint = use_inpaint
        self.use_autodepth = use_autodepth
        base_model = "stabilityai/stable-diffusion-xl-base-1.0"
        depth_model = "diffusers/controlnet-depth-sdxl-1.0"
        inpaint_model = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

        logger.info(f'loading model {base_model}...')
        logger.info(f'loading model {depth_model}...')
        logger.info(f'loading model {inpaint_model}...')

        # 0. ControlNet Depth
        self.depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
        self.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
        self.controlnet = ControlNetModel.from_pretrained(
            depth_model,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to(self.device)
        
        # 1. load vae for each model
        self.base_vae = AutoencoderKL.from_pretrained(base_model, subfolder='vae', use_auth_token=self.token).to(self.device)
        if self.use_inpaint:
            self.inpaint_vae = AutoencoderKL.from_pretrained(inpaint_model, subfolder='vae', use_auth_token=self.token).to(self.device)

        # 2. load tokenizers and text encoder
        # NOTE: We need to load two tokenizers and two text encoders for each model
        # NOTE : base = stable-diffusion-xl-base-1.0, inpaint = stable-diffusion-xl-1.0-inpainting-0.1

        self.base_tokenizer_1 = CLIPTokenizer.from_pretrained(base_model, subfolder='tokenizer', use_auth_token=self.token)
        self.base_tokenizer_2 = CLIPTokenizer.from_pretrained(base_model, subfolder='tokenizer_2', use_auth_token=self.token)
        self.base_text_encoder_1 = CLIPTextModel.from_pretrained(base_model, subfolder='text_encoder', use_auth_token=self.token,
                                                                 torch_dtype=torch.float16).to(self.device)
        self.base_text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(base_model, subfolder='text_encoder_2', use_auth_token=self.token,
                                                                 torch_dtype=torch.float16).to(self.device)

        if self.use_inpaint:
            self.inpaint_tokenizer_1 = CLIPTokenizer.from_pretrained(inpaint_model, subfolder='tokenizer', use_auth_token= self.token)
            self.inpaint_text_encoder_1 = CLIPTextModel.from_pretrained(inpaint_model, subfolder='text_encoder', use_auth_token=self.token,
                                                                     torch_dtype=torch.float16).to(self.device)
            self.inpaint_tokenizer_2 = CLIPTokenizer.from_pretrained(inpaint_model, subfolder='tokenizer_2', use_auth_token=self.token)
            self.inpaint_text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(inpaint_model, subfolder='text_encoder_2', use_auth_token=self.token,
                                                                     torch_dtype=torch.float16).to(self.device)
        
        # 3. Image encoder and processor
        self.image_encoder = None
        self.image_processor = None

        # 4. UNet for each model
        # NOTE : base = stable-diffusion-xl-base-1.0, inpaint = stable-diffusion-xl-1.0-inpainting-0.1
        self.base_unet = UNet2DConditionModel.from_pretrained(base_model, subfolder='unet', 
                                                              use_auth_token=self.token).to(self.device)
        if self.use_inpaint:
            self.inpaint_unet = UNet2DConditionModel.from_pretrained(inpaint_model, subfolder='unet', 
                                                                    use_auth_token=self.token).to(self.device)
        
        # 5. Scheduler
        # NOTE : Scheduler가 주는 영향 미미 타 원인 파악(250305)

        self.PNDMS_scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                       num_train_timesteps=self.num_train_timesteps, steps_offset=1,
                                       skip_prk_steps=True)
        
        self.DDIM_scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                       num_train_timesteps=self.num_train_timesteps, steps_offset=1)
        
        self.LMSDiscrete_scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                        num_train_timesteps=self.num_train_timesteps, steps_offset=1)
        self.DPMSolver_scheduler = DPMSolverMultistepScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                        num_train_timesteps=self.num_train_timesteps, steps_offset=1, 
                                        use_karras_sigmas=True)
        
        # NOTE : Diffusers version update required, Version mismatch for diffusers 0.21.3
        # self.DPMSolver_scheduler = DPMSolverMultistepScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        #                                 num_train_timesteps=self.num_train_timesteps, steps_offset=1, 
        #                                 use_karras_sigmas=True, sde_type="sde-dpmsolver++",
        #                                 euler_at_final=True, use_lu_lambdas=True)
        
        self.scheduler = self.DPMSolver_scheduler
        
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        # 6. Load concept -> Not used in this project
        # if concept_name is not None:
        #     self.load_concept(concept_name, concept_path)
        logger.info(f'\t successfully loaded stable diffusion!')

    #====================================================================#
    # train_config : concept_name Usage
    # A Textual-Inversion concept to use
    # concept_name: Optional[str] = None
    #====================================================================#
    # def load_concept(self, concept_name, concept_path=None):
    #     # NOTE: No need for both name and path, they are the same!
    #     if concept_path is None:
    #         repo_id_embeds = f"sd-concepts-library/{concept_name}"
    #         learned_embeds_path = hf_hub_download(repo_id=repo_id_embeds, filename="learned_embeds.bin")
    #         # token_path = hf_hub_download(repo_id=repo_id_embeds, filename="token_identifier.txt")
    #         # with open(token_path, 'r') as file:
    #         #     placeholder_token_string = file.read()
    #     else:
    #         learned_embeds_path = concept_path

    #     loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")

    #     # separate token and the embeds
    #     for trained_token in loaded_learned_embeds:
    #         # trained_token = list(loaded_learned_embeds.keys())[0]
    #         print(f'Loading token for {trained_token}')
    #         embeds = loaded_learned_embeds[trained_token]

    #         # cast to dtype of text_encoder
    #         dtype = self.text_encoder.get_input_embeddings().weight.dtype
    #         embeds.to(dtype)

    #         # add the token in tokenizer
    #         token = trained_token
    #         num_added_tokens = self.tokenizer.add_tokens(token)
    #         if num_added_tokens == 0:
    #             raise ValueError(
    #                 f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer.")

    #         # resize the token embeddings
    #         self.text_encoder.resize_token_embeddings(len(self.tokenizer))

    #         # get the id for the token and assign the embeds
    #         token_id = self.tokenizer.convert_tokens_to_ids(token)
    #         self.text_encoder.get_input_embeddings().weight.data[token_id] = embeds

    def get_text_embeds(self, prompt, negative_prompt=None):
        base_text_encoders = [self.base_text_encoder_1, self.base_text_encoder_2]
        base_tokenizers = [self.base_tokenizer_1, self.base_tokenizer_2]
        prompt_embeds_list = []
        print(len(prompt))
        # NOTE : https://github.com/mikhail-bot/stable-diffusion-negative-prompts
        if negative_prompt is None:
            negative_prompt = ['blurry, low contrast, low resolution, cloned face, watermark, text, bad anatomy, distorted, low quality, grainy'] * len(prompt)

        for tokenizer, text_encoder in zip(base_tokenizers, base_text_encoders):
            with torch.no_grad():
                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False)
                
                # 최종 layer embeddings 사용
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds[-1][-2]
                
                # 크기 변환
                bs_embed, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                prompt_embeds_list.append(prompt_embeds)

        # 두 개의 text encoder 결과를 concat
        base_prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)  # [1, 77, 2048]
        base_pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)  # [1, 1280]

        # Negative Prompt에 반복
        negative_prompt_embeds_list = []
        for tokenizer, text_encoder in zip(base_tokenizers, base_text_encoders):
            with torch.no_grad():
                negative_text_inputs = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                negative_text_input_ids = negative_text_inputs.input_ids
                negative_prompt_embeds = text_encoder(negative_text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False)

                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds[-1][-2]
                
                bs_embed, seq_len, _ = negative_prompt_embeds.shape
                negative_prompt_embeds = negative_prompt_embeds.view(bs_embed, seq_len, -1)
                negative_prompt_embeds_list.append(negative_prompt_embeds)

            base_negative_prompt_embeds = torch.cat(negative_prompt_embeds_list, dim=-1)  # [1, 77, 2048]
            base_negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.view(bs_embed, -1)  # [1, 1280]

        # Concat
        base_prompt_embeds = torch.cat([base_negative_prompt_embeds, base_prompt_embeds])  # [2, 77, 2048]
        base_pooled_prompt_embeds = torch.cat([base_negative_pooled_prompt_embeds, base_pooled_prompt_embeds])  # [2, 1280]
        
        # For inpainting
        if self.use_inpaint:
            inpaint_text_encoders = [self.inpaint_text_encoder_1, self.inpaint_text_encoder_2]
            inpaint_tokenizers = [self.inpaint_tokenizer_1, self.inpaint_tokenizer_2]
            prompt_embeds_list = []

            for tokenizer, text_encoder in zip(inpaint_tokenizers, inpaint_text_encoders):
                with torch.no_grad():
                    text_inputs = tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    text_input_ids = text_inputs.input_ids
                    prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False)
                    
                    pooled_prompt_embeds = prompt_embeds[0]
                    prompt_embeds = prompt_embeds[-1][-2]
                    
                    bs_embed, seq_len, _ = prompt_embeds.shape
                    prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                    prompt_embeds_list.append(prompt_embeds)

            inpaint_prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)  # [1, 77, 2048]
            inpaint_pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)  # [1, 1280]

            # Negative Prompt
            negative_prompt_embeds_list = []
            for tokenizer, text_encoder in zip(inpaint_tokenizers, inpaint_text_encoders):
                with torch.no_grad():
                    negative_text_inputs = tokenizer(
                        negative_prompt,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    negative_text_input_ids = negative_text_inputs.input_ids
                    negative_prompt_embeds = text_encoder(negative_text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False)

                    negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                    negative_prompt_embeds = negative_prompt_embeds[-1][-2]
                    
                    bs_embed, seq_len, _ = negative_prompt_embeds.shape
                    negative_prompt_embeds = negative_prompt_embeds.view(bs_embed, seq_len, -1)
                    negative_prompt_embeds_list.append(negative_prompt_embeds)

            inpaint_negative_prompt_embeds = torch.cat(negative_prompt_embeds_list, dim=-1)  # [1, 77, 2048]
            inpaint_negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.view(bs_embed, -1)  # [1, 1280]

            inpaint_prompt_embeds = torch.cat([inpaint_negative_prompt_embeds, inpaint_prompt_embeds])  # [2, 77, 2048]
            inpaint_pooled_prompt_embeds = torch.cat([inpaint_negative_pooled_prompt_embeds, inpaint_pooled_prompt_embeds])  # [2, 1280]
            
        else:
            inpaint_prompt_embeds = None
            inpaint_pooled_prompt_embeds = None
        ########################################################
        # base_prompt_embeds : [2, 77, 2048]
        # base_pooled_prompt_embeds : [2, 1280]
        # inpaint_prompt_embeds : [2, 77, 2048]
        # inpaint_pooled_prompt_embeds : [2, 1280]
        ########################################################

        return (base_prompt_embeds, base_pooled_prompt_embeds), (inpaint_prompt_embeds, inpaint_pooled_prompt_embeds)
    
    def img2img_step(self, text_embeddings, inputs, depth_mask, guidance_scale=100, strength=1.0,
                     num_inference_steps=50, update_mask=None, latent_mode=False, check_mask=None,
                     fixed_seed=None, check_mask_iters=0.5, intermediate_vis=False):
        intermediate_results = []

        # depth_mask.shape : [1, 1, H, W]
        # inputs.shape : [1, 3, H, W]

        # Option 1 : Depth Estimator with MiDAS
        self.image = inputs
        def get_depth_map(image):
            image = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
            with torch.no_grad(), torch.autocast("cuda"):
                depth_map = self.depth_estimator(image).predicted_depth

            depth_map = torch.nn.functional.interpolate(
                depth_map.unsqueeze(1),
                size=(1024, 1024),
                mode="bicubic",
                align_corners=False,
            )
            depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
            depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
            image = torch.cat([depth_map] * 3, dim=1)

            # image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
            # image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
            controlnet_depth_mask = image
            return controlnet_depth_mask
        
        def sample(latents, depth_mask, cond_depth_mask, strength, num_inference_steps, update_mask=None, check_mask=None,
                   masked_latents=None):
            
            (base_prompt_embeds, base_pooled_prompt_embeds), (inpaint_prompt_embeds, inpaint_pooled_prompt_embeds) = text_embeddings
            self.scheduler.set_timesteps(num_inference_steps)
            noise = None
            
            if latents is None:
                # Last chanel is reserved for depth
                # self.base_unet.in_channels = 4 : -1 필요없음
                latents = torch.randn(
                    (
                        base_pooled_prompt_embeds.shape[0] // 2, self.base_unet.config.in_channels, depth_mask.shape[2],
                        depth_mask.shape[3]),
                    device=self.device)
                # print("latents", latents.shape) # [1, 4, 128, 128]
                timesteps = self.scheduler.timesteps
            else:
                # Strength has meaning only when latents are given
                timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
                latent_timestep = timesteps[:1]
                if fixed_seed is not None:
                    seed_everything(fixed_seed)
                noise = torch.randn_like(latents)
                # print("noise", noise.shape) # [1, 4, 128, 128]
                if update_mask is not None:
                    # NOTE: I think we might want to use same noise?
                    gt_latents = latents
                    latents = torch.randn(
                        (base_pooled_prompt_embeds.shape[0] // 2, self.base_unet.config.in_channels, depth_mask.shape[2],
                         depth_mask.shape[3]),
                        device=self.device)
                    # print("gt latents", gt_latents.shape) # [1, 4, 128, 128]
                    # print("latents", latents.shape) # [1, 4, 128, 128]
                else:
                    latents = self.scheduler.add_noise(latents, noise, latent_timestep)

            depth_mask = torch.cat([depth_mask] * 2) # [2 1 128 128]
            # print("depth_mask", depth_mask.shape)
            cond_depth_mask = torch.cat([cond_depth_mask] * 3, dim=1) # [1,3, 1024, 1024]

            with torch.autocast('cuda'):
                for i, t in tqdm(enumerate(timesteps)):
                    if self.use_inpaint:
                        is_inpaint_range = self.use_inpaint and (10 < i < 20)
                        mask_constraints_iters = True  # i < 20
                        is_inpaint_iter = is_inpaint_range  # and i %2 == 1
                        # print(mask_constraints_iters)
                    else :
                        is_inpaint_range = False
                        is_inpaint_iter = False
                        mask_constraints_iters = True
                        # print(mask_constraints_iters)

                    if not is_inpaint_range and mask_constraints_iters:
                        if update_mask is not None:
                            noised_truth = self.scheduler.add_noise(gt_latents, noise, t.unsqueeze(0))
                            if check_mask is not None and i < int(len(timesteps) * check_mask_iters):
                                curr_mask = check_mask
                            else:
                                curr_mask = update_mask
                            latents = latents * curr_mask + noised_truth * (1 - curr_mask)

                    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                    latent_model_input = torch.cat([latents] * 2)
                    # print("latent_model_input", latent_model_input.shape) # [2, 4, 128, 128]
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input,
                                                                          t)  # NOTE: This does nothing

                    # SDXL inpaint
                    if is_inpaint_iter:
                        latent_mask = torch.cat([update_mask] * 2) # [2, 1, 128, 128]
                        latent_image = torch.cat([masked_latents] * 2) # [2, 4, 128, 128] 
                        latent_model_input_inpaint = torch.cat([latent_model_input, latent_mask, latent_image], dim=1)
                        # [2, 9, 128, 128]
                        
                        inpaint_time_ids = self.get_add_time_ids_inpaint(
                            original_size=[[1024, 1024]], crops_coords_top_left=[[0, 0]], target_size=[1024, 1024],
                            aesthetic_score=0, negative_aesthetic_score=0
                        )[0]
                        inpaint_time_ids = torch.cat([inpaint_time_ids] * 2).to(self.device)

                        inpaint_added_cond_kwargs = {
                            "text_embeds": inpaint_pooled_prompt_embeds, # [2, 1280]
                            "time_ids": inpaint_time_ids  # [2,6]
                        }
                        
                        with torch.no_grad():
                            noise_pred_inpaint = \
                                self.inpaint_unet(latent_model_input_inpaint, t, 
                                                  encoder_hidden_states=inpaint_prompt_embeds, added_cond_kwargs=inpaint_added_cond_kwargs)[
                                    'sample']
                            noise_pred = noise_pred_inpaint
                    
                    # ControlNet Depth + SDXL
                    else:
                        latent_model_input = latent_model_input # [2, 4, 128, 128]
                        base_time_ids = self.get_add_time_ids_base(
                            original_size=[[1024, 1024]], crops_coords_top_left=[[0, 0]], target_size=[1024, 1024],
                            aesthetic_score=0, negative_aesthetic_score=0
                        )[0]
                        base_time_ids = torch.cat([base_time_ids] * 2).to(self.device)
                        base_added_cond_kwargs = {
                            "text_embeds": base_pooled_prompt_embeds,  # [2, 1280]
                            "time_ids": base_time_ids  # [2,6]
                        }
                        with torch.no_grad():
                            # ControlNet 적용 - Depth 정보를 추가하여 Latent를 변형
                            #=======================================================#
                            # NOTE : ❗Todo❗ Depth 정보를 어떻게 넣을 것인가?
                            # NOTE : Method 1 : ControlNet Depth Pseudo code와 마찬가지로 depth MiDAS로 연산해서 넣기
                            # cond_depth_mask = get_depth_map(inputs) # [1, 3, 1024, 1024]
                            # cond_depth_mask = torch.cat([cond_depth_mask] * 2) # [2 3 1024 1024]

                            # NOTE : Method 2 : render에서 연산한 depth mask 받기 : Channel 맞춰야함
                            # NOTE : Method 3 : Sample에 cond_depth_mask를 따로 추가
                            # [1, 1, H, W] -> [1, 1, 128, 128] -> [1, 1, 1024, 1024] ❌
                            # [1, 1, H, W] -> [1, 1, 1024, 1024] ⭕
                            # cond_depth_mask = torch.cat([cond_depth_mask] * 3, dim=1)  # (B, 1, H, W) → (B, 3, H, W)
                            # cond_depth_mask = F.interpolate(cond_depth_mask, size=(1024, 1024), mode='bicubic',
                            #        align_corners=False) # [2, 3, 1024, 1024]

                            # NOTE : Method 3 사용 (250304)
                            #=======================================================#
                            controlnet_latent = self.controlnet(
                                latent_model_input,  # 현재 Latent 입력
                                t,  # 현재 timestep
                                encoder_hidden_states=base_prompt_embeds,  # Text Condition
                                controlnet_cond=cond_depth_mask,  # ControlNet에 Depth 정보를 입력
                                added_cond_kwargs=base_added_cond_kwargs,
                                return_dict=True  # NOTE : v2(250303) Dict 로 반환해서 down block, mid block 반환
                            )  # ControlNet이 변형 한 latent 출력

                            #=======================================================#
                            # NOTE : ❗Todo❗ ISSUE 1 . Blurry Output
                            # Blurry output의 원인으로는 ControlNet의 Feature mismatch


                            # controlnet_output = controlnet_latent['sample'] # [2, 320, 128, 128]
                            # controlnet_down_features = controlnet_latent['down_block_res_samples']
                            # controlnet_mid_features = controlnet_latent['mid_block_res_sample'] #[2, 1280, 32, 32]

                            controlnet_scale = 0.5  
                            controlnet_down_features = [
                                feature * controlnet_scale for feature in controlnet_latent['down_block_res_samples']
                            ]
                            controlnet_mid_features = controlnet_latent['mid_block_res_sample'] * controlnet_scale
                            
                            # print controlnet_features shape for each step
                            # print('controlnet_features', controlnet_features[0].shape) # [2, 320, 128, 128]
                            # print('controlnet_features', controlnet_features[1].shape) # [2, 320, 128, 128]
                            # print('controlnet_features', controlnet_features[2].shape) # [2, 320, 64, 64]
                            # print('controlnet_features', controlnet_features[3].shape) # [2, 640, 64, 64]
                            # print('controlnet_features', controlnet_features[4].shape) # [2, 640, 32, 32]
                            # print('controlnet_features', controlnet_features[5].shape) # [2, 640, 32, 32]
                            # print('controlnet_features', controlnet_features[6].shape) # [2, 1280, 32, 32]
                            # print('controlnet_features', controlnet_features[7].shape) # [2, 1280, 32, 32]
                            
                            # mid_block_additional_residual 
                            # — (torch.Tensor, optional): A tensor that if specified is added to the residual of the middle unet block.
                            # NOTE : v2(250303) mid_block_additional_residuals 추가 완료

                            # cross_attention_kwargs (dict, optional) — 
                            # A kwargs dictionary that if specified is passed along to the AttentionProcessor as defined under self.processor in diffusers.models.attention_processor.
                            
                            noise_pred = self.base_unet(
                                latent_model_input,  # 초기 latent (ControlNet을 거치지 않은 원본 latent)
                                t,  
                                encoder_hidden_states=base_prompt_embeds,  # Text Condition
                                added_cond_kwargs=base_added_cond_kwargs,  # SDXL Time Embedding 추가
                                down_block_additional_residuals=controlnet_down_features,  # ControlNet Downblock feature
                                mid_block_additional_residual=controlnet_mid_features,  # ControlNet Midblock feature
                            )['sample']  # SDXL ControlNet Depth

                    # perform guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1

                    if intermediate_vis:
                        vis_alpha_t = torch.sqrt(self.scheduler.alphas_cumprod)
                        vis_sigma_t = torch.sqrt(1 - self.scheduler.alphas_cumprod)
                        a_t, s_t = vis_alpha_t[t], vis_sigma_t[t]
                        vis_latents = (latents - s_t * noise) / a_t
                        if is_inpaint_iter:
                            vis_latents = 1 / 0.13025 * vis_latents
                            image = self.inpaint_vae.decode(vis_latents).sample
                        else: 
                            vis_latents = 1 / 0.13025 * vis_latents
                            # vis_latents = 1 / 0.18215 * vis_latents
                            image = self.base_vae.decode(vis_latents).sample
                        image = (image / 2 + 0.5).clamp(0, 1)
                        image = image.cpu().permute(0, 2, 3, 1).numpy()
                        image = Image.fromarray((image[0] * 255).round().astype("uint8"))
                        intermediate_results.append(image)
                    latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

            return latents
        
        # depth_mask = F.interpolate(depth_mask, size=(64, 64), mode='bicubic',
        #                            align_corners=False)
        tmp_depth_mask = depth_mask #[1, 1, H, W]
        depth_mask = F.interpolate(depth_mask, size=(128, 128), mode='bicubic',
                                   align_corners=False)
        cond_depth_mask = F.interpolate(tmp_depth_mask, size=(1024, 1024), mode='bicubic',
                                        align_corners=False) # [1, 1, 1024, 1024]
        masked_latents = None
        if inputs is None:
            latents = None
        elif latent_mode:
            latents = inputs
        else:
            pred_rgb_1024 = F.interpolate(inputs, (1024, 1024), mode='bilinear',
                                         align_corners=False)
            latents = self.encode_imgs(pred_rgb_1024)
            # print('latents', latents.shape) [1, 4, 128, 128]
            if self.use_inpaint:
                update_mask_1024 = F.interpolate(update_mask, (1024, 1024))
                masked_inputs = pred_rgb_1024 * (update_mask_1024 < 0.5) + 0.5 * (update_mask_1024 >= 0.5)
                masked_latents = self.encode_imgs(masked_inputs)

        if update_mask is not None:
            update_mask = F.interpolate(update_mask, (128, 128), mode='nearest')
        if check_mask is not None:
            check_mask = F.interpolate(check_mask, (128, 128), mode='nearest')

        depth_mask = 2.0 * (depth_mask - depth_mask.min()) / (depth_mask.max() - depth_mask.min()) - 1.0
        cond_depth_mask = 2.0 * (cond_depth_mask - cond_depth_mask.min()) / (cond_depth_mask.max() - cond_depth_mask.min()) - 1.0
        # print('depth_mask', depth_mask.shape) [1, 1, 128, 128]

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        # t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        t = (self.min_step + self.max_step) // 2

        with torch.no_grad():
            target_latents = sample(latents, depth_mask, cond_depth_mask, strength=strength, num_inference_steps=num_inference_steps,
                                    update_mask=update_mask, check_mask=check_mask, masked_latents=masked_latents)
            target_rgb = self.decode_latents(target_latents)

        if latent_mode:
            return target_rgb, target_latents
        else:
            return target_rgb, intermediate_results
        
    def get_add_time_ids_base(
                            self, original_size, crops_coords_top_left, target_size, 
                            aesthetic_score, negative_aesthetic_score, dtype=torch.float16
                        ):
                            # if self.config.requires_aesthetics_score:
                            #     add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
                            #     add_neg_time_ids = list(original_size + crops_coords_top_left + (negative_aesthetic_score,))
                            # else:
                            #     add_time_ids = list(original_size + crops_coords_top_left + target_size)
                            #     add_neg_time_ids = list(original_size + crops_coords_top_left + target_size)

                            add_time_ids_list = []
                            add_neg_time_ids_list = []

                            for original_size, crop_top_left in zip(original_size, crops_coords_top_left):
                                add_time_ids = list(original_size + crop_top_left + target_size)
                                add_neg_time_ids = list(original_size + crop_top_left + target_size)

                                add_time_ids_list.append(torch.tensor(add_time_ids, dtype=dtype))
                                add_neg_time_ids_list.append(torch.tensor(add_neg_time_ids, dtype=dtype))

                            add_time_ids = torch.stack(add_time_ids_list, dim=0)  # Shape: (batch_size, num_features)
                            add_neg_time_ids = torch.stack(add_neg_time_ids_list, dim=0)

                            add_time_ids_base = add_time_ids
                            add_neg_time_ids_base = add_neg_time_ids

                            return add_time_ids_base, add_neg_time_ids_base
    
    def get_add_time_ids_inpaint(
                            self, original_size, crops_coords_top_left, target_size, 
                            aesthetic_score, negative_aesthetic_score, dtype=torch.float16
                        ):
                            # if self.config.requires_aesthetics_score:
                            #     add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
                            #     add_neg_time_ids = list(original_size + crops_coords_top_left + (negative_aesthetic_score,))
                            # else:
                            #     add_time_ids = list(original_size + crops_coords_top_left + target_size)
                            #     add_neg_time_ids = list(original_size + crops_coords_top_left + target_size)

                            add_time_ids_list = []
                            add_neg_time_ids_list = []

                            for original_size, crop_top_left in zip(original_size, crops_coords_top_left):
                                add_time_ids = list(original_size + crop_top_left + target_size)
                                add_neg_time_ids = list(original_size + crop_top_left + target_size)

                                add_time_ids_list.append(torch.tensor(add_time_ids, dtype=dtype))
                                add_neg_time_ids_list.append(torch.tensor(add_neg_time_ids, dtype=dtype))

                            add_time_ids = torch.stack(add_time_ids_list, dim=0)  # Shape: (batch_size, num_features)
                            add_neg_time_ids = torch.stack(add_neg_time_ids_list, dim=0)

                            add_time_ids_inpaint = add_time_ids
                            add_neg_time_ids_inpaint = add_neg_time_ids

                            return add_time_ids_inpaint, add_neg_time_ids_inpaint
    
    #====================================================================#
    # NOTE : No use in trainer.py
    #====================================================================#
    
    def train_step(self, text_embeddings, inputs, depth_mask, guidance_scale=100):

        # interp to 512x512 to be fed into vae.

        # _t = time.time()
        if not self.latent_mode:
            # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
            pred_rgb_1024 = F.interpolate(inputs, (1024, 1024), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_1024)
            depth_mask = F.interpolate(depth_mask, size=(128, 128), mode='bicubic',
                                       align_corners=False)

        else:
            latents = inputs

        depth_mask = 2.0 * (depth_mask - depth_mask.min()) / (depth_mask.max() - depth_mask.min()) - 1.0
        # depth_mask = F.interpolate(depth_mask, size=(64,64), mode='bicubic',
        #                            align_corners=False)
        depth_mask = torch.cat([depth_mask] * 2)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # encode image into latents with vae, requires grad!
        # _t = time.time()

        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            if self.no_noise:
                noise = torch.zeros_like(latents)
                latents_noisy = latents
            else:
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            # add depth
            latent_model_input = torch.cat([latent_model_input, depth_mask], dim=1)
            noise_pred = self.base_unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), alpha_t * sigma_t^2
        w = (1 - self.alphas[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        # clip grad for stable training?
        # grad = grad.clamp(-1, 1)
        grad = torch.nan_to_num(grad)

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        # _t = time.time()
        latents.backward(gradient=grad, retain_graph=True)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: backward {time.time() - _t:.4f}s')

        return 0  # dummy loss value

    def produce_latents(self, text_embeddings_tuple, depth_mask, height=1024, width=1024, num_inference_steps=50,
                    guidance_scale=7.5, latents=None, strength=0.5):

        text_embeds, inpaint_text_embeds = text_embeddings_tuple

        self.scheduler.set_timesteps(num_inference_steps)

        if latents is None:
            # Last channel is reserved for depth
            latents = torch.randn((text_embeds.shape[0] // 2, self.base_unet.config.in_channels, height // 8, width // 8),
                                device=self.device)
            timesteps = self.scheduler.timesteps
        else:
            # Strength has meaning only when latents are given
            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
            latent_timestep = timesteps[:1]
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(latents, noise, latent_timestep)

        depth_mask = torch.cat([depth_mask] * 2)
        with torch.autocast('cuda'):
            for i, t in tqdm(enumerate(timesteps)):
                is_inpaint_iter = (10 < i < 20)  # inpaint 구간 조건

                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat([latent_model_input, depth_mask], dim=1)

                # 선택된 text_embeddings에 따라 unet 예측 수행
                chosen_text_embeddings = inpaint_text_embeds if is_inpaint_iter else text_embeds

                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=chosen_text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents


    def decode_latents(self, latents):
        # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
        # latents = 1 / 0.18215 * latents # SDv2
        # NOTE : Scaling Factor changed
        # SOURCE : https://github.com/huggingface/diffusers/issues/6923
        latents = 1 / 0.13025 * latents
        vae = self.base_vae
        if self.use_inpaint:
            vae = self.inpaint_vae

        with torch.no_grad():
            imgs = vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.base_vae.encode(imgs).latent_dist
        # latents = posterior.sample() * 0.18215
        latents = posterior.sample() * 0.13025

        return latents

    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

    def prompt_to_img(self, prompts, depth_mask, height=1024, width=1024, num_inference_steps=50, guidance_scale=7.5,
                      latents=None, strength=0.5):

        if isinstance(prompts, str):
            prompts = [prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts)  # [2, 77, 768]
        # new should be torch.Size([2, 77, 1024])

        # depth is in range of 20-1500 of size 1x384x384, normalized to -1 to 1, mean was -0.6
        # Resized to 64x64 # TODO: Understand range here
        depth_mask = 2.0 * (depth_mask - depth_mask.min()) / (depth_mask.max() - depth_mask.min()) - 1.0
        depth_mask = F.interpolate(depth_mask.unsqueeze(1), size=(height // 8, width // 8), mode='bicubic',
                                   align_corners=False)

        # Added as an extra channel to the latents

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, depth_mask=depth_mask, height=height, width=width, latents=latents,
                                       num_inference_steps=num_inference_steps,
                                       guidance_scale=guidance_scale, strength=strength)  # [1, 4, 128, 128]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs