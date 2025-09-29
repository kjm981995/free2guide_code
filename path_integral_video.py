############################################################
# 1. Define Reward Function
############################################################

############################################################
# 1.1. Imagereward model
############################################################

import os
import ImageReward as imagereward
import functools
from PIL import Image
import torch

weight_dtype = torch.float16
image_reward = imagereward.load("ImageReward-v1.0")
image_reward.half()
image_reward.requires_grad_(False)

def image_reward_get_reward(
    model, pil_image, prompt, weight_dtype
):
  """Gets rewards using ImageReward model."""
  image = (
      model.preprocess(pil_image).unsqueeze(0).to(weight_dtype).to(model.device)
  )
  image_embeds = model.blip.visual_encoder(image)
  image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
      model.device
  )

  text_input = model.blip.tokenizer(
      prompt,
      padding="max_length",
      truncation=True,
      max_length=35,
      return_tensors="pt",
  ).to(model.device)
  text_output = model.blip.text_encoder(
      text_input.input_ids,
      attention_mask=text_input.attention_mask,
      encoder_hidden_states=image_embeds,
      encoder_attention_mask=image_atts,
      return_dict=True,
  )
  txt_features = text_output.last_hidden_state[:, 0, :]
  rewards = model.mlp(txt_features)
  rewards = (rewards - model.mean) / model.std
  return rewards, txt_features


def _calculate_reward_ir(
    weight_dtype,
    image_reward,
    imgs,
    prompts,
    reward_filter=0,
):
  """Computes reward using ImageReward model."""

  blip_reward, _ = image_reward_get_reward(
      image_reward, imgs, prompts, weight_dtype
  )
  if reward_filter == 1:
    blip_reward = torch.clamp(blip_reward, min=0)
  return blip_reward.cpu().squeeze(0).squeeze(0)


calculate_reward = functools.partial(
        _calculate_reward_ir,
        weight_dtype,
        image_reward,
    )

def get_reward(image, prompt):
    with torch.no_grad():
        reward = calculate_reward(image, prompt).sum()

    return reward


############################################################
# 1.2. CLIP Model
############################################################

from transformers import CLIPProcessor, CLIPModel
import torch
import os 
from PIL import Image

clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")    
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def get_reward(image, prompt):
    with torch.no_grad():

        inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True)
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        reward = clip(**inputs).logits_per_image.sum()

    return reward


############################################################
# 1.3. LVLM
############################################################

import asyncio
import nest_asyncio
from openai import AsyncOpenAI 
import base64
import io
from PIL import Image

nest_asyncio.apply()

def encode_image(image):
    buffered = io.BytesIO()
    image = image.resize((512, 512))
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def calculate_prob(response):
    if response.choices[0].logprobs.content[-1].token in [".", "!", "?"]:
        top_logs = response.choices[0].logprobs.content[-2].top_logprobs
    else :
        top_logs = response.choices[0].logprobs.content[-1].top_logprobs

    try:
        positive = float(top_logs[0].token)
    except:
        positive = 0.0

    return positive

async def send_request(image, question, verbose = False):
    image_encoded = encode_image(image)
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", 
            "content": "You are a useful helper that responds to video quality assessments." \
            "The given image is a grid of four key frames of a video: the top left is the first frame, the top right is the second frame, the bottom left is the third frame, and finally the bottom right is the fourth frame." \
            " Answer the reason first and the final answer later. Start the reason first with 'Reasoning: ' in front of the reason part and review your reasoning logically." \
            " After reviewing your reasoning, give the final answer with 'Answer: '." \
            " You should check all frame and comparing them, and ensure your reasoning leads to a sound final answer." \
            " Your final 'answer' should one score only and the score must be from 1 to 9 without decimals." \
            " Let's think step by step."
            },
            {"role": "user", "content": [
                {
                "type" : "text",
                "text" : question
                },
                {
                "type" : "image_url",
                "image_url" : {
                    "url" : f"data:image/png;base64,{image_encoded}",
                    "detail" : "low"
                }
                }
            ]
            }
        ],
        temperature=0.3,
        logprobs=True,
        top_logprobs=1,
        seed=42
    )

    if verbose:
        print(response.choices[0].message.content)

    return float(calculate_prob(response))

async def get_reward(image_list, prompt, verbose = False):

    question = "For a given image as keyframes of video, Rate the following questions:" \
        " Considering all four images, does the prompt,"+ prompt + ", describe the video well enough?" \
        " Review your reasoning thoroughly and then respond with your final decision prefixed by 'Answer: '."
    
    tasks = [send_request(image, question, verbose) for image in image_list]
    reward_list = await asyncio.gather(*tasks)

    return reward_list

MODEL="gpt-4o-2024-08-06"
client = AsyncOpenAI(api_key="YourAPI")


############################################################
# 2. Generates Multiple Denoised Video Samples 
############################################################

import numpy as np
from tqdm import tqdm
import torch
from lvdm.models.utils_diffusion import make_ddim_sampling_parameters, make_ddim_timesteps
from PIL import Image

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.counter = 0

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))
        self.use_scale = self.model.use_scale
        print('DDIM scale', self.use_scale)

        if self.use_scale:
            self.register_buffer('scale_arr', to_torch(self.model.scale_arr))
            ddim_scale_arr = self.scale_arr.cpu()[self.ddim_timesteps]
            self.register_buffer('ddim_scale_arr', ddim_scale_arr)
            ddim_scale_arr = np.asarray([self.scale_arr.cpu()[0]] + self.scale_arr.cpu()[self.ddim_timesteps[:-1]].tolist())
            self.register_buffer('ddim_scale_arr_prev', ddim_scale_arr)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)


    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, repeat = 1,
                      uc_type=None, conditional_guidance_scale_temporal=None, **kwargs):
        b, *_, device = *x.shape, x.device
        if x.dim() == 5:
            is_video = True
        else:
            is_video = False
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c, **kwargs) # unet denoiser
        else:
            # with unconditional condition
            if isinstance(c, torch.Tensor):
                e_t = self.model.apply_model(x, t, c, **kwargs)
                e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, **kwargs)
            elif isinstance(c, dict):
                e_t = self.model.apply_model(x, t, c, **kwargs)
                e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, **kwargs)
            else:
                raise NotImplementedError
            # text cfg
            if uc_type is None:
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            else:
                if uc_type == 'cfg_original':
                    e_t = e_t + unconditional_guidance_scale * (e_t - e_t_uncond)
                elif uc_type == 'cfg_ours':
                    e_t = e_t + unconditional_guidance_scale * (e_t_uncond - e_t)
                else:
                    raise NotImplementedError
            # temporal guidance
            if conditional_guidance_scale_temporal is not None:
                e_t_temporal = self.model.apply_model(x, t, c, **kwargs)
                e_t_image = self.model.apply_model(x, t, c, no_temporal_attn=True, **kwargs)
                e_t = e_t + conditional_guidance_scale_temporal * (e_t_temporal - e_t_image)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        
        if is_video:
            size = (b, 1, 1, 1, 1)
        else:
            size = (b, 1, 1, 1)
        a_t = torch.full(size, alphas[index], device=device)
        a_prev = torch.full(size, alphas_prev[index], device=device)
        sigma_t = torch.full(size, sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(size, sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

        x_prevs = []
        for i in range(repeat):

            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            
            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            if self.use_scale:
                scale_arr = self.model.scale_arr if use_original_steps else self.ddim_scale_arr
                scale_t = torch.full(size, scale_arr[index], device=device)
                scale_arr_prev = self.model.scale_arr_prev if use_original_steps else self.ddim_scale_arr_prev
                scale_t_prev = torch.full(size, scale_arr_prev[index], device=device)
                if i == 0:
                    pred_x0 /= scale_t 
                x_prev = a_prev.sqrt() * scale_t_prev * pred_x0 + dir_xt + noise
            else:
                x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

            x_prevs.append(x_prev)

        if repeat == 1:
            x_prevs = x_prevs[0]

        torch.cuda.empty_cache()   
        return x_prevs, pred_x0

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               schedule_verbose=False,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               prompt="",
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        
        # check condition bs
        if conditioning is not None:
            if isinstance(conditioning, dict):
                try:
                    cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                except:
                    cbs = conditioning[list(conditioning.keys())[0]][0].shape[0]

                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=schedule_verbose)
        
        # make shape
        if len(shape) == 3:
            C, H, W = shape
            size = (batch_size, C, H, W)
        elif len(shape) == 4:
            C, T, H, W = shape
            size = (batch_size, C, T, H, W)
        # print(f'Data shape for DDIM sampling is {size}, eta {eta}')
        
        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    verbose=verbose,
                                                    prompt=prompt,
                                                    **kwargs)
        torch.cuda.empty_cache()
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, verbose=True,
                      cond_tau=1., target_size=None, start_timesteps=None, prompt="",
                      **kwargs):
        device = self.model.betas.device        
        print('ddim device', device)
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
        
        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]
            
        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        if verbose:
            iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        else:
            iterator = time_range

        init_x0 = False
        clean_cond = kwargs.pop("clean_cond", False)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            if start_timesteps is not None:
                assert x0 is not None
                if step > start_timesteps*time_range[0]:
                    continue
                elif not init_x0:
                    img = self.model.q_sample(x0, ts) 
                    init_x0 = True

            ############################################################
            # 3. Free2Guide
            ############################################################

            repeat = 10

            if i >=0 and i <= 5 and repeat > 1:
                reward_list = []

                if i == 0 : 
                    x_prevs = []
                    for j in range(repeat):
                        x_prev, pred_x0 = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                            quantize_denoised=quantize_denoised, temperature=temperature,
                            noise_dropout=noise_dropout, score_corrector=score_corrector,
                            corrector_kwargs=corrector_kwargs,
                            unconditional_guidance_scale=unconditional_guidance_scale,
                            unconditional_conditioning=unconditional_conditioning,
                            x0=x0, repeat = 1,
                            **kwargs)
                        
                        x_prevs.append(x_prev)

                        k = self.model.decode_first_stage_2DAE(pred_x0)
                        k = torch.clamp(k, -1., 1.)
                        k = (k+1)/2.

                        key_frames = [0, 5, 10, 15]

                        reward = 0
                        for idx, frame in enumerate(key_frames):
                            image = Image.fromarray((k[0, :, frame, :, :].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
                            reward += get_reward(image, prompt)

                        reward_list.append(reward.detach().cpu().numpy())
                        img = torch.randn(shape, device=device)
                    
                else:
                    x_prevs, _  = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                            quantize_denoised=quantize_denoised, temperature=temperature,
                            noise_dropout=noise_dropout, score_corrector=score_corrector,
                            corrector_kwargs=corrector_kwargs,
                            unconditional_guidance_scale=unconditional_guidance_scale,
                            unconditional_conditioning=unconditional_conditioning,
                            x0=x0, repeat = repeat,
                            **kwargs)
                    

                    tsm1 = ts - 20
                    indexm1 = index - 1

                    for j in range(repeat):

                        _, pred_x0 = self.p_sample_ddim(x_prevs[j], cond, tsm1, index=indexm1, use_original_steps=ddim_use_original_steps,
                            quantize_denoised=quantize_denoised, temperature=temperature,
                            noise_dropout=noise_dropout, score_corrector=score_corrector,
                            corrector_kwargs=corrector_kwargs,
                            unconditional_guidance_scale=unconditional_guidance_scale,
                            unconditional_conditioning=unconditional_conditioning,
                            x0=x0, repeat = 1,
                            **kwargs)

                        k = self.model.decode_first_stage_2DAE(pred_x0)
                        k = torch.clamp(k, -1., 1.)
                        k = (k+1)/2.
                        
                        key_frames = [0, 5, 10, 15]
                        reward = 0
                        for idx, frame in enumerate(key_frames):
                            image = Image.fromarray((k[0, :, frame, :, :].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
                            reward += get_reward(image, prompt)

                        reward_list.append(reward.detach().cpu().numpy())
                

                img = x_prevs[np.argmax(reward_list)]
                
            else:
                outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                        quantize_denoised=quantize_denoised, temperature=temperature,
                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                        corrector_kwargs=corrector_kwargs,
                        unconditional_guidance_scale=unconditional_guidance_scale,
                        unconditional_conditioning=unconditional_conditioning,
                        x0=x0,
                        **kwargs)

                img, pred_x0 = outs 

            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        torch.cuda.empty_cache()   

        return img, intermediates


    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)

        def extract_into_tensor(a, t, x_shape):
            b, *_ = t.shape
            out = a.gather(-1, t)
            return out.reshape(b, *((1,) * (len(x_shape) - 1)))

        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec

############################################################
# 4. Generate Reward-aligned Video
############################################################

import os, sys, glob
import numpy as np
from collections import OrderedDict
from decord import VideoReader, cpu
import cv2

import torch
import torchvision
sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))

def batch_ddim_sampling(model, cond, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1.0,\
                        cfg_scale=1.0, temporal_cfg_scale=None, prompt="", **kwargs):
    ddim_sampler = DDIMSampler(model)
    uncond_type = model.uncond_type
    batch_size = noise_shape[0]

    ## construct unconditional guidance
    if cfg_scale != 1.0:
        if uncond_type == "empty_seq":
            prompts = batch_size * [""]
            #prompts = N * T * [""]  ## if is_imgbatch=True
            uc_emb = model.get_learned_conditioning(prompts)
        elif uncond_type == "zero_embed":
            c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
            uc_emb = torch.zeros_like(c_emb)
                
        ## process image embedding token
        if hasattr(model, 'embedder'):
            uc_img = torch.zeros(noise_shape[0],3,224,224).to(model.device)
            ## img: b c h w >> b l c
            uc_img = model.get_image_embeds(uc_img)
            uc_emb = torch.cat([uc_emb, uc_img], dim=1)
        
        if isinstance(cond, dict):
            uc = {key:cond[key] for key in cond.keys()}
            uc.update({'c_crossattn': [uc_emb]})
        else:
            uc = uc_emb
    else:
        uc = None
    
    x_T = None
    batch_variants = []
    #batch_variants1, batch_variants2 = [], []
    for _ in range(n_samples):
        if ddim_sampler is not None:
            kwargs.update({"clean_cond": True})
            samples, _ = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=noise_shape[0],
                                            shape=noise_shape[1:],
                                            verbose=True,
                                            unconditional_guidance_scale=cfg_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            temporal_length=noise_shape[2],
                                            conditional_guidance_scale_temporal=temporal_cfg_scale,
                                            x_T=x_T,
                                            prompt=prompt,
                                            **kwargs
                                            )
        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage_2DAE(samples)
        batch_variants.append(batch_images)
    ## batch, <samples>, c, t, h, w
    batch_variants = torch.stack(batch_variants, dim=1)

    torch.cuda.empty_cache()
    return batch_variants


def get_filelist(data_dir, ext='*'):
    file_list = glob.glob(os.path.join(data_dir, '*.%s'%ext))
    file_list.sort()
    return file_list

def get_dirlist(path):
    list = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file in files:
            m = os.path.join(path,file)
            if (os.path.isdir(m)):
                list.append(m)
    list.sort()
    return list


def load_model_checkpoint(model, ckpt):
    def load_checkpoint(model, ckpt, full_strict):
        state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
        try:
            ## deepspeed
            new_pl_sd = OrderedDict()
            for key in state_dict['module'].keys():
                new_pl_sd[key[16:]]=state_dict['module'][key]
            model.load_state_dict(new_pl_sd, strict=full_strict)
        except:
            if "state_dict" in list(state_dict.keys()):
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=full_strict)
        return model
    load_checkpoint(model, ckpt, full_strict=True)
    print('>>> model checkpoint loaded.')
    return model


def load_prompts(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list


def save_videos(batch_tensors, savedir, filenames, fps=10):
    # b,samples,c,t,h,w
    n_samples = batch_tensors.shape[1]
    for idx, vid_tensor in enumerate(batch_tensors):
        video = vid_tensor.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n_samples)) for framesheet in video] #[3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        savepath = os.path.join(savedir, f"{filenames[idx]}.mp4")
        torchvision.io.write_video(savepath, grid, fps=fps, video_codec='h264', options={'crf': '10'})

############################################################
# 5. Argument Setting
############################################################

from ast import arg
import argparse, os, sys, glob, yaml, math, random
import datetime, time
import numpy as np
from omegaconf import OmegaConf
from collections import OrderedDict
from tqdm import trange, tqdm

from einops import rearrange, repeat
from functools import partial
import torch
from pytorch_lightning import seed_everything

from utils.utils import instantiate_from_config

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123, help="seed for seed_everything")
    parser.add_argument("--mode", default="base", type=str, help="which kind of inference mode: {'base', 'i2v'}")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/base_512_v2/model.ckpt", help="checkpoint path")
    parser.add_argument("--config", type=str, default = "configs/inference_t2v_512_v2.0.yaml", help="config (yaml) path")
    parser.add_argument("--prompt_file", type=str, default="prompts/spatial_relationship.txt", help="a text file containing many prompts")
    parser.add_argument("--savedir", type=str, default="results/clip1-5/spatial_relationship", help="results saving path")
    parser.add_argument("--savefps", type=str, default=10, help="video fps to generate")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference")
    parser.add_argument("--height", type=int, default=320, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frames", type=int, default=-1, help="frames num to inference")
    parser.add_argument("--fps", type=int, default=28)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=12.0, help="prompt classifier-free guidance")
    parser.add_argument("--unconditional_guidance_scale_temporal", type=float, default=None, help="temporal consistency guidance")
    ## for conditional i2v only
    parser.add_argument("--cond_input", type=str, default=None, help="data dir of conditional input")
    return parser


def run_inference(args, gpu_num, gpu_no, **kwargs):
    ## step 1: model config
    ## -----------------------------------------------------------------
    config = OmegaConf.load(args.config)
    #data_config = config.pop("data", OmegaConf.create())
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)
    assert os.path.exists(args.ckpt_path), f"Error: checkpoint [{args.ckpt_path}] Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()
    
    ## sample shape
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    frames = model.temporal_length if args.frames < 0 else args.frames
    channels = model.channels
    
    ## saving folders
    os.makedirs(args.savedir, exist_ok=True)

    ## step 2: load data
    ## -----------------------------------------------------------------
    assert os.path.exists(args.prompt_file), "Error: prompt file NOT Found!"
    prompt_list = load_prompts(args.prompt_file)
    num_samples = len(prompt_list)
    # filename_list = [f"{id+1:04d}" for id in range(num_samples)]
    filename_list = prompt_list

    samples_split = num_samples // gpu_num
    residual_tail = num_samples % gpu_num
    print(f'[rank:{gpu_no}] {samples_split}/{num_samples} samples loaded.')
    indices = list(range(samples_split*gpu_no, samples_split*(gpu_no+1)))
    if gpu_no == 0 and residual_tail != 0:
        indices = indices + list(range(num_samples-residual_tail, num_samples))
    prompt_list_rank = [prompt_list[i] for i in indices]
    filename_list_rank = [filename_list[i] for i in indices]

    ## step 3: run over samples
    ## -----------------------------------------------------------------
    start = time.time()
    n_rounds = len(prompt_list_rank) // args.bs
    n_rounds = n_rounds+1 if len(prompt_list_rank) % args.bs != 0 else n_rounds
    for idx in range(0, n_rounds):
        print(f'[rank:{gpu_no}] batch-{idx+1} ({args.bs})x{args.n_samples} ...')
        idx_s = idx*args.bs
        idx_e = min(idx_s+args.bs, len(prompt_list_rank))
        batch_size = idx_e - idx_s
        filenames = filename_list_rank[idx_s:idx_e]
        noise_shape = [batch_size, channels, frames, h, w]
        fps = torch.tensor([args.fps]*batch_size).to(model.device).long()

        prompts = prompt_list_rank[idx_s:idx_e]
        print(prompts)
        if isinstance(prompts, str):
            prompts = [prompts]
        #prompts = batch_size * [""]
        text_emb = model.get_learned_conditioning(prompts)

        cond = {"c_crossattn": [text_emb], "fps": fps}

        ## inference
        batch_samples = batch_ddim_sampling(model, cond, noise_shape, args.n_samples, \
                                                args.ddim_steps, args.ddim_eta, args.unconditional_guidance_scale, prompt = prompts[0], **kwargs)
        torch.cuda.empty_cache()
        ## b,samples,c,t,h,w
        save_videos(batch_samples, args.savedir, filenames, fps=args.savefps)

    print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")
    torch.cuda.empty_cache()


if __name__ == '__main__':
    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@CoLVDM Inference: %s"%now)
    parser = get_parser()
    args = parser.parse_args(args=[])
    seed_everything(args.seed)
    rank, gpu_num = 0, 1
    run_inference(args, gpu_num, rank)