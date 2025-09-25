import os
import sys
import yaml
import copy
from pathlib import Path
from PIL import Image
from einops import rearrange, repeat
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import StableDiffusionPipeline, DDIMScheduler

from anytoedit.anytoedit_utils import prepare_input, save_results, MultiModalInput
from third_party.tokenflow.run_tokenflow_pnp import TokenFlow
from third_party.tokenflow.preprocess import Preprocess
from third_party.tokenflow.util import save_video

class AnytoEdit(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = self.config.device
        self.dtype = torch.float16 if self.config.precision==16 else torch.float32

        self.vision_decoder, self.vision_scheduler = self.init_sd()
        self.codi = self.init_codi_encoder()
        # self.codi.inverse_codi_to_clip = self.init_codi_to_clip().to(self.device)

    def edit(self):
        # Load tokenflow
        editor = TokenFlow(self.vision_decoder, self.vision_scheduler, self.config)

        # Read data
        inputs = prepare_input(self.config.data.input, self.config) 
        anchor = prepare_input(self.config.data.anchor, self.config) 
        negative = prepare_input(self.config.data.negative, self.config) 
        condition = [prepare_input(d, self.config) for d in self.config.data.condition]
        save_results(
            MultiModalInput(data=inputs.data, data_type=inputs.data_type), 
            self.args.outputs, 'original'
        )

        # Encode data
        inputs_latent = editor.encode_imgs(rearrange(inputs.data, 'c h w f -> f c h w'))

        # Encode conditions
        uncond_embeds = self.encode_condition(anchor)
        negative_embeds = self.encode_condition(negative)
        condition_embeds =  [self.encode_condition(c, uncond_embeds=uncond_embeds) for c in condition]
        
        # Downcast datatype
        uncond_embeds.data = uncond_embeds.data.to(device=self.device, dtype=self.dtype)
        negative_embeds.data = negative_embeds.data.to(device=self.device, dtype=self.dtype)
        for c in condition_embeds:
            c.data = c.data.to(device=self.device, dtype=self.dtype)

        # DDIM inversion
        latent_path = '_'.join(['sd', str(self.config.model.sd_version),
                    Path(inputs.path).stem,
                    'steps', str(self.config.inversion.n_timesteps),
                    'nframes', str(inputs.n_frames)]) 
        latent_path_tensor = os.path.join(self.args.latents, latent_path, 'latents')
        Path(latent_path_tensor).mkdir(parents=True, exist_ok=True)
        
        if (len(os.listdir(latent_path_tensor)) - self.config.n_timesteps) < 0:
            print("Inverted latents not found. Generating it...")
            with torch.autocast(device_type=self.device, dtype=self.dtype):
                prep = Preprocess(
                    pipeline = self.vision_decoder, 
                    scheduler = copy.deepcopy(self.vision_scheduler),
                    config = self.config
                )
                
                prep.extract_latents(
                    inputs_latent=inputs_latent, 
                    cond_embeds=uncond_embeds, 
                    save_path=os.path.join(self.args.latents, latent_path),
                    data_type=inputs.data_type
                )
        
        # Edit!
        print(f'Editing {inputs.data_type}..')
        results = editor.edit_video(
            latents=inputs_latent, latents_path=latent_path_tensor, 
            uncond_embeds=uncond_embeds, negative_embeds=negative_embeds, condition_embeds=condition_embeds,
            n_frames = inputs.n_frames
        )
        save_results(
            MultiModalInput(data=rearrange(results, 'f c h w -> c h w f'), data_type=inputs.data_type), 
            self.args.outputs, 'edit_results'
        )
        print('Done!')

    def init_codi_encoder(self):
        ''' CoDi '''
        origin_path = os.getcwd()

        if not os.path.exists('./third_party/CoDi/checkpoints/CoDi_encoders.pth'):
            assert os.path.exists(self.config.model.codi_checkpoint), "Please download and locate CoDi checkpoint."
            os.system(f'cp {self.config.model.codi_checkpoint} ./third_party/CoDi/checkpoints/CoDi_encoders.pth')

        os.chdir('./third_party/CoDi')
        sys.path.append('.')
        from core.models.model_module_infer import model_module
        model_load_paths = ['CoDi_encoders.pth']

        codi = model_module(data_dir='./checkpoints', pth=model_load_paths).net.to(device=self.device, dtype=self.dtype)
        del(codi.audioldm)
        del(codi.autokl)
        del(codi.optimus)
        del(codi.model)

        os.chdir(origin_path)
        return codi

    def init_sd(self):
        sd_version = self.config.model.sd_version
        if sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        elif sd_version == 'depth':
            model_key = "stabilityai/stable-diffusion-2-depth"
        else:
            raise ValueError(f'Stable-diffusion version {sd_version} not supported.')

        # Create SD models
        vision_decoder = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.dtype).to("cuda")

        # self.vision_decoder.enable_xformers_memory_efficient_attention()
        vision_scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        vision_scheduler.set_timesteps(self.config.n_timesteps, device=self.device)
        # del(self.vision_decoder.text_encoder)
        return vision_decoder, vision_scheduler
 
    #Encode conditions with CoDi encoders
    def encode_condition(self, data, uncond_embeds=None):
        if (data.data_type != 'text') and (uncond_embeds is None):
            assert True, "When you encode non-text condition, you must provide additional information."
        
        if data.data_type in ['image', 'video']:
            print('Using visual editing prompt is not well investigate yet.')
            assert uncond_embeds is not None, "When you wish to use non-text conditions, you must provide text condition for reference."

            with torch.autocast(device_type='cuda', dtype=torch.float32):
                img = Image.open(data.path)
                img = self.codi.clip.processor(images=img, return_tensors="pt")['pixel_values'].to('cuda')
                codi_feature = self.codi.clip.model.vision_model(pixel_values=img)
            
            with torch.autocast(device_type='cuda', dtype=torch.float64):
                codi_feature = self.codi.clip.model.visual_projection(codi_feature.pooler_output)
                # codi_feature = F.normalize(codi_feature, dim=-1)
                codi_feature = codi_feature / codi_feature.norm(p=2, dim=-1, keepdim=True)

                codi_text_feature = uncond_embeds.data_proj / uncond_embeds.data_norm

                noise = torch.randn_like(codi_text_feature)
                u, d = torch.mean(noise), torch.std(noise)
                noise = (noise - u) / d
                u, d = 0, 3e-2
                noise = (noise * d) + u

                codi_feature = codi_text_feature + noise

                clip_feature = self.codi_to_clip_space(codi_feature, uncond_embeds).unsqueeze(0)

            assert data.token_repeat_times is not None, "When you wish to use non-text conditions, you must specify the token repeat times."
            sos_feat, eos_feat = uncond_embeds.data[..., 0:1, :], uncond_embeds.data_pooled
            clip_feature = torch.cat(
                [sos_feat] + [clip_feature] * data.token_repeat_times + [eos_feat] * (77 - data.token_repeat_times - 1),
                dim=1
            ).to(device=self.device, dtype=self.dtype)

            data_encode = copy.deepcopy(data)
            data_encode.data = clip_feature
            return data_encode
            
        #Audio condition
        elif data.data_type == 'audio':
            assert uncond_embeds is not None, "When you wish to use audio conditions, you must provide text condition for reference."
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                ctemp = data.data[None].repeat(1, 1, 1)
                codi_feature = self.codi.clap(ctemp).squeeze(0)
                print(torch.std(codi_feature), torch.mean(codi_feature))

            with torch.autocast(device_type='cuda', dtype=torch.float64):
                clip_feature = self.codi_to_clip_space(codi_feature, uncond_embeds).unsqueeze(0)
                print(torch.std(clip_feature), torch.mean(clip_feature))
                # exit()

            assert data.token_repeat_times is not None, "When you wish to use audio conditions, you must specify the token repeat times."
            sos_feat, eos_feat = uncond_embeds.data[..., 0:1, :], uncond_embeds.data_pooled
            clip_feature = torch.cat(
                [sos_feat] + [clip_feature] * data.token_repeat_times + [eos_feat] * (77 - data.token_repeat_times - 1),
                dim=1
            )
            data_encode = copy.deepcopy(data)
            data_encode.data = clip_feature
            return data_encode

        #Text condition
        elif data.data_type == 'text':
            text = [data.data]

            batch_encoding = self.codi.clip.tokenizer(text, truncation=True, max_length=self.codi.clip.max_length, return_length=True,
                                            return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
            tokens = batch_encoding["input_ids"].to(self.codi.clip.get_device())

            with torch.autocast(device_type='cuda', dtype=torch.float32):
                clip_feature = self.codi.clip.model.text_model(input_ids=tokens)

            with torch.autocast(device_type='cuda', dtype=torch.float64):
                data_encode = copy.deepcopy(data)
                data_encode.data = clip_feature.last_hidden_state
                data_encode.data_pooled = clip_feature.pooler_output.unsqueeze(0)
                data_encode.data_proj = self.codi.clip.model.text_projection(clip_feature.pooler_output)
                data_encode.data_norm = data_encode.data_proj.norm(p=2, dim=-1, keepdim=True)

            return data_encode

    def check_weight_matrix_condition(self, W):
        # Perform Singular Value Decomposition
        U, S, Vh = torch.linalg.svd(W)
        
        # Compute the condition number (ratio of largest to smallest singular value)
        cond_number = S.max() / S.min()
        
        # Check if the smallest singular value is close to zero
        is_singular = torch.isclose(S.min(), torch.tensor(0.0, device=S.device, dtype=S.dtype))
        
        print(f"Condition Number of W: {cond_number.item():.2e}")
        print(f"Is W singular (smallest singular value close to zero)? {'Yes' if is_singular else 'No'}")
        
        # Optionally, print the smallest singular value
        print(f"Smallest Singular Value of W: {S.min().item():.2e}")

    '''
    using tikhonov regularization 
    '''
    def codi_to_clip_space(self, codi_feature, ref_feature, lambda_reg=1e-5):
        # Ensure codi_feature is in the correct dtype and device
        # codi_feature = codi_feature * ref_feature.data_norm
        codi_feature = codi_feature.to(device=self.device, dtype=torch.float64)

        # Extract the weight matrix from the text_projection layer
        W = self.codi.clip.model.text_projection.weight.clone().detach()
        W = W.to(device=self.device, dtype=torch.float64)
        
        # Check the condition number of W
        self.check_weight_matrix_condition(W)
        
        # Prepare W and y_T
        y_T = codi_feature.T  # Shape: (out_features, batch_size)
        y_T = y_T.to(dtype=W.dtype)
        
        # Compute W^T W + lambda * I
        WtW = W.T @ W
        identity = torch.eye(WtW.shape[0], device=WtW.device, dtype=WtW.dtype)
        regularized_matrix = WtW + lambda_reg * identity

        # Compute the right-hand side W^T y_T
        Wt_y_T = W.T @ y_T

        # Solve the regularized linear system
        x_T = torch.linalg.solve(regularized_matrix, Wt_y_T)

        # Transpose back to get x (original features)
        codi_feature_proj = x_T.T  # Shape: (batch_size, in_features)

        return self.adain(codi_feature_proj, ref_feature.data_pooled)

    def adain(self, input_feature, ref_feature):
        #Adain layer.
        u, d = torch.mean(input_feature), torch.std(input_feature)
        input_feature = (input_feature - u ) / d
        u, d = torch.mean(ref_feature), torch.std(ref_feature)
        input_feature = (input_feature * d) + u
        return input_feature