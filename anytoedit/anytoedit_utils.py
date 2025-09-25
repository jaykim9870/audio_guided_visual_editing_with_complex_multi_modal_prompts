import os
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from PIL import Image
from einops import rearrange, repeat
import difflib

import torch
import torchaudio
import torchvision
import torchvision.transforms as T
from torchvision.io import read_video,write_video

totensor = torchvision.transforms.ToTensor()
topilimage = torchvision.transforms.ToPILImage()

@dataclass
class MultiModalInput:
    data: Union[str, torch.Tensor] = None
    data_type: str = None
    path: str = None
    n_frames: int = None
    cfg_scale: float = None
    token_repeat_times: int = None

def load_data(path, data_type, im_size=512, device='cuda', dtype=torch.float16, n_frames=16):
    if data_type=='text':
        data = path
    elif data_type=='image':
        data = load_image(path, im_size=im_size).to(device=device, dtype=dtype)
        data = rearrange(data, 'c h w -> c h w 1')
    elif data_type=='video':
        filenames = sorted(os.listdir(path))
        filenames = [f for f in filenames if f.split('.')[-1] in ['png', 'jpg', 'jpeg']]
        
        filenames = filenames[0::int(len(filenames)//n_frames)][:n_frames]
        data = [
            load_image(os.path.join(path,filename), im_size=im_size).to(device=device, dtype=dtype)
            for filename in filenames
        ]
        data = torch.stack(data, dim=-1)
        
    elif data_type=='audio':
        data, sr = torchaudio.load(path)
        if sr != 16000:
            data = torchaudio.functional.resample(waveform=data, orig_freq=sr, new_freq=16000)
        data = data[:int(16000 * 10.0)]
        if data.shape[0]!=1:
            data = torch.mean(data, dim=0).unsqueeze(0).to(device=device, dtype=dtype)
    return data

#Read raw data into tensor
def prepare_input(config, global_config):
    data_type = config.type
    path = config.data
    n_frames = config.n_frames if hasattr(config, "n_frames") else 1
    cfg_scale = config.cfg_scale if hasattr(config, "cfg_scale") else None
    token_repeat_times = config.token_repeat_times if hasattr(config, "token_repeat_times") else None

    device = global_config.device
    dtype = torch.float16 if global_config.precision==16 else torch.float32

    data = load_data(path, data_type, im_size=global_config.model.image_size, device=device, dtype=dtype, n_frames=n_frames)

    return MultiModalInput(data=data, data_type=data_type, path=path, n_frames=n_frames, cfg_scale=cfg_scale, token_repeat_times=token_repeat_times)

def load_image(image_path, left=0, right=0, top=0, bottom=0, im_size=512):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = totensor(Image.fromarray(image).resize((im_size, im_size)))
    return image

@torch.no_grad()
def encode_latents(vae, x, config, data_type='image', stft=None):
    batch_size = config.model.model_image.vae_batch_size
    deterministic = config.model.model_image.vae_deterministic

    with torch.autocast(device_type='cuda', dtype=torch.float32):
        if data_type in ['image', 'video']:
            x = 2 * x - 1
            latents = []
            for i in range(0, len(x), batch_size):
                posterior = vae.encode(x[i:i + batch_size]).latent_dist
                latent = posterior.mean if deterministic else posterior.sample()
                latents.append(latent * 0.18215)
            latents = torch.cat(latents)
    return latents

@torch.no_grad()
def decode_latents(vae, latents, data_type='image'):
    with torch.autocast(device_type='cuda', dtype=torch.float32):
        if data_type in ['image', 'video']:
            decoded = []
            batch_size = 8
            for b in range(0, latents.shape[0], batch_size):
                latents_batch = 1 / 0.18215 * latents[b:b + batch_size]
                imgs = vae.decode(latents_batch).sample
                imgs = (imgs / 2 + 0.5).clamp(0, 1)
                decoded.append(imgs)
            return torch.cat(decoded)
        elif data_type == 'audio':
            # latents = 1 / vae.config.scaling_factor * latents
            mel_spectrogram = vae.decode(latents).sample
            return mel_spectrogram

def save_results(x_input, output_path, suffix):
    x, t = x_input.data, x_input.data_type
    if t=='image':
        if type(x)==np.ndarray:
            x = Image.fromarray(x)
        if type(x)==torch.Tensor:
            # x = topilimage(x.squeeze())
            x = Image.fromarray(np.uint8(255*np.array(x[..., 0].permute(1,2,0).cpu().detach())))
        x.save(os.path.join(output_path, suffix+'.png'))
    elif t=='video':
        save_video(
            rearrange(x, 'c h w f -> f c h w'),
            os.path.join(output_path, suffix+'.mp4'),
            fps=10
        )

    elif t=='audio':
        if x.shape[0]==1:
            x = torch.cat([torch.Tensor(x)]*2, dim=0)
        torchaudio.save(os.path.join(output_path, suffix+'.wav'), x.to(device='cpu', dtype=torch.float32), sample_rate=16000)

def save_video(raw_frames, save_path, fps=10):
    video_codec = "libx264"
    video_options = {
        "crf": "18",  # Constant Rate Factor (lower value = higher quality, 18 is a good balance)
        "preset": "slow",  # Encoding preset (e.g., ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
    }

    frames = (raw_frames * 255).to(torch.uint8).cpu().permute(0, 2, 3, 1)
    write_video(save_path, frames, fps=fps, video_codec=video_codec, options=video_options)

def find_differences_str(str1, str2):
    # Split both strings into lists of words
    words1 = str1.split()
    words2 = str2.split()
    
    # Use difflib to get a list of differences
    diff = list(difflib.ndiff(words1, words2))

    differences = []
    group = []
    for line in diff:
        if line.startswith('  '):
            if group:
                # Process the group of differences
                plus_words = [w[2:] for w in group if w.startswith('+ ')]
                minus_words = [w[2:] for w in group if w.startswith('- ')]
                if plus_words:
                    differences.extend(plus_words)
                elif minus_words:
                    differences.extend(minus_words)
                group = []
        else:
            group.append(line)
    # Process any remaining group
    if group:
        plus_words = [w[2:] for w in group if w.startswith('+ ')]
        minus_words = [w[2:] for w in group if w.startswith('- ')]
        if plus_words:
            differences.extend(plus_words)
        elif minus_words:
            differences.extend(minus_words)
    
    if not differences:
        print(str1, str2)
        assert False, "Problem occurred!"
        return "No differences found."

    # Join the list of differences into a single string
    return ' '.join(differences)