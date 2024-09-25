from typing import Optional, Tuple, List

from tqdm.auto import tqdm

import numpy as np

import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader

from diffusers.schedulers import DDPMScheduler

from assets.dataset import LMDBImageDataset
from assets.tokenizer import CharacterTokenizer
from assets.unet_2d_condition import UNet2DCharConditionModel
from assets.utils import (
    split_images,
    part_length_lookup,
    overlap_lookup,
    get_char_width
)

class DiffusionPipeline:

    def __init__(self,
        unet: UNet2DCharConditionModel,
        tokenizer: CharacterTokenizer,
        scheduler: DDPMScheduler,
        sample_size: int,
        num_parts: int,
    ):
        self.unet = unet
        self.tokenizer = tokenizer 
        self.scheduler = DDPMScheduler.from_config(scheduler.config)
        self.sample_size = sample_size
        self.num_parts = num_parts

    def sample_batch(self,
        batch_size,
        batch_label_ids,
        img_parts,
        device
    ) -> List[torch.Tensor]:
        
        shape = (batch_size,
                self.unet.config.in_channels if self.num_parts == 1 else self.unet.config.in_channels // 2,
                *img_parts.shape[-2:]
                )

        with torch.no_grad():
            gen_image_parts = []
            for i in range(self.num_parts):
                part_ids = batch_label_ids[i] 
                part_number = torch.tensor([i] * batch_size, device=device)

                latents = torch.randn(shape, device=device)
                latents = latents * self.scheduler.init_noise_sigma

                # denoising loop
                for step in tqdm(self.scheduler.timesteps, total=len(self.scheduler.timesteps)):
                    timestep = torch.tensor([step] * batch_size, device=device)

                    latent_model_input = self.scheduler.scale_model_input(latents, timestep)

                    if i == 0:
                        p_parts = torch.zeros_like(latent_model_input)
                    else:
                        p_parts = gen_image_parts[-1]

                    if self.num_parts > 1:
                        sample = torch.cat([latent_model_input, p_parts], dim=1)
                    else:
                        sample = latent_model_input

                    noise_pred = self.unet(
                        sample=sample,
                        timestep=timestep,
                        class_labels=part_number,
                        encoder_hidden_states=part_ids,
                        return_dict=False
                    )[0]

                    latents = self.scheduler.step(noise_pred, step, latents, return_dict=False)[0]
                
                gen_image_parts.append(latents)

        return gen_image_parts


    def sample_model(self,
            num_samples: int,
            batch_size: int,
            data_path: str,
            device: torch.DeviceObjType,
            sample_steps: int = 100,
            new_labels: bool = False,
            pad_parts: bool = False,
            return_part_seeds: bool = False,
            label_length = 0
        ):

        self.unet = self.unet.to(device)

        self.scheduler.set_timesteps(sample_steps)
        
        image_transforms = T.Compose(
            [
                T.Resize((self.sample_size, self.sample_size)),
                T.Grayscale(),
                T.ToTensor(),
                T.Normalize([0.5], [0.5])
            ]
        )

        dataset = LMDBImageDataset(
            data_path,
            transform=image_transforms,
            label_length=label_length
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        data_iter = iter(dataloader)


        labels_used = 0
        with torch.no_grad():
            while labels_used < num_samples:
                imgs, labels = next(data_iter)
                imgs = imgs.to(device)

                bs = min(num_samples - labels_used, len(imgs)) 

                imgs = imgs[:bs]
                if new_labels:
                    batch_labels = [
                        next(self.tokenizer.random_word_gen())
                        for _ in range(bs)]
                else:
                    batch_labels = labels[:bs]
                image_parts, batch_label_ids, _, _ = split_images(
                    imgs=imgs,
                    input_ids=self.tokenizer(batch_labels, max_length=label_length),
                    num_parts=self.num_parts,
                    flatten_batch_dim=False,
                    shuffle_output=False,
                    device=device
                )

                gen_image_parts = self.sample_batch(
                    bs,
                    batch_label_ids,
                    image_parts,
                    device=device
                )
                gen_image_parts = torch.stack(gen_image_parts, dim=1)

                _, _, num_channels, _, part_width = gen_image_parts.shape
                added_width = 0
                overlap = overlap_lookup[label_length][self.num_parts]
                char_width = get_char_width(imgs.shape[3], label_length)

                if self.num_parts > 1:
                    if pad_parts:
                        combined_parts = torch.zeros((
                            bs,
                            self.num_parts, 
                            num_channels,
                            self.sample_size,
                            self.sample_size
                        ), device=device)

                        for i in range(self.num_parts):
                            combined_parts[:, i, :, :, added_width:(added_width + part_width)] = gen_image_parts[:,i]
                            added_width = added_width + part_width - overlap * char_width

                        gen_image_parts = torch.stack([combined_parts[i//self.num_parts,i%self.num_parts,...] \
                            for i in range(self.num_parts * len(combined_parts))])

                        batch_labels = [labels[i//self.num_parts] for i in range(self.num_parts * len(gen_image_parts))]
                    else:

                        stacked_parts = torch.zeros((
                            bs,
                            num_channels,
                            self.sample_size,
                            self.sample_size,
                        ), device=device)

                        for i in range(self.num_parts):
                            stacked_parts[:, :, :, added_width:(added_width + part_width)] = gen_image_parts[:,i]
                            added_width = added_width + part_width - overlap * char_width

                        gen_image_parts = stacked_parts
                        batch_labels = labels
                else:
                    gen_image_parts = gen_image_parts.squeeze(1)

                labels_used += bs

                output = [gen_image_parts, batch_labels]
                if return_part_seeds:
                    yield output + [image_parts]
                else:
                    yield output

    def get_num_part_ids(num_chars, num_segments):
        num_part_ids = int(np.ceil(float(num_chars) / num_segments))
        if num_chars == 5 and num_segments == 3:
            num_part_ids = 3

        return num_part_ids