from typing import Optional, Tuple
import os
import logging
import argparse
import shutil
import json

from tqdm.auto import tqdm
import numpy as np
from PIL import Image

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms as T

import transformers

import mlflow

import diffusers
from diffusers.training_utils import EMAModel
from diffusers.schedulers import DDPMScheduler
from diffusers.optimization import get_scheduler

from assets.dataset import LMDBImageDataset
from assets.tokenizer import CharacterTokenizer
from assets.unet_2d_condition import UNet2DCharConditionModel
from assets.pipeline import DiffusionPipeline
from assets.utils import (
    compute_snr,
    BooleanAction,
    create_image_grid,
    tensor_to_pil,
    split_images
)

logger = get_logger(__name__, log_level="INFO")

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', required=True)
    parser.add_argument('--tokenizer_path')
    parser.add_argument('--noise_scheduler_path', required=True)
    parser.add_argument('--num_training_steps', type=int, required=True)
    parser.add_argument('--sample_size', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", default=None)
    parser.add_argument('--val_interval', type=int, default=200)
    parser.add_argument('--val_steps', type=int, default=5)   # Number of validation steps to perform per validation
    parser.add_argument('--checkpointing_steps', type=int, default=500)
    parser.add_argument('--checkpoints_total_limit', type=int, default=6)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--logging_dir", default="logs")
    parser.add_argument('--tracker_project_name', default='text-image-diffusion')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lr_scheduler', default="cosine")
    parser.add_argument('--lr_warmup_steps', type=int, default=500)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--adam_weight_decay', type=float, default=2e-2)
    parser.add_argument('--adam_epsilon', type=float, default=1e-2)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--resume_from_checkpoint')
    parser.add_argument('--noise_offset', type=float, default=0)
    parser.add_argument('--input_perturbation', type=float, default=0)
    parser.add_argument('--conditional', action=BooleanAction, default=True)
    parser.add_argument('--mixed_precision', choices=['no', 'fp16'], default='no')
    parser.add_argument('--num_parts', type=int, default=1)
    parser.add_argument('--threshold', type=int, default=255)
    parser.add_argument('--loss_weight', type=float, default=1)
    parser.add_argument('--snr_gamma', type=float, default=None) #Recommended value is 5.0
    parser.add_argument('--use_ema', default=False, action=BooleanAction)
    parser.add_argument('--label_length', type=int, default=0)

    args = parser.parse_args()

    return args


def main():

    args = parse_arguments()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir
        )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with='mlflow',
        project_config=accelerator_project_config,
        mixed_precision=args.mixed_precision,
        # cpu=True,
    )

    print(f"Running on: {accelerator.device}")
    
    # Repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(logging_dir, exist_ok=True)
            os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(logging_dir, 'logs.txt'),
        filemode='w',
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    # Load scheduler, tokenizer and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.noise_scheduler_path)
    unet = UNet2DCharConditionModel.from_pretrained(args.model_path, safe_serialization=False)

    if args.use_ema:
        ema_unet = UNet2DCharConditionModel.from_pretrained(args.model_path, safe_serialization=False)
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DCharConditionModel, model_config=ema_unet.config)

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        if args.use_ema:
            load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DCharConditionModel)
            ema_unet.load_state_dict(load_model.state_dict())
            ema_unet.to(accelerator.device)
            del load_model

        for _ in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            load_model = UNet2DCharConditionModel.from_pretrained(os.path.join(input_dir, "unet"))
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    
    tokenizer = CharacterTokenizer.from_config(args.tokenizer_path, local_files_only=True)

    unet.train()

    # Preprocessing the dataset
    image_transforms = T.Compose(
        [
            T.Resize((args.sample_size, args.sample_size)),
            T.Grayscale(),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ]
    )

    def collate_fn(examples):
        image_data = torch.stack([x[0] for x in examples])
        input_ids = torch.stack([tokenizer(x[1]) for x in examples])
        
        # Check splitting works for training
        image_data, input_ids, prev_parts, part_number = split_images(
            image_data,
            input_ids,
            args.num_parts,
            flatten_batch_dim=True,
            shuffle_output=True,
            device=accelerator.device
        )

        model_kwargs = {
            "image_data": image_data,
            "label_ids": input_ids,
            "part_number": part_number,
            "prev_parts": prev_parts,
        }

        return model_kwargs

    train_dataset = LMDBImageDataset(
        lmdb_path=args.train_data,
        transform=image_transforms,
        label_length=args.label_length,
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    def train_data():
        while True:
            yield from train_dataloader

    val_dataloader = None
    if args.val_data:
        val_dataset = LMDBImageDataset(
            lmdb_path=args.val_data,
            transform=image_transforms,
            label_length=args.label_length,
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        def valid_data():
            while True:
                yield from val_dataloader

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.num_training_steps * accelerator.num_processes,
    )

    unet, optimizer, train_dataloader, lr_scheduler, val_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler, val_dataloader
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    logger.info(json.dumps(dict(vars(args))))

    logger.info("******* Running Training ********")
    logger.info(f"    Sample size = {args.sample_size}")
    logger.info(f"    Num training steps = {args.num_training_steps}")
    logger.info(f"    Batch size = {args.batch_size}")
    logger.info(f"    Gradient accuulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"    Number of training devices = {accelerator.num_processes}")
    
    global_step = 0

    if args.resume_from_checkpoint:
        checkpoint_path = os.path.join(args.output_dir, "checkpoints")
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get more recent checkpoint
            dirs = os.listdir(checkpoint_path)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(checkpoint_path, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.num_training_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process
    )

    if args.val_data:
        best_val_loss = float('inf')
        if os.path.exists(os.path.join(args.output_dir, "BestUnet", 'best-val')):
            with open(os.path.join(args.output_dir, "BestUnet", "best-val"), 'r') as f:
                best_val_loss = float(f.readline())
    
    train_loss = 0.0
    while (global_step < args.num_training_steps):
        model_kwargs = next(train_data())    
        with accelerator.accumulate(unet):
            loss = get_loss(args, unet, noise_scheduler, model_kwargs)

            avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps

            #Backpropagation
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            if args.use_ema:
                ema_unet.step(unet.parameters())
            global_step += 1
            accelerator.log({"train_loss": train_loss}, step=global_step)
            train_loss = 0.0

            if global_step % args.checkpointing_steps == 0:
                if accelerator.is_main_process:
                    checkpoints = os.listdir(os.path.join(args.output_dir, "checkpoints"))
                    checkpoints = [d for d in checkpoints if d.startswith('checkpoint')]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    if len(checkpoints) >= args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(args.output_dir, "checkpoints", removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(args.output_dir, "checkpoints", f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")

                logger.info(f'Sampling at step: {global_step}')
                pipeline = DiffusionPipeline(
                    unet=unet,
                    tokenizer=tokenizer,
                    scheduler=noise_scheduler,
                    sample_size=args.sample_size,
                    num_parts=args.num_parts,
                )
                gen_samples, part_labels = next(pipeline.sample_model(
                    num_samples=9,
                    batch_size=9,
                    data_path=args.val_data if args.val_data else args.train_data,
                    device=accelerator.device,
                    sample_steps=10,
                    label_length=args.label_length,
                ))

                gen_samples = [tensor_to_pil(sample, 'L') for sample in gen_samples]

                grid = create_image_grid(
                    gen_samples,
                    num_rows=3,
                    num_columns=3*args.num_parts,
                    labels=part_labels,
                )

                mlflow.log_image(grid, f"Step-{global_step}-samples.png")
                accelerator.print(f"Sampled at step: {global_step}")
            
            if val_dataloader and global_step % args.val_interval == 0:

                with torch.no_grad():
                    val_loss = 0.0
                    for _ in range(args.val_steps):
                        model_kwargs = next(valid_data())    
            
                        loss = get_loss(args, unet, noise_scheduler, model_kwargs)
                        val_loss += loss.item()

                    val_loss /= args.val_steps

                    accelerator.log({"val_loss": val_loss}, step=global_step)

                if val_loss < best_val_loss:
                    best_path = os.path.join(args.output_dir, "BestUnet")
                    unet.save_pretrained(best_path)
                    accelerator.save_state(best_path)
                    best_val_loss = val_loss
                    with open(os.path.join(best_path, 'best-val'), 'w') as f:
                        f.write(f"{best_val_loss}\n{global_step}")
                        f.flush()

        
        logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)


    accelerator.wait_for_everyone()

    mlflow.log_artifact(os.path.join(args.output_dir, args.logging_dir, "logs.txt"))
    mlflow.end_run()
    accelerator.end_training()

def get_loss(args, unet, noise_scheduler, model_kwargs):
    latents = model_kwargs.pop('image_data')

    noise = torch.randn_like(latents)
    if args.noise_offset:
        noise += args.noise_offset * torch.randn(
            (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
        )

    if args.input_perturbation:
        new_noise = noise + args.input_perturbation * torch.randn_like(noise)
    bsz = latents.shape[0]

    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
    ).long()

    # Forward diffusion process
    if args.input_perturbation:
        noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
    else:
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    if noise_scheduler.config.prediction_type == 'epsilon':
        target = noise
    elif noise_scheduler.config.prediction_type == 'v_prediction':
        target = noise_scheduler.get_velocity(latents, noise, timesteps)

    # sample = noisy_latents
    if args.num_parts > 1:
        sample = torch.cat([noisy_latents, model_kwargs['prev_parts']], dim=1)
    else:
        sample = noisy_latents

    model_pred = unet(
        sample=sample,
        timestep = timesteps,
        class_labels=model_kwargs['part_number'],
        encoder_hidden_states=model_kwargs['label_ids'],
        return_dict=False,
    )[0]

    if args.snr_gamma is None:
        loss = F.mse_loss(model_pred.float(), target.float(), reduction='mean')
    else:
        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
        # This is discussed in Section 4.2 of the same paper.
        snr = compute_snr(noise_scheduler, timesteps)
        mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
            dim=1
        )[0]
        if noise_scheduler.config.prediction_type == "epsilon":
            mse_loss_weights = mse_loss_weights / snr
        elif noise_scheduler.config.prediction_type == "v_prediction":
            mse_loss_weights = mse_loss_weights / (snr + 1)

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
        loss = loss.mean()

    return loss


def disable_loggers():
    loggers = [
        "azureml.mlflow._internal.utils",
        "azure.core.pipeline.policies.http_logging_policy",
        "azure.identity._credentials.chained",
        "azureml.mlflow._common._cloud.cloud",
        "azure.identity._credentials.managed_identity",
    ]

    for log_id in loggers:
        logger = logging.getLogger(log_id)
        logger.setLevel(logging.WARNING)

if __name__ == "__main__":
    disable_loggers()
    main()