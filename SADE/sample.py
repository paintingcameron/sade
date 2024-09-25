
import os
import argparse
import logging
import ast
from io import BytesIO
from math import ceil
from random import shuffle

import numpy as np

import lmdb
from PIL import Image
from tqdm import tqdm
import mlflow

import torch

import transformers

import diffusers
from diffusers.models import AutoencoderKL
from diffusers.schedulers import DDPMScheduler
from diffusers.training_utils import EMAModel

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from assets.tokenizer import CharacterTokenizer
from assets.unet_2d_condition import UNet2DConditionModel
from assets.pipeline import DiffusionPipeline
from assets.utils import (
    create_parts_grid,
    write_batch_to_cache,
    write_cache,
    tensor_to_pil
)

logger = get_logger(__name__, log_level="INFO")

class TupleAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            setattr(namespace, self.dest, ast.literal_eval(values))
        except:
            parser.error(f"Invalid tuple argument: {values}")

class BooleanAction(argparse.Action):
    def __call__(self, parser, namespace, value, option_string=None):
        try:
            v = None
            if isinstance(value, bool):
                v = value
            if value.lower() in ("yes", "true", "t", "y", "1"):
                v = True
            elif value.lower() in ("no", "false", "f", "n", "0"):
                v = False
            else:
                raise argparse.ArgumentTypeError("boolean value expected")

            setattr(namespace, self.dest, v)
        except:
            parser.error(f"Invalid boolean argument {value}")
    
def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', required=True)
    parser.add_argument('--tokenizer_path', required=True)
    parser.add_argument('--noise_scheduler_path', required=True)
    parser.add_argument("--sample_size", type=int, default=128)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--checkpoint', default="latest")
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--logging_dir", default='sample-logs')
    parser.add_argument("--num_diffusion_timesteps", type=int, default=1000)
    parser.add_argument("--sample_words", action=TupleAction)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action=BooleanAction, default=False)
    parser.add_argument('--tracker_project_name', default="diffusion-sampling-exp")
    parser.add_argument('--training_dir', required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mixed_precision", choices=['no', 'fp16'], default='no')
    parser.add_argument("--num_segments", type=int, default=1)
    parser.add_argument('--use_ema', action=BooleanAction, default=False)
    parser.add_argument('--pad_outputs', action=BooleanAction, default=False)
    parser.add_argument('--label_length', type=int, default=0)

    args = parser.parse_args()

    return args


def main():

    args = parse_arguments()
    
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        log_with='mlflow',
        project_config=accelerator_project_config,
        mixed_precision=args.mixed_precision,
        # cpu=True
    )
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(logging_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(logging_dir, 'logs.txt'),
        filemode='w',
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)
    logger.info(f"Cuda version: {torch.version.cuda}")
    logger.info(f"GPU available: {torch.cuda.is_available()}")
    logger.info(torch.__version__)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    logger.info("\n".join([f"{key}:\t{value}" for key, value in args.__dict__.items()]))

    if args.seed is not None:
        set_seed(args.seed)

    logger.info("Loading DDPM scheduler")

    noise_scheduler = DDPMScheduler.from_pretrained(args.noise_scheduler_path)

    assert args.tokenizer_path is not None
    logger.info("Loading tokenizer")
    tokenizer = CharacterTokenizer.from_config(args.tokenizer_path)

    if args.checkpoint == "latest":
        assert args.training_dir is not None
        dirs = os.listdir(os.path.join(args.training_dir, "checkpoints"))
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
    
        assert path is not None, "No checkpoint found in training directory"
        
        checkpoint_path = os.path.join(args.training_dir, "checkpoints", path)
    else:
        checkpoint_path = args.checkpoint

    accelerator.print(f"Sampling from checkpoint {checkpoint_path}")
    logger.info(f"Sampling from checkpoint{checkpoint_path}")

    unet = UNet2DConditionModel.from_pretrained(os.path.join(checkpoint_path, "unet"))
    if args.use_ema:
        ema_unet = EMAModel.from_pretrained(os.path.join(checkpoint_path, "unet_ema"), UNet2DConditionModel)
        ema_unet.copy_to(unet.parameters())
    logger.info("Loaded state")
    unet = accelerator.prepare(unet)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        logger.info("Initiating trackers")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)
        logger.info("Tracker initiated")

    logger.info("*********** Running Sampling ************")
    logger.info(f"   Number of samples = {args.num_samples}")
    logger.info(f"   Batch size = {args.batch_size}")
    
    env = lmdb.open(os.path.join(args.output_dir, "samples"), map_size=1099511627776)
    cache = {}

    pipeline = DiffusionPipeline(
        unet=unet,
        tokenizer=tokenizer,
        scheduler=noise_scheduler,
        sample_size=args.sample_size,
        num_parts=args.num_segments,
    )

    num_sampled = 0
    try:
        sample_generator = pipeline.sample_model(
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            data_path=args.data_path,
            device=accelerator.device,
            sample_steps=args.num_diffusion_timesteps,
            pad_parts=args.pad_outputs,
            label_length=args.label_length,
        )
        while True:
            gen_images, labels = next(sample_generator)

            images = (gen_images / 2 + 0.5).clamp(0,1)
            images = images.cpu().permute(0, 2, 3, 1).numpy()
            images = (images * 255).astype("uint8")

            write_batch_to_cache(labels, images, cache, num_sampled)

            num_sampled += len(images) 
            accelerator.print(f"Sampled: {num_sampled}")
        
            if len(cache) > 500:
                cache['num-samples'.encode()] = str(num_sampled).encode()
                write_cache(env, cache)
                accelerator.print(f"Written {len(cache)} samples to env")
                cache = {}

    except StopIteration:
        pass

    cache['num-samples'.encode()] = str(num_sampled).encode()
    write_cache(env, cache)
    logger.info(f"Written {len(cache)} samples to env")
    
    mlflow.log_artifact(os.path.join(logging_dir, "logs.txt"))

    mlflow.end_run()
    env.close()
    torch.cuda.empty_cache()

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