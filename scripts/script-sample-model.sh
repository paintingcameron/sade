#!/bin/bash

cd "./src"

exp_path="../experiments/test"
data_path="../datasets"

training_dir="$exp_path/training"
tokenizer_path="$exp_path/tokenizer"
model_path="$exp_path/unet"
scheduler_path="$exp_path/scheduler"
data_path="$data_path/lmdb-test"

sample_size=120
batch_size=16
checkpoint="latest"
num_samples=50
num_diffusion_timesteps=20
output_dir="$exp_path/sampling"
tracker_project_name="test-experiment"
seed=43
deterministic="False"
mixed_precision="no"
num_segments=2
use_ema="True"
pad_outputs="False"
label_length=6

cmd="python sample.py\
    --model_path $model_path\
    --tokenizer_path $tokenizer_path\
    --noise_scheduler_path $scheduler_path\
    --sample_size $sample_size\
    --batch_size $batch_size\
    --checkpoint $checkpoint\
    --num_samples $num_samples\
    --num_diffusion_timesteps $num_diffusion_timesteps\
    --training_dir $training_dir\
    --output_dir $output_dir\
    --tracker_project_name $tracker_project_name\
    --seed $seed\
    --deterministic $deterministic\
    --mixed_precision $mixed_precision\
    --num_segments $num_segments\
    --use_ema $use_ema\
    --data_path $data_path\
    --pad_outputs $pad_outputs\
    --label_length $label_length\
    "
    # --resume_from_checkpoint $resume_from_checkpoint\
    # "
    # --sample_words \"[ 'AAAAQX', 'QAMXAM', 'QAXAXX', 'MLAQQA' ]\"\

echo $cmd
eval $cmd