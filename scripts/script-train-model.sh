#!/bin/bash

quote_and_comma_separate() {
    local need_quotes=$1
    shift  # Remove the first argument
    local list=("$@")
    local quoted_list=()
    for item in "${list[@]}"; do
        if [ "$need_quotes" = "true" ]; then
            quoted_list+=("'$item'")
        else
            quoted_list+=("$item")
        fi
    done
    local result="$(IFS=,; echo "${quoted_list[*]}")"
    if [ "$need_quotes" = "true" ]; then
        echo "$result"
    else
        echo $result
    fi
}

exp_path="../experiments/example"
data_path="../datasets/example-data"

model_path="$exp_path/unet"
tokenizer_path="$exp_path/tokenizer"
noise_scheduler_path="$exp_path/scheduler"
lr_scheduler="cosine"
learning_rate=0.006
lr_warmup_steps=200
adam_weight_decay=0.002
num_training_steps=5000
sample_size=64
batch_size=8
train_data="$data_path/lmdb-train"
val_data="$data_path/lmdb-valid"
val_interval=20
val_steps=2
checkpointing_steps=20
output_dir="$exp_path/training"
tracker_project_name="example-experiment"
resume_from_checkpoint="latest"
mixed_precision="no"
gradient_accumulation_steps=1
num_parts=2
threshold=85
loss_weight=1
snr_gamma=5.0
use_ema="True"
label_length=5

cd ./SADE
cmd="python3 train.py\
    --model_path $model_path\
    --tokenizer_path $tokenizer_path\
    --noise_scheduler_path $noise_scheduler_path\
    --num_training_steps $num_training_steps\
    --sample_size $sample_size\
    --batch_size $batch_size\
    --train_data $train_data\
    --val_data $val_data\
    --val_interval $val_interval\
    --val_steps $val_steps\
    --checkpointing_steps $checkpointing_steps\
    --output_dir $output_dir\
    --tracker_project_name $tracker_project_name\
    --lr_scheduler $lr_scheduler\
    --learning_rate $learning_rate\
    --lr_warmup_steps $lr_warmup_steps\
    --adam_weight_decay $adam_weight_decay\
    --gradient_accumulation_steps $gradient_accumulation_steps\
    --resume_from_checkpoint $resume_from_checkpoint\
    --num_parts $num_parts\
    --threshold $threshold\
    --loss_weight $loss_weight\
    --snr_gamma $snr_gamma\
    --use_ema $use_ema\
    --label_length $label_length\
    "

echo $cmd
eval $cmd