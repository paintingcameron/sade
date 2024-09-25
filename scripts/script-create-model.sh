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

exp_dir="../experiments"

in_channels=2
out_channels=1
down_block_types=(
    "DownBlock2D"
    "CrossAttnDownBlock2D"
    "CrossAttnDownBlock2D"
)
up_block_types=(
    "CrossAttnUpBlock2D"
    "CrossAttnUpBlock2D"
    "UpBlock2D"
)
block_out_channels=(64 128 256)
dropout=0.0
char_embed_dim=16
encoder_hid_dim=16
num_unique_chars=10
attention_head_dim=16
cross_attention_dim=48     # Must be char_embed_dim * (number of chars per image part)
time_embedding_type="positional"
addition_embed_type=""
class_embed_type=""

num_unique_chars=$((num_unique_chars + 1)) # Add default char

save_dir="$exp_dir/example/unet"
experiment_name="example-experiment"

down_block_types="$(quote_and_comma_separate "true" "${down_block_types[@]}")"
up_block_types="$(quote_and_comma_separate "true" "${up_block_types[@]}")"
block_out_channels="$(quote_and_comma_separate "false" "${block_out_channels[@]}")"

cd ./SADE
cmd="python3 create_model.py\
    --in_channels $in_channels\
    --out_channels $out_channels\
    --down_block_types \"($down_block_types)\"\
    --up_block_types \"($up_block_types)\"\
    --block_out_channels \"($block_out_channels)\"\
    --dropout $dropout\
    --encoder_hid_dim $encoder_hid_dim\
    --attention_head_dim $attention_head_dim\
    --time_embedding_type $time_embedding_type\
    --char_embed_dim $char_embed_dim\
    --cross_attention_dim $cross_attention_dim\
    --num_unique_chars $num_unique_chars\
    --save_dir $save_dir\
    --experiment_name $experiment_name\
"

echo $cmd
eval $cmd