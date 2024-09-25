
#!/bin/bash

alphabet="ALMQX23456"
max_length=7
experiment_name="example-experiment"

save_dir="../experiments/example/tokenizer"

cd ./SADE
cmd="python3 create_tokenizer.py \
    --alphabet $alphabet\
    --max_length $max_length\
    --save_dir $save_dir\
    --experiment_name $experiment_name\
    "

echo $cmd
eval $cmd
