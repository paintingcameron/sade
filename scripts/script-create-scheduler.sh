#!/bin/bash

scheduler_type="ddim"
num_training_timesteps=1000
variance_type="fixed_small"
beta_schedule="squaredcos_cap_v2"
experiment_name="example-experiment"
prediction_type="v_prediction"

save_dir="../experiments/example/scheduler"

cd ./SADE
cmd="python3 create_scheduler.py\
    --scheduler_type $scheduler_type\
    --num_training_timesteps $num_training_timesteps\
    --variance_type $variance_type\
    --beta_schedule $beta_schedule\
    --save_dir $save_dir\
    --experiment_name $experiment_name
    --prediction_type $prediction_type \
    "

echo $cmd
eval $cmd