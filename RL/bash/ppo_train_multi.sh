#!/bin/bash

cd ..

# ############################ curriculum learning setting ############################
# python train_ppo_multi.py --load_model="None" --map_name "Manhattan" "Manhattan" "Manhattan" "Manhattan" \
#     --time_interval_list 2.0 2.0 2.0 2.0 --graph_size_list 15 150 600 2000 --agent_type "MLP" \
#     --num_stop_action_list 20 40 80 200 --num_episodes_list 1000 2000 2000 0 --lr=0.00001 --gamma=0.9