#!/bin/sh
cd ..

train_size=$1
miss=$2

python test_hetero_od_incomplete.py --map_name='ANAHEIM' --gpu=1 \
--train_data_dir_list 'data_ANAHEIM_minor_00_cc' 'data_ANAHEIM_moderate_00_cc' 'data_ANAHEIM_major_00_cc' \
--train_num_sample_list 3000 3000 3000 --train_ratio=$train_size --test_ratio=0.2 \
--model_idx=13 --batch_size=128 --epoch=200 --conservation_loss=0 --loss=1 --miss=$miss --test_miss=0.2
