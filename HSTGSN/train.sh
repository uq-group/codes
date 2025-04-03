#!/bin/bash

train_ratio=$1

python3 train_hetero.py --need_train=1 --need_residue=0 --need_test=1 --pred_type='ratio' \
    --rnn_name='LSTM' --model_name='hetero_2rnn' --train_dir='anaheim_v10_0' --save_dir='anaheim_v11' \
    --epoch=100 --batch_size=8 --n_sample=100 --train_ratio=$train_ratio

python3 train_homo.py --need_train=1 --need_residue=0 --need_test=1 --pred_type='ratio' \
    --rnn_name='LSTM' --model_name='gat_nr' --train_dir='anaheim_v10_0' --save_dir='anaheim_v11' \
    --epoch=100 --batch_size=8 --n_sample=100 --train_ratio=$train_ratio

python3 train_homo.py --need_train=1 --need_residue=0 --need_test=1 --pred_type='ratio' \
    --rnn_name='LSTM' --model_name='gcn_nr' --train_dir='anaheim_v10_0' --save_dir='anaheim_v11' \
    --epoch=100 --batch_size=8 --n_sample=100 --train_ratio=$train_ratio



