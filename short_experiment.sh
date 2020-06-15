#!/bin/sh

# this is an example that allows easily trying out the code without creating configuration files

python label_distillation_or2.py \
--epochs 5 \
--batch_size 256 \
--inner_batch_size 50 \
--num_base_examples 100 \
--expname short_experiment \
--source emnist \
--target mnist \
--balanced_source True \
--baseline False \
--resnet False \
--num_steps_analysis False \
--test_various_models False \
--target_set_size 50000 \
--random_seed 1234 \
--label_smoothing 0 \
--inner_lr 0.01