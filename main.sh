#! /bin/bash

python main.py \
  --prompt "Black Leopard" \
  --source "Lion" --init_image "input_example/lion1.jpg" --output_path "../result/output_leopard" \
  --use_range_restart \
  --seed 42 \
  --use_noise_aug_all \
  --regularize_content \
  --model_output_size 256 \
  --skip_timesteps 40 \
  --diff_iter 100 \
  --iterations_num 10