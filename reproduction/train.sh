#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/Tina
source reproduction/env.sh
mode="${1:-main}"
if [[ "$mode" == smoke ]]; then
  export TINA_RUN_NAME=smoke_open_rs3
  accelerate launch --config_file recipes/accelerate_ds_cfgs/ds_zero2.yaml \
    --num_processes 2 --main_process_port 29501 tina/post_train_hf/grpo.py \
    --config reproduction/open_rs3.yaml --max_steps 2 --max_completion_length 512 \
    --gradient_accumulation_steps 1 --save_steps 2 --warmup_ratio 0.0 --cosine_max_len 3584
elif [[ "$mode" == main ]]; then
  export TINA_RUN_NAME=grpo_open_rs3
  export TINA_STOP_STEP=850
  accelerate launch --config_file recipes/accelerate_ds_cfgs/ds_zero2.yaml \
    --num_processes 2 --main_process_port 29501 tina/post_train_hf/grpo.py \
    --config reproduction/open_rs3.yaml --cosine_max_len 3584
else
  echo "Usage: bash reproduction/train.sh [smoke|main]" >&2
  exit 2
fi
