#!/bin/bash
model_name="$1"

echo "Start training：Model name = ${model_name}"

CUDA_VISIBLE_DEVICES=0 torchrun \
  --nproc_per_node=1 \
  --master_addr=127.0.0.1 \
  --master_port=29610 \
  train.py \
  --root /public/lwt/PycharmProjects/Text-IF-main/dataset/ACCV \
  --vis_dir vis \
  --ir_dir ir \
  --text_vis_dir text_vis \
  --text_ir_dir text_ir \
  --out_path "./checkpoint/${model_name}" \
  --batch-size 16 \
  --epochs 120 \
# --use_ddp
