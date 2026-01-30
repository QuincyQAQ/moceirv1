#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --multi_gpu --num_processes 2 --main_process_port 29502 \
  test.py