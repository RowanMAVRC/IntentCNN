#!/bin/bash

# nohup 
python3 cnn_main.py \
  --device "cuda:1" \
  --data_path IntentCNN/Useable/XY/800pad_0 \
  --run_name 800pad_0 \
  --num_epochs 20\
  --project_name IntentCNN_XY # > 800XY.out &
