#!/bin/bash

# Declare an associative array for devices and pad configurations
declare -A configs=(
  ["800pad_0"]="1" ["700pad_0"]="2" ["600pad_0"]="3" ["500pad_0"]="4" ["400pad_0"]="5" ["300pad_0"]="6" ["200pad_0"]="7" ["100pad_0"]="1"
  ["800pad_33"]="2" ["700pad_33"]="3" ["600pad_33"]="4" ["500pad_33"]="5" ["400pad_33"]="6" ["300pad_33"]="7" ["200pad_33"]="2" ["100pad_33"]="3"
  ["800pad_66"]="4" ["700pad_66"]="5" ["600pad_66"]="4" ["500pad_66"]="5" ["400pad_66"]="6" ["300pad_66"]="7" ["200pad_66"]="1" ["100pad_66"]="2"
)

# General execution loop for most configurations
for pad in "${!configs[@]}"
do
  device="${configs[$pad]}"
  nohup python3 cnn_main.py --device "cuda:${device}" --data_path "/data/TGSSE/UpdatedIntentions/XY/${pad}" --run_name "${pad}" --project_name "IntentCNN_XY" > "${pad}XY.out" &
done

# Special configurations with kernel sizes
declare -a kernels=(2 4 6 10 12 14 16 18 20 30 40 50 60 70 80)
for kernel in "${kernels[@]}"
do
  nohup python3 cnn_main.py --device "cuda:5" --kernel_size "${kernel}" --data_path "/data/TGSSE/UpdatedIntentions/XY/800pad_66" --run_name "800pad66_kernel${kernel}" --project_name "IntentCNN_XY_KERNEL" > "800pad66XY_kernel${kernel}.out" &
done

echo "All commands have been executed in the background."
