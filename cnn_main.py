"""
 _____       _             _                           
|_   _|     | |           | |    ____  _   _  _   _                           
  | |  _ __ | |_ ___ _ __ | |_  / ___|| \ | || \ | |
  | | | '_ \| __/ _ \ '_ \| __|| |   ||  \| ||  \| |
 _| |_| | | | ||  __/ | | | |_ | |___|| |\  || |\  |
|_____|_| |_|\__\___|_| |_|\__| \____||_| \_||_| \_|

## Summary
This script sets up an environment for training a CNN-based deep learning model to classify flight trajectories.
It utilizes PyTorch, sklearn, and Wandb for cross-validation and tracking. The data consists of flight trajectories,
and the model predicts the intention of the object based on these trajectories. The script includes functions for 
loading and preprocessing data, defining the model architecture, training the model, evaluating performance, and 
running inference.
"""

# ------------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------------- #

# This needs to be done before other imports
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Python Imports
import argparse
import yaml
import time

# Package Imports
import numpy as np
import torch
import wandb
from sklearn.model_selection import KFold

# File Imports
from tools.load_data import load_flight_data
from cfgs.models.intentCNN import train_cnn, inference, prepare_dataloader
from tools.utils import generate_histogram_and_pie_chart, generate_histogram_and_pie_chart_for_split

# ------------------------------------------------------------------------------------- #
# Functions
# ------------------------------------------------------------------------------------- #

def main():
    debug = True
    
    if debug:
        data_path = "/data/TGSSE/UpdatedIntentions/XYZ/400pad_66"
        labels_path = "/data/TGSSE/UpdatedIntentions/labels.yaml"
        label_detailed_path = "/data/TGSSE/UpdatedIntentions/XYZ/800pad_66/trajectory_with_intentions_800_pad_533_min_151221_label_detailed.yaml"
        num_epochs = 100
        augment = True
        batch_size = 8
        n_kfolds = 5
        project_name = "cnn_trajectory_classification"
        run_name = "400pad_66"
        
    else:
        parser = argparse.ArgumentParser(description='Flight Trajectory Classification')
        parser.add_argument('--data_path', type=str, help='Path to trajectory data: Directory containing .pt files')
        parser.add_argument('--labels_path', type=str, help='Path to labels data: .yaml file')
        parser.add_argument('--label_detailed_path', type=str, help='Path to detailed labels data: .yaml file')
        parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
        parser.add_argument('--augment', type=str, default='False', help='Whether to augment the data')
        parser.add_argument('--n_kfolds', type=int, default=5, help='Number of splits for KFold cross-validation')
        parser.add_argument('--project_name', type=str, default='cnn_trajectory_classification', help='Project name for wandb')
        parser.add_argument('--run_name', type=str, default='mobilebert_run', help='Run name for wandb')
        parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
        args = parser.parse_args()
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_path = args.data_path
        labels_path = args.labels_path
        label_detailed_path = args.label_detailed_path
        num_epochs = args.num_epochs
        augment = args.augment.lower() in ['true', '1', 't', 'y', 'yes']
        n_kfolds = args.n_kfolds
        project_name = args.project_name
        run_name = f"{args.run_name}_augment" if augment else args.run_name
        batch_size = args.batch_size
        
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")       
    
    wandb.init(project=project_name, name=f"Overall_Labels_{run_name}")
    
    flight_data, flight_labels, id2label, label2id, id2label_detailed = load_flight_data(data_path, labels_path, label_detailed_path, augment)
    print(f"Total trajectories: {len(flight_data)}")
    
    test_size = flight_data.shape[0] // 10
    print(f"Test set size: {test_size}")
    indices = np.arange(len(flight_data))
    np.random.shuffle(indices)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    test_data = flight_data[test_indices]
    test_labels = np.array(flight_labels)[test_indices]
    flight_data = flight_data[train_indices]
    flight_labels = np.array(flight_labels)[train_indices]
    
    stat_img = generate_histogram_and_pie_chart(flight_labels, id2label, data_path)
    wandb.log({"Overall Stats": [wandb.Image(stat_img)]})
    
    wandb.finish()
    
    kf = KFold(n_splits=n_kfolds, shuffle=True, random_state=42)
    
    models = {}
    
    print("Starting training of CNN model with K-fold cross-validation...\n")
    
    total_start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(flight_data)):
        print(f"\nFold {fold+1}/{kf.n_splits}\n")
        
        fold_start_time = time.time()
        
        wandb.init(project=project_name, name=f"{run_name}_fold{fold+1}")
        
        train_trajectories, val_trajectories = flight_data[train_idx], flight_data[val_idx]
        train_labels, val_labels = np.array(flight_labels)[train_idx], np.array(flight_labels)[val_idx]
        print(f"Training set size: {len(train_trajectories)}")
        print(f"Validation set size: {len(val_trajectories)}")
        
        split_stat = generate_histogram_and_pie_chart_for_split(train_labels, val_labels, id2label, f'{run_name}_fold{fold+1}')
        wandb.log({"Split Distribution": wandb.Image(split_stat)})
        
        models[f'CNN_fold_{fold}'] = train_cnn(train_trajectories, train_labels, val_trajectories, val_labels, fold, 'CNN', num_epochs=num_epochs)
        
        fold_end_time = time.time()
        fold_time = fold_end_time - fold_start_time
        print(f"Time taken for fold {fold+1}: {fold_time:.2f} seconds")
        
        print(f"\nPerforming inference with CNN_fold_{fold} model...")
        
        inference_start_time = time.time()
        preds = inference(models[f'CNN_fold_{fold}'], prepare_dataloader(test_data, test_labels, batch_size=batch_size, shuffle=False))
        inference_end_time = time.time()
        inference_time = inference_end_time - inference_start_time
        avg_inference_time = inference_time / len(test_labels)

        print(f"Predictions: {preds[:10]}")
        print(f"True Labels: {test_labels[:10].tolist()}")
        accuracy = np.mean(preds == test_labels)
        print(f"Accuracy of CNN_fold_{fold} model: {accuracy:.4f}")
        print(f"Inference time for fold {fold+1}: {inference_time:.2f} seconds")
        print(f"Average time per guess: {avg_inference_time:.4f} seconds")

        wandb.log({f"inference_accuracy": accuracy,
                   f"inference_time": inference_time,
                   f"avg_time_per_guess": avg_inference_time})

        wandb.finish()
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    avg_epoch_time = total_time / (num_epochs * n_kfolds)
    
    print(f"\nTotal time taken for training: {total_time:.2f} seconds")
    print(f"Average time per epoch: {avg_epoch_time:.2f} seconds")
    print("\nFinished training CNN model with K-fold cross-validation.")

# ------------------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()
