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

import argparse
import time

import numpy as np
import torch
import wandb
from sklearn.model_selection import KFold

from tools.load_data import load_flight_data_single
from intentCNN import train_cnn, inference, prepare_dataloader
from tools.utils import generate_histogram_and_pie_chart, generate_histogram_and_pie_chart_for_split

# ------------------------------------------------------------------------------------- #
# Functions
# ------------------------------------------------------------------------------------- #

def parse_arguments():
    """
    Parse command-line arguments or use default debug configuration.

    Returns:
    - args (argparse.Namespace): Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Flight Trajectory Classification')
    parser.add_argument('--data_path', type=str, default="/data/TGSSE/UpdatedIntentions/XY/100pad_0", help='Path to trajectory data (.pt files).')
    parser.add_argument('--intent_labels', type=str, default='/data/TGSSE/UpdatedIntentions/intent_labels.yaml', help='Path to intent labels (.yaml file).')
    parser.add_argument('--detect_labels', type=str, default='/data/TGSSE/UpdatedIntentions/detect_labels.yaml', help='Path to detailed detection labels (.yaml file).')
    parser.add_argument('--num_epochs', type=int, default=2000, help='Number of training epochs.')
    parser.add_argument('--augment', type=str, default='False', help='Flag to enable data augmentation.')
    parser.add_argument('--n_kfolds', type=int, default=5, help='Number of splits for K-Fold cross-validation.')
    parser.add_argument('--project_name', type=str, default='cnn_trajectory_classificationXY', help='Project name for Wandb tracking.')
    parser.add_argument('--run_name', type=str, default='100pad_0', help='Run name for Wandb tracking.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--kernel_size', type=int, default=8, help='Kernel size for CNN model.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., "cuda" or "cpu").')
    return parser.parse_args()

def load_and_split_data(data_path, intent_labels, detect_labels, augment, test_size_ratio=0.1):
    """
    Load and split data into training and test sets.

    Parameters:
    - data_path (str): Path to the data files.
    - intent_labels (str): Path to the intent labels file.
    - detect_labels (str): Path to the detection labels file.
    - augment (bool): Whether to apply data augmentation.
    - test_size_ratio (float): Proportion of data to be used for testing.

    Returns:
    - train_data (numpy.ndarray): Training data.
    - train_labels (numpy.ndarray): Training labels.
    - test_data (numpy.ndarray): Test data.
    - test_labels (numpy.ndarray): Test labels.
    - id2label (dict): Mapping from id to label.
    """
    flight_data, flight_labels, _, id2label, _, _, _ = load_flight_data_single(
        data_path, intent_labels, detect_labels, augment, kwargs={"x_max": 480, "y_max": 480}
    )
    print(f"Total trajectories: {len(flight_data)}")

    test_size = int(len(flight_data) * test_size_ratio)
    print(f"Test set size: {test_size}")

    indices = np.arange(len(flight_data))
    np.random.shuffle(indices)

    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    train_data = flight_data[train_indices]
    train_labels = np.array(flight_labels)[train_indices]
    test_data = flight_data[test_indices]
    test_labels = np.array(flight_labels)[test_indices]

    return train_data, train_labels, test_data, test_labels, id2label

def train_and_evaluate(train_data, train_labels, test_data, test_labels, id2label, config):
    """
    Train and evaluate the CNN model using K-Fold cross-validation.

    Parameters:
    - train_data (numpy.ndarray): Training data.
    - train_labels (numpy.ndarray): Training labels.
    - test_data (numpy.ndarray): Test data.
    - test_labels (numpy.ndarray): Test labels.
    - id2label (dict): Mapping from id to label.
    - config (dict): Configuration dictionary for WandB and model settings.
    """
    wandb.init(project=config["project_name"], group=config['run_name'], name=f"Overall_Labels_{config['run_name']}", config=config)
    
    stat_img = generate_histogram_and_pie_chart(train_labels, id2label, config["data_path"])
    wandb.log({"Overall Stats": [wandb.Image(stat_img)]})
    wandb.finish()

    kf = KFold(n_splits=config["n_kfolds"], shuffle=True, random_state=42)
    models = {}

    print("Starting training of CNN model with K-fold cross-validation...\n")

    total_start_time = time.time()

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
        fold_start_time = time.time()
        print(f"\nFold {fold + 1}/{kf.n_splits}\n")

        wandb.init(project=config["project_name"], group=config['run_name'], name=f"{config['run_name']}_fold{fold + 1}", config=config)

        train_trajectories, val_trajectories = train_data[train_idx], train_data[val_idx]
        train_fold_labels, val_fold_labels = train_labels[train_idx], train_labels[val_idx]

        print(f"Training set size: {len(train_trajectories)}")
        print(f"Validation set size: {len(val_trajectories)}")

        split_stat = generate_histogram_and_pie_chart_for_split(
            train_fold_labels, 
            val_fold_labels, 
            id2label, 
            f'{config["run_name"]}_fold{fold + 1}'
        )

        wandb.log({"Split Distribution": wandb.Image(split_stat)})

        models[f'CNN_fold_{fold}'] = train_cnn(
            train_trajectories, 
            train_fold_labels, 
            val_trajectories, 
            val_fold_labels, 
            fold, 
            f'{config["run_name"]}',
            num_epochs=config["num_epochs"], 
            batch_size=config["batch_size"], 
            kernel_size=config["kernel_size"], 
            device=config["device"],
            id2label=id2label
        )

        fold_time = time.time() - fold_start_time
        print(f"Time taken for fold {fold + 1}: {fold_time:.2f} seconds")

        print(f"\nPerforming inference with CNN_fold_{fold} model...")

        inference_start_time = time.time()
        preds = inference(models[f'CNN_fold_{fold}'], prepare_dataloader(test_data, test_labels, batch_size=config["batch_size"], shuffle=False), device=config["device"])
        inference_time = time.time() - inference_start_time
        avg_inference_time = inference_time / len(test_labels)

        accuracy = np.mean(preds == test_labels)
        print(f"Accuracy of CNN_fold_{fold} model: {accuracy:.4f}")
        print(f"Inference time for fold {fold + 1}: {inference_time:.2f} seconds")
        print(f"Average time per guess: {avg_inference_time:.4f} seconds")
        wandb.log({f"inference_accuracy": accuracy, f"inference_time": inference_time, f"avg_time_per_guess": avg_inference_time})

        wandb.finish()

    total_time = time.time() - total_start_time
    avg_epoch_time = total_time / (config["num_epochs"] * config["n_kfolds"])

    print(f"\nTotal time taken for training: {total_time:.2f} seconds")
    print(f"Average time per epoch: {avg_epoch_time:.2f} seconds")
    print("\nFinished training CNN model with K-fold cross-validation.")

def main():
    """
    Main function to set up the environment, parse arguments, and execute 
    the training and evaluation of the CNN model using K-Fold cross-validation.
    """
    args = parse_arguments()

    config = {
        "data_path": args.data_path,
        "intent_labels": args.intent_labels,
        "detect_labels": args.detect_labels,
        "num_epochs": args.num_epochs,
        "augment": args.augment.lower() in ['true', '1', 't', 'y', 'yes'],
        "batch_size": args.batch_size,
        "n_kfolds": args.n_kfolds,
        "kernel_size": args.kernel_size,
        "project_name": args.project_name,
        "run_name": f"{args.run_name}_augment" if args.augment.lower() in ['true', '1', 't', 'y', 'yes'] else args.run_name,
        "device": torch.device(args.device if torch.cuda.is_available() else "cpu")
    }

    train_data, train_labels, test_data, test_labels, id2label = load_and_split_data(
        config["data_path"], config["intent_labels"], config["detect_labels"], config["augment"]
    )

    train_and_evaluate(
        train_data, train_labels, test_data, test_labels, id2label, config
    )

# ------------------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()
