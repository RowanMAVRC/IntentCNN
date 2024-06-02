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
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight

# File Imports
from tools.load_data import load_flight_data
from cfgs.models.multiTaskCNN import inference, prepare_dataloader, MultitaskCNNModel, train_model
from tools.utils import generate_histogram_and_pie_chart, generate_histogram_and_pie_chart_for_split

# ------------------------------------------------------------------------------------- #
# Functions
# ------------------------------------------------------------------------------------- #

def main():
    """
    Main function to set up the environment, parse arguments or use debug configuration,
    and execute the training and evaluation of the CNN model using K-Fold cross-validation.
    """
    debug = True
    
    if debug:
        # Debug configuration
        data_path = "/data/TGSSE/UpdatedIntentions/XYZ/400pad_66"
        intent_labels = "/data/TGSSE/UpdatedIntentions/intent_labels.yaml"
        detect_labels = "/data/TGSSE/UpdatedIntentions/detect_labels.yaml"
        num_epochs = 100
        augment = True
        batch_size = 128
        n_kfolds = 5
        kernel_size = 8
        project_name = "cnn_trajectory_classification"
        run_name = "400pad_66"
        
    else:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description='Flight Trajectory Classification')
        parser.add_argument('--data_path', type=str, help='Path to trajectory data: Directory containing .pt files')
        parser.add_argument('--intent_labels', type=str, help='Path to labels data: .yaml file')
        parser.add_argument('--detect_labels', type=str, help='Path to detailed labels data: .yaml file')
        parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
        parser.add_argument('--augment', type=str, default='False', help='Whether to augment the data')
        parser.add_argument('--n_kfolds', type=int, default=5, help='Number of splits for KFold cross-validation')
        parser.add_argument('--project_name', type=str, default='cnn_trajectory_classification', help='Project name for wandb')
        parser.add_argument('--run_name', type=str, default='mobilebert_run', help='Run name for wandb')
        parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
        parser.add_argument('--kernel_size', type=int, default=8, help='Kernel size for CNN model')
        args = parser.parse_args()
    
        # Definitions
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_path = args.data_path
        intent_labels = args.intent_labels
        detect_labels = args.detect_labels
        num_epochs = args.num_epochs
        augment = args.augment.lower() in ['true', '1', 't', 'y', 'yes']
        n_kfolds = args.n_kfolds
        project_name = args.project_name
        run_name = f"{args.run_name}_augment" if augment else args.run_name
        batch_size = args.batch_size
        kernel_size = args.kernel_size
        
        # Print configuration
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")   
            
    # Define a config dictionary object for WandB
    config = {
        "data_path": data_path,
        "intent_labels": intent_labels,
        "detect_labels": detect_labels,
        "num_epochs": num_epochs,
        "augment": str(augment),
        "batch_size": batch_size,
        "n_kfolds": n_kfolds,
        "kernel_size": kernel_size,
        "project_name": project_name,
        "run_name": run_name
    }    
    
    wandb.init(project=project_name, name=f"Overall_Labels_{run_name}", config=config)
    
    # Load dataset
    flight_data, flight_labels, task_names, id2label, label2id, id2label_detailed = load_flight_data(data_path, intent_labels, detect_labels, augment)
    print(f"Total trajectories: {len(flight_data)}")
    
    # Split data into training and test sets
    test_size = flight_data.shape[0] // 10
    print(f"Test set size: {test_size}")
    indices = np.arange(len(flight_data))
    np.random.shuffle(indices)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    test_data = flight_data[test_indices]
    test_labels = np.array(flight_labels)[test_indices]
    test_task_names = np.array(task_names)[test_indices]
    flight_data = flight_data[train_indices]
    flight_labels = np.array(flight_labels)[train_indices]
    task_names = np.array(task_names)[train_indices]
    
    # Generate charts for the entire flight data
    stat_img = generate_histogram_and_pie_chart(flight_labels, id2label, data_path)
    wandb.log({"Overall Stats": [wandb.Image(stat_img)]})
    
    # End the current wandb run
    wandb.finish()
    
    # Initialize KFold
    kf = KFold(n_splits=n_kfolds, shuffle=True, random_state=42)
    
    models = {}
    
    print("Starting training of CNN model with K-fold cross-validation...\n")
    
    total_start_time = time.time()
    
    # Task output dimensions as specified
    task_output_dims = {
        "0": 2,  # BAYRAKTART2: Travel, Kamikaze
        "1": 3,  # HEX1: Travel, Recon, Area Denial
        "2": 3,  # QUAD: Travel, Recon, Area Denial
        "3": 2,  # SWITCHBLADE: Travel, Kamikaze
    }
    
    # Iterate over folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(flight_data)):
        print(f"\nFold {fold+1}/{kf.n_splits}\n")
        
        fold_start_time = time.time()
        
        # Initialize a new WandB run for each fold
        wandb.init(project=project_name, name=f"{run_name}_fold{fold+1}", config=config)
        
        # Split data into train and validation sets
        train_trajectories, val_trajectories = flight_data[train_idx], flight_data[val_idx]
        train_labels, val_labels = np.array(flight_labels)[train_idx], np.array(flight_labels)[val_idx]
        train_task_names, val_task_names = task_names[train_idx], task_names[val_idx]
        print(f"Training set size: {len(train_trajectories)}")
        print(f"Validation set size: {len(val_trajectories)}")
        
        # Generate charts for the split
        split_stat = generate_histogram_and_pie_chart_for_split(train_labels, val_labels, id2label, f'{run_name}_fold{fold+1}')
        wandb.log({"Split Distribution": wandb.Image(split_stat)})
        
        # Define input dimension
        input_dim = train_trajectories.shape[2]
        
        # Initialize the multitask CNN model
        model = MultitaskCNNModel(input_dim, task_output_dims, kernel_size).cuda()
        
        # Train the model for the current fold
        models[f'CNN_fold_{fold}'] = {}
        for task_name in np.unique(train_task_names):
            task_train_idx = np.where(train_task_names == task_name)[0]
            task_val_idx = np.where(val_task_names == task_name)[0]
            task_train_trajectories = train_trajectories[task_train_idx]
            task_train_labels = train_labels[task_train_idx]
            task_val_trajectories = val_trajectories[task_val_idx]
            task_val_labels = val_labels[task_val_idx]
            task_train_loader = prepare_dataloader(task_train_trajectories, task_train_labels, batch_size=batch_size, shuffle=True)
            task_val_loader = prepare_dataloader(task_val_trajectories, task_val_labels, batch_size=batch_size, shuffle=False)
            
            # Ensure task labels are within range
            # assert all(task_train_labels >= 0) and all(task_train_labels < task_output_dims[str(task_name)])
            # assert all(task_val_labels >= 0) and all(task_val_labels < task_output_dims[str(task_name)])
            
            # Compute class weights to handle class imbalance
            unique_classes = np.unique(task_train_labels)
            if len(unique_classes) > 1:
                class_weights = compute_class_weight('balanced', classes=unique_classes, y=task_train_labels)
                class_weights = torch.tensor(class_weights, dtype=torch.float).cuda()
                criterion = nn.CrossEntropyLoss(weight=class_weights)
            else:
                criterion = nn.CrossEntropyLoss()
            
            # Define the optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Train the model for the current task
            models[f'CNN_fold_{fold}'][task_name] = train_model(model, task_train_loader, task_val_loader, criterion, optimizer, num_epochs, fold, str(task_name))
        
        fold_end_time = time.time()
        fold_time = fold_end_time - fold_start_time
        print(f"Time taken for fold {fold+1}: {fold_time:.2f} seconds")
        
        print(f"\nPerforming inference with CNN_fold_{fold} model...")
        
        # Perform inference on the test set using the trained model for each task
        inference_start_time = time.time()
        for task_name in np.unique(test_task_names):
            task_test_idx = np.where(test_task_names == task_name)[0]
            task_test_data = test_data[task_test_idx]
            task_test_labels = test_labels[task_test_idx]
            task_test_loader = prepare_dataloader(task_test_data, task_test_labels, batch_size=batch_size, shuffle=False)
            preds = inference(models[f'CNN_fold_{fold}'][task_name], task_test_loader, str(task_name))
            
            # Print and log inference results
            print(f"Predictions for task {task_name}: {preds[:10]}")
            print(f"True Labels for task {task_name}: {task_test_labels[:10].tolist()}")
            accuracy = np.mean(preds == task_test_labels)
            print(f"Accuracy of CNN_fold_{fold} model for task {task_name}: {accuracy:.4f}")
            wandb.log({f"inference_accuracy_task_{task_name}": accuracy})
        
        inference_end_time = time.time()
        inference_time = inference_end_time - inference_start_time
        avg_inference_time = inference_time / len(test_labels)
        
        print(f"Inference time for fold {fold+1}: {inference_time:.2f} seconds")
        print(f"Average time per guess: {avg_inference_time:.4f} seconds")
        wandb.log({f"inference_time_fold_{fold}": inference_time, f"avg_time_per_guess_fold_{fold}": avg_inference_time})
        
        # End the current wandb run
        wandb.finish()
    
    # Calculate & Print total training time and average time per epoch
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