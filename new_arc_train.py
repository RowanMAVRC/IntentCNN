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
import random
import time

# Package Imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm

# File Imports
from data_tools.chart_generator import generate_histogram_and_pie_chart, generate_histogram_and_pie_chart_for_split
from data_tools.normalization import mean_removed_all, mean_removed_single, compute_trajectory_stats, normalize
from data_tools.augmentations import flip_trajectories_x, augment_with_jitters

# ------------------------------------------------------------------------------------- #
# Functions & Definitions
# ------------------------------------------------------------------------------------- #

# Define the CNN model architecture
class CNNModel(nn.Module):
    """
    CNN Model for flight trajectory classification.

    Args:
        input_dim (int): Number of input features (dimensions of the trajectory).
        output_dim (int): Number of output classes (intentions).

    Methods:
        forward(x): Forward pass of the model.
    """
    def __init__(self, input_dim, output_dim):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        x = x.permute(0, 2, 1)  # Change to (batch_size, input_dim, sequence_length)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max(x, dim=2)[0]  # Global max pooling
        x = self.dropout(x)
        x = self.fc(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, fold, model_name):
    """
    Trains the model and logs training progress to Wandb.

    Args:
        model (nn.Module): The CNN model to train.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.
        optimizer (Optimizer): Optimizer for updating model parameters.
        num_epochs (int): Number of training epochs.
        fold (int): Current fold number for cross-validation.
        model_name (str): Name of the model.

    Returns:
        nn.Module: Trained model.
    """
    with tqdm(total=num_epochs, desc=f"Training Fold {fold}") as pbar:
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode
            epoch_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)
            pbar.set_postfix({
                'Epoch': f'{epoch+1}/{num_epochs}',
                'Train Loss': f'{epoch_loss/len(train_loader):.4f}',
                'Val Loss': f'{val_loss:.4f}',
                'Val Acc': f'{val_accuracy:.4f}'
            })
            pbar.update(1)
            wandb.log({
                f"train_loss": epoch_loss/len(train_loader),
                f"val_loss": val_loss,
                f"val_accuracy": val_accuracy
            })
    return model

# Define the function to evaluate the model
def evaluate_model(model, data_loader, criterion):
    """
    Evaluates the model on the validation data.

    Args:
        model (nn.Module): The trained CNN model.
        data_loader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.

    Returns:
        tuple: (Average validation loss, Validation accuracy)
    """
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return total_loss / len(data_loader), accuracy

# Define the function to load and preprocess flight data
def load_flight_data(data_path: str, 
                     labels_path: str,
                     label_detailed_path: str,
                     augment: bool = False) -> tuple:
    """
    Load flight trajectory data and associated labels for a binary classification task.

    Args:
        data_path (str): Path to the directory containing trajectory data (.pt files) or to a single trajectory data file.
        labels_path (str): Path to the file containing labels corresponding to trajectory data (.yaml).
        augment (bool, optional): Whether to augment the data by flipping trajectories. Defaults to False.

    Returns:
        tuple: A tuple  containing:
            - numpy.ndarray: Training trajectories represented as 3-dimensional coordinates.
            - list: Binary labels corresponding to the training trajectories.
            - dict: Mapping of trajectory IDs to their labels.
            - dict: Mapping of labels to their corresponding IDs.
        (train_trajectories, train_labels, id2label, label2id, id2label_detailed)
            
    Raises:
        FileNotFoundError: If either data_path or labels_path does not exist.
    """
    # Check if data directory exists
    if not os.path.exists(data_path) or data_path == "" or data_path is None:
        raise FileNotFoundError(f"Data directory '{data_path}' does not exist.")
    
    # Check if labels file exists
    if not os.path.exists(labels_path) or labels_path == "" or labels_path is None:
        raise FileNotFoundError(f"Labels file '{labels_path}' does not exist.")
    
    # Check if detailed labels file exists
    if not os.path.exists(label_detailed_path) or label_detailed_path == "" or label_detailed_path is None:
        raise FileNotFoundError(f"Labels file '{label_detailed_path}' does not exist.")
    
    # Load training trajectories
    if os.path.isfile(data_path):
        # If data_path is a file, load just that file
        trajectory_files = [data_path]
    else:
        # If data_path is a directory, load all .pt files in that directory
        trajectory_files = [os.path.join(data_path, file_name) for file_name in os.listdir(data_path) if file_name.endswith('.pt')]

    # Load training trajectories from .pt files in the directory
    train_trajectories = []
    for file_name in trajectory_files:
        trajectory = torch.load(file_name).numpy()
        # Concatenate sequences within each trajectory
        train_trajectories.extend(trajectory)
        
    # Convert the list of trajectories to a numpy array
    train_trajectories = np.array(train_trajectories)

    # Copy all the first points
    first_points = [trajectory[0].copy() for trajectory in train_trajectories]
    
    # Remove the first points from all trajectories
    remaining_trajectories = [trajectory[1:] for trajectory in train_trajectories]
    remaining_trajectories = np.array(remaining_trajectories)
    
    # Extract labels from trajectories
    train_labels = [int(trajectory[0][0]) for trajectory in remaining_trajectories]
    remaining_trajectories = np.delete(remaining_trajectories, 0, axis=2)

    # Normalize and mean remove the remaining trajectories
    normalized_trajectories = normalize(remaining_trajectories)
    mean_removed_trajectories = mean_removed_all(normalized_trajectories)
    
    # Data augmentation
    if augment:
        print("Augment")
        # Flip trajectories along the x-axis
        flipped_trajectories = flip_trajectories_x(mean_removed_trajectories)
        # Concatenate original and flipped trajectories along the first axis (num_trajectories)
        mean_removed_trajectories = np.concatenate((mean_removed_trajectories, flipped_trajectories), axis=0)
        train_labels += train_labels
        first_points += first_points
    
    # Add the first points back to the processed trajectories
    processed_trajectories = []
    for first_point, mean_removed_trajectory in zip(first_points, mean_removed_trajectories):
        processed_trajectory = np.vstack((first_point[1:], mean_removed_trajectory))
        processed_trajectories.append(processed_trajectory)
    
    train_trajectories = np.array(processed_trajectories)

    # Load a map of the expected ids to their labels with `id2label` and `label2id`
    with open(labels_path, "r") as stream:
        id2label = yaml.safe_load(stream)
    label2id = {v: k for k, v in id2label.items()}
    
    with open(label_detailed_path, "r") as stream:
        id2label_detailed = yaml.safe_load(stream)
    label_detailed2id = {v: k for k, v in id2label_detailed.items()}

    return train_trajectories, train_labels, id2label, label2id, id2label_detailed

# Define the function to prepare the data loader
def prepare_dataloader(trajectories, labels, batch_size=32, shuffle=True):
    """
    Prepares the data loader for training and validation datasets.

    Args:
        trajectories (numpy.ndarray): Array of trajectories.
        labels (list): List of labels corresponding to the trajectories.
        batch_size (int, optional): Batch size for the data loader. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

    Returns:
        DataLoader: DataLoader for the dataset.
    """
    class FlightDataset(torch.utils.data.Dataset):
        def __init__(self, trajectories, labels):
            self.trajectories = torch.tensor(trajectories, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.trajectories)

        def __getitem__(self, idx):
            return self.trajectories[idx], self.labels[idx]

    dataset = FlightDataset(trajectories, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# Define the function to train the CNN model
def train_cnn(train_trajectories, train_labels, val_trajectories, val_labels, fold, model_name, num_epochs=10, batch_size=32):
    """
    Trains the CNN model with the provided data and cross-validation fold.

    Args:
        train_trajectories (numpy.ndarray): Training trajectories.
        train_labels (list): Labels for the training trajectories.
        val_trajectories (numpy.ndarray): Validation trajectories.
        val_labels (list): Labels for the validation trajectories.
        fold (int): Current fold number for cross-validation.
        model_name (str): Name of the model.

    Returns:
        nn.Module: Trained CNN model.
    """
    print(f"Training CNN model for fold {fold}...")
    
    # Prepare data loaders for training and validation sets
    train_loader = prepare_dataloader(train_trajectories, train_labels, batch_size=batch_size, shuffle=True)
    val_loader = prepare_dataloader(val_trajectories, val_labels, batch_size=batch_size, shuffle=True)
    
    # Define input and output dimensions
    input_dim = train_trajectories.shape[2]
    output_dim = len(np.unique(train_labels))
    
    # Compute class weights to handle class imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).cuda()

    # Initialize the model, criterion, and optimizer
    model = CNNModel(input_dim, output_dim).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, fold=fold, model_name=model_name)
    
    # Save the trained model
    model_save_path = f"trained_models/{model_name}_fold_{fold}.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    print(f"Finished training CNN model for fold {fold}.")
    return model

# Define the function for inference
def inference(model, data_loader):
    """
    Performs inference on the validation data.

    Args:
        model (nn.Module): Trained CNN model.
        data_loader (DataLoader): DataLoader for the validation data.

    Returns:
        list: Predictions for the validation data.
    """
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.cuda()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
    return all_preds

def load_model(model_class, model_path, input_dim, output_dim):
    """
    Loads a model from the specified path.

    Args:
        model_class (type): The class of the model to be loaded.
        model_path (str): Path to the saved model state dictionary.
        input_dim (int): Number of input features (dimensions of the trajectory).
        output_dim (int): Number of output classes (intentions).

    Returns:
        nn.Module: Loaded model.
    """
    model = model_class(input_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def plot_and_log_statistics(train_labels: list, id2label: dict) -> tuple:
    """
    Plots the histogram and pie chart of the dataset.

    Args:
        train_labels (list): List of labels.
        id2label (dict): Dictionary mapping label ids to label names.

    Returns:
        tuple: (hist_img, pie_img) where hist_img and pie_img are numpy arrays of the figures.
    """
    # Count the number of trajectories for each intention
    unique, counts = np.unique(train_labels, return_counts=True)
    label_counts = dict(zip(unique, counts))
    
    # Plot histogram
    fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
    ax_hist.bar(range(len(label_counts)), list(label_counts.values()), align='center')
    ax_hist.set_xticks(range(len(label_counts)))
    ax_hist.set_xticklabels([id2label[key] for key in label_counts.keys()])
    ax_hist.set_xlabel('Intention')
    ax_hist.set_ylabel('Number of Trajectories')
    ax_hist.set_title('Number of Trajectories per Intention')
    
    fig_hist.canvas.draw()
    hist_img = np.frombuffer(fig_hist.canvas.tostring_rgb(), dtype=np.uint8)
    hist_img = hist_img.reshape(fig_hist.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig_hist)
    
    # Plot pie chart
    fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
    ax_pie.pie(list(label_counts.values()), labels=[id2label[key] for key in label_counts.keys()], autopct='%1.1f%%')
    ax_pie.set_title('Distribution of Trajectories per Intention')
    
    fig_pie.canvas.draw()
    pie_img = np.frombuffer(fig_pie.canvas.tostring_rgb(), dtype=np.uint8)
    pie_img = pie_img.reshape(fig_pie.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig_pie)
    
    return hist_img, pie_img

def generate_histogram_and_pie_chart(labels, label_mapping, file_name):
    """
    Generate a histogram and a pie chart for label distribution.

    Args:
    labels (list): List of labels.
    label_mapping (dict): Mapping of labels to numeric values.
    file_name (str): Name of the file to save the charts.

    Returns:
    None
    """
    # Map labels to numeric values and calculate counts
    mapped_labels = np.sort(np.array([label_mapping[label] for label in labels]))
    unique, counts = np.unique(mapped_labels, return_counts=True)
    
    # Plot Histogram
    fig = plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    counts, edges, bars = plt.hist(mapped_labels, bins=np.arange(len(unique)+1) - 0.5, edgecolor='black')
    plt.bar_label(bars)
    plt.title('Label Distribution')
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha="right")
    plt.subplots_adjust(wspace=0.5)  # Adjust spacing between subplots
    plt.suptitle(file_name, fontsize=16, y=0.95)  # Add file name as title
    
    # Plot Pie Chart
    plt.subplot(1, 2, 2)
    plt.pie(counts, labels=unique, autopct='%1.1f%%', startangle=140)
    plt.title('Label Proportion')
    plt.subplots_adjust(wspace=0.5)  # Adjust spacing between subplots
    plt.text(0.5, 0.05, f"Total Labels: {len(labels)}", fontsize=12, transform=plt.gca().transAxes)
    
    fig.canvas.draw()
    fig_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    fig_img = fig_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return fig_img

def generate_histogram_and_pie_chart_for_split(train_labels, val_labels, label_mapping, file_name):
    """
    Generate a histogram and a pie chart for label distribution in train and validation sets.

    Args:
    train_labels (list): List of labels in the training set.
    val_labels (list): List of labels in the validation set.
    label_mapping (dict): Mapping of labels to numeric values.
    file_name (str): Name of the file to save the charts.

    Returns:
    None
    """
    # Map labels to numeric values and calculate counts for train and validation sets
    train_mapped_labels = np.sort(np.array([label_mapping[label] for label in train_labels]))
    val_mapped_labels = np.sort(np.array([label_mapping[label] for label in val_labels]))
    
    unique_train, counts_train = np.unique(train_mapped_labels, return_counts=True)
    unique_val, counts_val = np.unique(val_mapped_labels, return_counts=True)
    
    # Plot Histogram
    fig = plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    counts, edges, bars = plt.hist(train_mapped_labels, bins=np.arange(len(unique_train)+1) - 0.5, edgecolor='black', alpha=0.5, label='Train')
    plt.bar_label(bars)
    counts, edges, bars = plt.hist(val_mapped_labels, bins=np.arange(len(unique_val)+1) - 0.5, edgecolor='black', alpha=0.5, label='Validation')
    plt.bar_label(bars)
    plt.title('Label Distribution')
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.subplots_adjust(wspace=0.5)  # Adjust spacing between subplots
    plt.suptitle(file_name, fontsize=16, y=0.95)  # Add file name as title
    
    # Plot Pie Chart
    plt.subplot(1, 2, 2)
    plt.pie(counts_train, labels=unique_train, autopct='%1.1f%%', startangle=140, radius=0.7)
    plt.pie(counts_val, labels=unique_val, autopct='%1.1f%%', startangle=140, radius=1)
    plt.title('Label Proportion')
    plt.subplots_adjust(wspace=0.5)  # Adjust spacing between subplots
    plt.text(0.5, 0.01, f"Total Labels: Train - {len(train_labels)}, Validation - {len(val_labels)}", fontsize=12, transform=plt.gca().transAxes)
    
    fig.canvas.draw()
    fig_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    fig_img = fig_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return fig_img

# ------------------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    
    debug = True
    
    if debug:
        # Debug configuration
        # data_path = "/data/TGSSE/UpdatedIntentions/XYZ/800pad_66"
        # labels_path = "/data/TGSSE/UpdatedIntentions/labels.yaml"
        # label_detailed_path = "/data/TGSSE/UpdatedIntentions/XYZ/800pad_66/trajectory_with_intentions_800_pad_533_min_151221_label_detailed.yaml"
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
        # Define command-line arguments
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
    
        # Extract command-line arguments
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
        
        # Print configuration
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")       
    
    # Initialize Wandb
    wandb.init(project=project_name, name=f"Overall_Labels_{run_name}")
    
    # Load the data
    flight_data, flight_labels, id2label, label2id, id2label_detailed = load_flight_data(data_path, labels_path, label_detailed_path, augment)
    print(f"Total trajectories: {len(flight_data)}")
    
    # Reserve 10% random trajectories for the test set
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
    
    # Plot and log the initial statistics
    stat_img = generate_histogram_and_pie_chart(flight_labels, id2label, data_path)
    wandb.log({"Overall Stats": [wandb.Image(stat_img)]})
    
    # Finish the initial Wandb run
    wandb.finish()
    
    # Initialize KFold for cross-validation
    kf = KFold(n_splits=n_kfolds, shuffle=True, random_state=42)
    
    models = {}
    
    print("Starting training of CNN model with K-fold cross-validation...\n")
    
    total_start_time = time.time()  # Start the timer for the entire training
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(flight_data)):
        print(f"\nFold {fold+1}/{kf.n_splits}\n")
        
        fold_start_time = time.time()  # Start the timer for each fold
        
        # Initialize a new Wandb run for each fold
        wandb.init(project=project_name, name=f"{run_name}_fold{fold+1}")
        
        # Split data into training and validation sets
        train_trajectories, val_trajectories = flight_data[train_idx], flight_data[val_idx]
        train_labels, val_labels = np.array(flight_labels)[train_idx], np.array(flight_labels)[val_idx]
        print(f"Training set size: {len(train_trajectories)}")
        print(f"Validation set size: {len(val_trajectories)}")
        
        # Generate charts for the split
        split_stat = generate_histogram_and_pie_chart_for_split(train_labels, val_labels, id2label, f'{run_name}_fold{fold+1}')
        wandb.log({"Split Distribution": wandb.Image(split_stat)})
        
        # Train the CNN model
        models[f'CNN_fold_{fold}'] = train_cnn(train_trajectories, train_labels, val_trajectories, val_labels, fold, 'CNN', num_epochs=num_epochs)
        
        fold_end_time = time.time()  # End the timer for each fold
        fold_time = fold_end_time - fold_start_time  # Calculate the time taken for each fold
        print(f"Time taken for fold {fold+1}: {fold_time:.2f} seconds")
        
        # Perform inference and print results for each fold using the test set
        print(f"\nPerforming inference with CNN_fold_{fold} model...")
        
        inference_start_time = time.time()  # Start the timer for inference
        preds = inference(models[f'CNN_fold_{fold}'], prepare_dataloader(test_data, test_labels, batch_size=batch_size, shuffle=False))
        inference_end_time = time.time()  # End the timer for inference
        inference_time = inference_end_time - inference_start_time  # Calculate the time taken for inference
        avg_inference_time = inference_time / len(test_labels)  # Calculate the average time for one guess

        print(f"Predictions: {preds[:10]}")  # Print first 10 predictions for brevity
        print(f"True Labels: {test_labels[:10].tolist()}")  # Print first 10 true labels for brevity
        accuracy = np.mean(preds == test_labels)
        print(f"Accuracy of CNN_fold_{fold} model: {accuracy:.4f}")
        print(f"Inference time for fold {fold+1}: {inference_time:.2f} seconds")
        print(f"Average time per guess: {avg_inference_time:.4f} seconds")

        wandb.log({f"inference_accuracy": accuracy,
                   f"inference_time": inference_time,
                   f"avg_time_per_guess": avg_inference_time})

        # Finish the Wandb run for the fold
        wandb.finish()
    
    total_end_time = time.time()  # End the timer for the entire training
    total_time = total_end_time - total_start_time  # Calculate the total time taken
    avg_epoch_time = total_time / (num_epochs * n_kfolds)  # Calculate the average time per epoch
    
    print(f"\nTotal time taken for training: {total_time:.2f} seconds")
    print(f"Average time per epoch: {avg_epoch_time:.2f} seconds")
    print("\nFinished training CNN model with K-fold cross-validation.")