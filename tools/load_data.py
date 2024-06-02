# ------------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------------- #

# Python Imports
import os
import yaml

# Package Imports
import numpy as np
import torch

# File Imports
from tools.normalization import normalize, mean_removed_all
from tools.augmentations import flip_trajectories_x

# ------------------------------------------------------------------------------------- #
# Functions
# ------------------------------------------------------------------------------------- #

def load_flight_data_single(data_path: str, 
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
        tuple: A tuple containing:
            - numpy.ndarray: Training trajectories represented as 3-dimensional coordinates.
            - list: Binary labels corresponding to the training trajectories.
            - dict: Mapping of trajectory IDs to their labels.
            - dict: Mapping of labels to their corresponding IDs.
            - dict: Mapping of detailed labels to their corresponding IDs.
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

    # Copy all the drone classes
    drone_classes = [trajectory[0].copy() for trajectory in train_trajectories]
    
    # Remove the drone classes from all trajectories
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
        drone_classes += drone_classes
    
    # Add the drone classes back to the processed trajectories
    processed_trajectories = []
    for first_point, mean_removed_trajectory in zip(drone_classes, mean_removed_trajectories):
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

    return train_trajectories, train_labels, drone_classes, id2label, label2id, id2label_detailed

def load_flight_data(data_path: str, labels_path: str, label_detailed_path: str, augment: bool = False) -> tuple:
    """
    Load flight trajectory data and associated labels for a binary classification task.

    Args:
        data_path (str): Path to the directory containing trajectory data (.pt files) or to a single trajectory data file.
        labels_path (str): Path to the file containing labels corresponding to trajectory data (.yaml).
        label_detailed_path (str): Path to the file containing detailed labels corresponding to trajectory data (.yaml).
        augment (bool, optional): Whether to augment the data by flipping trajectories. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Training trajectories represented as 3-dimensional coordinates.
            - list: Labels corresponding to the training trajectories.
            - list: Task names corresponding to the training trajectories.
            - dict: Mapping of trajectory IDs to their labels.
            - dict: Mapping of labels to their corresponding IDs.
            - dict: Mapping of detailed labels to their corresponding IDs.
            
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
    task_names = []
    for file_name in trajectory_files:
        trajectory = torch.load(file_name).numpy()
        # Concatenate sequences within each trajectory
        train_trajectories.extend(trajectory)
        
    # Convert the list of trajectories to a numpy array
    train_trajectories = np.array(train_trajectories)

    # Extract the drone classes as the task names (drone classes)
    task_names = [int(trajectory[0][0]) for trajectory in train_trajectories]
    
    # Remove the drone classes (drone classes) from all trajectories
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
        task_names += task_names
    
    # Add the drone classes (drone classes) back to the processed trajectories
    processed_trajectories = []
    for mean_removed_trajectory in mean_removed_trajectories:
        processed_trajectories.append(mean_removed_trajectory)
    
    train_trajectories = np.array(processed_trajectories)

    # Load a map of the expected ids to their labels with `id2label` and `label2id`
    with open(labels_path, "r") as stream:
        id2label = yaml.safe_load(stream)
    label2id = {v: k for k, v in id2label.items()}
    
    with open(label_detailed_path, "r") as stream:
        id2label_detailed = yaml.safe_load(stream)
    label_detailed2id = {v: k for k, v in id2label_detailed.items()}

    return train_trajectories, train_labels, task_names, id2label, label2id, id2label_detailed
