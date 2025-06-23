"""
 _____       _             _                           
|_   _|     | |           | |    ____  _   _  _   _                           
  | |  _ __ | |_ ___ _ __ | |_  / ___|| \ | || \ | |
  | | | '_ \| __/ _ \ '_ \| __|| |   ||  \| ||  \| |
 _| |_| | | | ||  __/ | | | |_ | |___|| |\  || |\  |
|_____|_| |_|\__\___|_| |_|\__| \____||_| \_||_| \_|
"""
# ------------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------------- #

import os
import yaml
import numpy as np
import torch
from tools.normalization import normalize_single, mean_removed_single, normalize_all, mean_removed_all
from tools.augmentations import flip_trajectories_x

# ------------------------------------------------------------------------------------- #
# Functions
# ------------------------------------------------------------------------------------- #

def validate_paths(*paths):
    """
    Validate if the given paths exist.

    Args:
        *paths (str): Paths to validate.

    Raises:
        FileNotFoundError: If any of the paths do not exist.
    """
    for path in paths:
        if not os.path.exists(path) or not path:
            raise FileNotFoundError(f"Path '{path}' does not exist.")


def load_trajectories(data_path):
    """
    Load trajectory data from .pt files.

    Args:
        data_path (str): Path to the directory containing trajectory data (.pt files) or to a single trajectory data file.

    Returns:
        numpy.ndarray: Loaded trajectories.
    """
    if os.path.isfile(data_path):
        trajectory_files = [data_path]
    else:
        trajectory_files = [os.path.join(data_path, file_name) for file_name in os.listdir(data_path) if file_name.endswith('.pt')]

    trajectories = []
    for file_name in trajectory_files:
        trajectory = torch.load(file_name).numpy()
        trajectories.extend(trajectory)

    return np.array(trajectories)


def extract_and_preprocess_trajectories(trajectories, augment, normalization_func, mean_removal_func, **kwargs):
    """
    Extract and preprocess trajectory data.

    Args:
        trajectories (numpy.ndarray): Loaded trajectory data.
        augment (bool): Whether to augment the data by flipping trajectories.
        normalization_func (function): Function to normalize the data.
        mean_removal_func (function): Function to remove mean from the data.
        **kwargs: Additional arguments for normalization.

    Returns:
        tuple: Preprocessed trajectories and corresponding labels.
    """
    
    drone_classes = [int(trajectory[0][0].copy()) for trajectory in trajectories]
    remaining_trajectories = np.array([trajectory[1:] for trajectory in trajectories])

    train_labels = [int(trajectory[0][0]) for trajectory in remaining_trajectories]
    remaining_trajectories = np.delete(remaining_trajectories, 0, axis=2)

    normalized_trajectories = normalization_func(remaining_trajectories, **kwargs)
    mean_removed_trajectories = mean_removal_func(normalized_trajectories)

    if augment:
        flipped_trajectories = flip_trajectories_x(mean_removed_trajectories)
        mean_removed_trajectories = np.concatenate((mean_removed_trajectories, flipped_trajectories), axis=0)
        train_labels += train_labels
        drone_classes += drone_classes

    print(f"Extracted labels: {set(train_labels)}")

    return np.array(mean_removed_trajectories), train_labels, drone_classes


def load_labels(label_path, label_detailed_path):
    """
    Load label data from yaml files.

    Args:
        label_path (str): Path to the file containing labels.
        label_detailed_path (str): Path to the file containing detailed labels.

    Returns:
        tuple: Mappings of labels to ids and detailed labels to ids.
    """

    with open(label_path, "r") as stream:
        id2label = yaml.safe_load(stream)
    label2id = {v: k for k, v in id2label.items()}

    with open(label_detailed_path, "r") as stream:
        id2label_detailed = yaml.safe_load(stream)
    label_detailed2id = {v: k for k, v in id2label_detailed.items()}

    return id2label, label2id, id2label_detailed, label_detailed2id


def load_flight_data(data_path, label_path, label_detailed_path, augment=False, normalization_func=normalize_single, mean_removal_func=mean_removed_single, **kwargs):
    """
    Load flight trajectory data and associated labels for a binary classification task.

    Args:
        data_path (str): Path to the directory containing trajectory data (.pt files) or to a single trajectory data file.
        label_path (str): Path to the file containing labels corresponding to trajectory data (.yaml).
        label_detailed_path (str): Path to the file containing detailed labels corresponding to trajectory data (.yaml).
        augment (bool, optional): Whether to augment the data by flipping trajectories. Defaults to False.
        normalization_func (function, optional): Function to normalize the data. Defaults to normalize_single.
        mean_removal_func (function, optional): Function to remove mean from the data. Defaults to mean_removed_single.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Training trajectories represented as 3-dimensional coordinates.
            - list: Binary labels corresponding to the training trajectories.
            - list: Drone classes.
            - dict: Mapping of trajectory IDs to their labels.
            - dict: Mapping of labels to their corresponding IDs.
            - dict: Mapping of detailed labels to their corresponding IDs.
    """
    validate_paths(data_path, label_path, label_detailed_path)
    trajectories = load_trajectories(data_path)
    train_trajectories, train_labels, drone_classes = extract_and_preprocess_trajectories(trajectories, augment, normalization_func, mean_removal_func, **kwargs)
    id2label, label2id, id2label_detailed, label_detailed2id = load_labels(label_path, label_detailed_path)

    return train_trajectories, train_labels, drone_classes, id2label, label2id, id2label_detailed, label_detailed2id


def load_flight_data_single(data_path, label_path, label_detailed_path, augment=False, **kwargs):
    """
    Wrapper for load_flight_data with single trajectory normalization and mean removal functions.

    Args:
        data_path (str): Path to the directory containing trajectory data (.pt files) or to a single trajectory data file.
        label_path (str): Path to the file containing labels corresponding to trajectory data (.yaml).
        label_detailed_path (str): Path to the file containing detailed labels corresponding to trajectory data (.yaml).
        augment (bool, optional): Whether to augment the data by flipping trajectories. Defaults to False.

    Returns:
        tuple: Preprocessed trajectories and associated label information.
    """
    return load_flight_data(data_path, label_path, label_detailed_path, augment, normalize_single, mean_removed_single, **kwargs)


def load_flight_data_all(data_path, label_path, label_detailed_path, augment=False):
    """
    Wrapper for load_flight_data with all trajectory normalization and mean removal functions.

    Args:
        data_path (str): Path to the directory containing trajectory data (.pt files) or to a single trajectory data file.
        label_path (str): Path to the file containing labels corresponding to trajectory data (.yaml).
        label_detailed_path (str): Path to the file containing detailed labels corresponding to trajectory data (.yaml).
        augment (bool, optional): Whether to augment the data by flipping trajectories. Defaults to False.

    Returns:
        tuple: Preprocessed trajectories and associated label information.
    """
    return load_flight_data(data_path, label_path, label_detailed_path, augment, normalize_all, mean_removed_all)

