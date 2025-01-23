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
import torch
import matplotlib.pyplot as plt
import numpy as np
import yaml

# ------------------------------------------------------------------------------------- #
# Functions
# ------------------------------------------------------------------------------------- #

def plot_label_statistics(labels_tensor, label_mapping, save_path, file_name):
    """
    Plot label statistics including distribution, proportion, and count.

    Args:
    - labels_tensor (torch.Tensor): Tensor containing labels.
    - label_mapping (dict): Mapping of numeric labels to string labels.
    - save_path (str): Path to save the plot.
    - file_name (str): Name of the file being processed.
    """
    # Convert tensor to numpy array for easier processing with matplotlib
    labels = labels_tensor.numpy()
    # Map numeric labels to string labels
    mapped_labels = np.array([label_mapping[label] for label in labels])

    # Calculate unique labels and their counts
    unique, counts = np.unique(mapped_labels, return_counts=True)
    
    # Plot Histogram
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 3, 1)
    plt.hist(mapped_labels, bins=np.arange(len(unique)+1) - 0.5, edgecolor='black')
    plt.title('Label Distribution')
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha="right")
    plt.suptitle(file_name, fontsize=16, y=0.95)  # Add file name as title
    plt.text(0.5, 0.5, f"Total Labels: {len(labels)}", fontsize=12, transform=plt.gca().transAxes)
    
    # Plot Pie Chart
    plt.subplot(1, 3, 2)
    plt.pie(counts, labels=unique, autopct='%1.1f%%', startangle=140)
    plt.title('Label Proportion')
    plt.text(0.5, 0.5, f"Total Labels: {len(labels)}", fontsize=12, transform=plt.gca().transAxes)
    
    # Plot Bar Chart
    plt.subplot(1, 3, 3)
    plt.bar(unique, counts, color='skyblue')
    plt.title('Label Count')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha="right")
    plt.text(0.5, 0.5, f"Total Labels: {len(labels)}", fontsize=12, transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_all_label_statistics(label_tensors, label_mappings, path_names, save_path):
    """
    Plot label statistics across multiple datasets.

    Args:
    - label_tensors (list of torch.Tensor): List of tensors containing labels.
    - label_mappings (list of dict): List of mappings of numeric labels to string labels.
    - path_names (list of str): List of file paths.
    - save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(15, 7))
    
    # Find the union of all labels across the different mappings
    all_labels = set()
    for mapping in label_mappings:
        all_labels.update(mapping.values())
    
    # Create an ordered list of unique labels
    label_order = sorted(list(all_labels))
    
    # Map each label to its index in label_order to create consistent bins
    label_to_index = {label: i for i, label in enumerate(label_order)}
    
    # Create bins for each label index plus one for the end; subtract 0.5 for alignment
    bins = np.arange(len(label_order) + 1) - 0.5
    
    # Decide whether to show legend based on the number of files
    show_legend = len(path_names) <= 5
    
    for labels_tensor, label_mapping, path_name in zip(label_tensors, label_mappings, path_names):
        labels = labels_tensor.numpy()
        mapped_labels = [label_mapping[label] for label in labels]
        label_indices = [label_to_index[label] for label in mapped_labels]
        
        plt.hist(label_indices, bins=bins, alpha=0.5, label=path_name, edgecolor='black', align='mid')
    
    plt.title('Label Distribution Across Paths')
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.xticks(ticks=np.arange(len(label_order)), labels=label_order, rotation=45, ha="right")
    
    if show_legend:
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def get_files_recursive(root_dir, extension='.pt'):
    """
    Recursively get all files with a given extension in a directory.

    Args:
    - root_dir (str): Root directory to start the search from.
    - extension (str): File extension to filter files.

    Returns:
    - files (list of str): List of file paths.
    """
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(os.path.join(dirpath, filename))
    return files

# ------------------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Define root directory and get file paths
    root_directory = "IntentCNN/Useable/XYZ/800pad_66"
    paths = get_files_recursive(root_directory)

    # Define label mapping
    label_mapping = {
        0: 'DRONE - Area Denial',
        1: 'DRONE - Kamikaze',
        2: 'DRONE - Recon',
        3: 'DRONE - Travel'
    }

    label_tensors = []
    label_mappings = []

    # Load data, process labels, and plot statistics for each file
    for i, path in enumerate(paths):
        data = torch.load(path)
        labels = data[:, 0, 0].view(-1)
        label_tensors.append(labels)

        # Load label mapping from YAML file
        yaml_path = "/data/TGSSE/UpdatedIntentions/labels.yaml"
        with open(yaml_path, 'r') as file:
            label_mapping = yaml.safe_load(file)

        label_mappings.append(label_mapping)

        # Plot label statistics for the current file
        plot_label_statistics(labels, label_mapping, f"temp{i}.png", os.path.basename(path))

    # Plot overall label statistics across all files
    plot_all_label_statistics(label_tensors, label_mappings, paths, f"temp{i+1}.png")
