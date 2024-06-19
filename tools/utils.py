# ------------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------------- #

# Package Imports
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
import torch
import torch.nn as nn

# ------------------------------------------------------------------------------------- #
# Functions
# ------------------------------------------------------------------------------------- #

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

def plot_drone_intent_statistics(train_trajectories, train_labels, drone_classes, id2label, id2label_detailed, output_path):
    """
    Plot a bar graph where each drone class has a bar for each intent class, showing how many sequences are of that drone class doing that intention.

    Args:
        train_trajectories (numpy.ndarray): Array of trajectories.
        train_labels (list): List of labels corresponding to the trajectories.
        drone_classes (list): List of drone classes corresponding to the trajectories.
        id2label (dict): Mapping of label IDs to labels.
        id2label_detailed (dict): Mapping of detailed label IDs to detailed labels.
        output_path (str): Path to save the output plot.
    """
    # Create a dictionary to hold the counts
    counts = {drone_class: {intent: 0 for intent in id2label_detailed.keys()} for drone_class in id2label.keys()}

    # Count the occurrences
    for drone_class, intent in zip(drone_classes, train_labels):
        counts[drone_class[0]][intent] += 1

    # Plot the counts
    fig, ax = plt.subplots(figsize=(10, 7))
    bar_width = 0.2
    bar_positions = np.arange(len(id2label_detailed))

    for i, (drone_class, intents) in enumerate(counts.items()):
        counts_list = [intents[intent] for intent in sorted(intents.keys())]
        bar_positions_offset = bar_positions + (i * bar_width)
        ax.bar(bar_positions_offset, counts_list, bar_width, label=id2label[drone_class])

    ax.set_xlabel('Intent Classes')
    ax.set_ylabel('Number of Sequences')
    ax.set_title('Number of Sequences per Drone Class and Intent')
    ax.set_xticks(bar_positions + bar_width * (len(counts) - 1) / 2)
    ax.set_xticklabels([id2label_detailed[intent] for intent in sorted(id2label_detailed.keys())])
    ax.legend(title='Drone Classes')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
def generate_and_save_attribution_map(model, input_data, target_class, output_path):
    """
    Generate and save an attribution map using Integrated Gradients.

    Args:
        model (torch.nn.Module): The trained CNN model.
        input_data (numpy.ndarray): The input data for which to generate the attribution map.
        target_class (int): The target class for which to calculate attributions.
        output_path (str): The path to save the attribution map image.
    """
    # Preprocess the input data
    def preprocess_input(input_data):
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        return input_tensor

    input_tensor = preprocess_input(input_data)

    # Initialize Integrated Gradients
    ig = IntegratedGradients(model)

    # Calculate attributions
    attributions, delta = ig.attribute(input_tensor, target=target_class, return_convergence_delta=True)

    # Visualize attributions
    attributions = attributions.squeeze().cpu().detach().numpy()

    plt.figure(figsize=(10, 6))
    for i in range(attributions.shape[0]):
        plt.plot(attributions[i], label=f'Feature {i+1}')
    plt.title('Integrated Gradients Attributions')
    plt.xlabel('Time Steps')
    plt.ylabel('Attribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Attribution map saved as '{output_path}'")
