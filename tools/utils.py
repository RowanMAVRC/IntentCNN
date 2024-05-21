# ------------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------------- #

# Package Imports
import numpy as np
import matplotlib.pyplot as plt

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