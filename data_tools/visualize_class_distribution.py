import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import yaml

def plot_label_statistics(labels_tensor, label_mapping, save_path, file_name):
    labels = labels_tensor.numpy() # Convert tensor to numpy array for easier processing with matplotlib
    mapped_labels = np.array([label_mapping[label] for label in labels]) # Map numeric labels to string labels
    
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
    
    # Plot Pie Chart
    plt.subplot(1, 3, 2)
    plt.pie(counts, labels=unique, autopct='%1.1f%%', startangle=140)
    plt.title('Label Proportion')
    
    # Plot Bar Chart
    plt.subplot(1, 3, 3)
    plt.bar(unique, counts, color='skyblue')
    plt.title('Label Count')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_all_label_statistics(label_tensors, label_mappings, path_names, save_path):
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
    
    if len(path_names) > 5:  # Remove legend if the number of files is greater than 5
        show_legend = False
    else:
        show_legend = True
    
    for labels_tensor, label_mapping, path_name in zip(label_tensors, label_mappings, path_names):
        labels = labels_tensor.numpy()  # Convert tensor to numpy array for easier processing
        # Map numeric labels to string labels according to each dataset's specific mapping
        mapped_labels = [label_mapping[label] for label in labels]
        # Convert string labels to indices for consistent binning
        label_indices = [label_to_index[label] for label in mapped_labels]
        
        plt.hist(label_indices, bins=bins, alpha=0.5, label=path_name, edgecolor='black', align='mid')
    
    plt.title('Label Distribution Across Paths')
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    
    # Use the label order for x-ticks
    plt.xticks(ticks=np.arange(len(label_order)), labels=label_order, rotation=45, ha="right")
    
    if show_legend:
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Function to recursively get all files with a given extension in a directory
def get_files_recursive(root_dir, extension='.pt'):
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(os.path.join(dirpath, filename))
    return files

root_directory = "/data/TGSSE/UpdatedIntentions/XY/"
paths = get_files_recursive(root_directory)

label_mapping = {
    0: 'DRONE - Area Denial',
    1: 'DRONE - Kamikaze',
    2: 'DRONE - Recon',
    3: 'DRONE - Travel'
}

label_tensors = []
label_mappings = []
for i, path in enumerate(paths):
    data = torch.load(path)
    labels = data[:,0,0].view(-1)
    label_tensors.append(labels)

    yaml_path = path.rsplit('.', 1)[0] + '.yaml'
    with open(yaml_path, 'r') as file:
        label_mapping = yaml.safe_load(file)

    label_mappings.append(label_mapping)

    plot_label_statistics(labels, label_mapping, f"temp{i}.png", os.path.basename(path))  # Pass file name

plot_all_label_statistics(label_tensors, label_mappings, paths, f"temp{i+1}.png")
