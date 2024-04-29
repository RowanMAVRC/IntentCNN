import numpy as np
import matplotlib.pyplot as plt

def generate_histogram_and_pie_chart(labels, label_mapping, file_name):
    mapped_labels = np.sort(np.array([label_mapping[label] for label in labels]))
    unique, counts = np.unique(mapped_labels, return_counts=True)
    
    # Plot Histogram
    plt.figure(figsize=(15, 7))
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
    
    # Save the figure
    plt.savefig(f'{file_name}_overall_distribution.png', bbox_inches='tight')  # Adjust bounding box to fit the entire plot

def generate_histogram_and_pie_chart_for_split(train_labels, val_labels, label_mapping, file_name):
    train_mapped_labels = np.sort(np.array([label_mapping[label] for label in train_labels]))
    val_mapped_labels = np.sort(np.array([label_mapping[label] for label in val_labels]))
    
    unique_train, counts_train = np.unique(train_mapped_labels, return_counts=True)
    unique_val, counts_val = np.unique(val_mapped_labels, return_counts=True)
    
    # Plot Histogram
    plt.figure(figsize=(15, 7))
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
    
    # Save the figure
    plt.savefig(f'{file_name}_split_distribution.png', bbox_inches='tight')  # Adjust bounding box to fit the entire plot
