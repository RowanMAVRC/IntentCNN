#-----------------------------------------------------------------------------#
# Imports
#-----------------------------------------------------------------------------#

import os
import zipfile
import pickle
import torch
import yaml
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------#
# Functions
#-----------------------------------------------------------------------------#

def plot_padding_statistics(padding_stats, data_file_path, max_length, min_length, save_signature, output_file=None):
    """
    Creates a bar graph of padding statistics.

    Args:
        padding_stats (dict): Dictionary containing padding statistics.
        data_file_path (str): Path to the input pickle file containing sequences.
        max_length (int): Maximum length for trajectories.
        min_length (int): Minimum length for trajectories.
        save_signature (str): Unique identifier for saving the output files.
    """
    intentions = list(padding_stats.keys())
    total_paddings = [value['total_padding'] for value in padding_stats.values()]
    average_paddings = [value['total_padding'] / value['num_chunks'] for value in padding_stats.values()]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(intentions, total_paddings, color='skyblue', label='Total Padding')
    plt.bar_label(bars)
    bars = plt.barh(intentions, average_paddings, color='orange', alpha=0.5, label='Average Padding')
    plt.bar_label(bars)
    plt.xlabel('Padding')
    plt.ylabel('Intention')

    plt.title(f'Padding Statistics per Intention\nInput File: {os.path.basename(data_file_path)}, Max Length: {max_length}, Min Length: {min_length}')
    plt.legend()
    if output_file:
        plt.savefig(output_file)
    else:
        # Ensure the 'graphs' directory exists
        os.makedirs('graphs', exist_ok=True)
        plt.savefig(f'graphs/padding_statistics_{os.path.basename(data_file_path)}_{max_length}_{min_length}_{save_signature}.png')

def calculate_padding_statistics(sequences, max_length, min_length, data_file_path, save_signature):
    """
    Calculate padding statistics for trajectories.

    Args:
        sequences (dict): Dictionary containing trajectory data.
        max_length (int): Maximum length for trajectories.
        min_length (int): Minimum length for trajectories.
        data_file_path (str): Path to the input pickle file containing sequences.
        save_signature (str): Unique identifier for saving the output files.

    Returns:
        dict: Dictionary containing padding statistics.
    """
    padding_stats = {}

    for id_intention, trajectories in sequences.items():
        keep = []
        for i, trajectory in enumerate(trajectories):
            if len(trajectory) < min_length:
                continue  # Skip trajectories shorter than the minimum length
            else:
                keep.append(i)

            num_chunks = len(trajectory) // max_length
            if len(trajectory) % max_length > 0:
                num_chunks += 1

            total_length_needed = num_chunks * max_length
            total_padding_needed = total_length_needed - len(trajectory)

            padding_per_chunk, extra_padding = divmod(total_padding_needed, num_chunks)

            for i in range(num_chunks):
                padding = max_length - (len(trajectory[i * (max_length - padding_per_chunk): (i + 1) * (max_length - padding_per_chunk) - min(i, extra_padding)]))
                label = trajectory[0]['label']
                intention = trajectory[0]['intention']
                key = f"{label} - {intention}"
                if key not in padding_stats:
                    padding_stats[key] = {'total_padding': 0, 'num_chunks': 0}
                padding_stats[key]['total_padding'] += padding
                padding_stats[key]['num_chunks'] += 1

    # Plot the padding statistics
    plot_padding_statistics(padding_stats, data_file_path, max_length, min_length, save_signature)
    
    return padding_stats
        
def process_trajectories(data_file_path, save_signature, dimensions=3, max_length=100, min_factor=2/3):
    """
    Processes video data to extract and modify trajectories, and saves the results to a tensor file.

    Args:
        data_file_path (str): Path to the input pickle file containing sequences.
        save_signature (str): Unique identifier for saving the output files.
        dimensions (int): Number of dimensions for the trajectory data (2 or 3).
        max_length (int): Maximum length for trajectories.
        min_factor (float): Minimum length factor for trajectories.
    """
    # Calculate the minimum length based on the provided factor
    min_length = int(max_length * min_factor)
    print(f"Processing file: {data_file_path}")
    print(f"Max length: {max_length}, Min length: {min_length}")

    # Load the trajectory data from the pickle file
    with open(data_file_path, 'rb') as handle:
        sequences = pickle.load(handle)

    # Calculate padding statistics for the trajectories
    padding_stats = calculate_padding_statistics(sequences, max_length, min_length, data_file_path, save_signature)

    # Print padding statistics
    for key, value in padding_stats.items():
        total_padding = value['total_padding']
        num_chunks = value['num_chunks']
        average_padding = total_padding / num_chunks
        print(f"Intention: {key}, Total Padding: {total_padding}, Average Padding: {average_padding}")

    # Initialize variables
    unique_label_intentions = set()
    unique_label_detailed = set()

    # Extract unique label-intention pairs and generate mappings
    for id_intention, trajectories in sequences.items():
        for trajectory in trajectories:
            for frame in trajectory:
                label = frame['label']
                intention = frame['intention']
                label_detailed = frame['label_detailed']
                unique_label_intentions.add((label, intention))
                unique_label_detailed.add(label_detailed)
    unique_label_intentions = sorted(unique_label_intentions)
    unique_label_detailed = sorted(unique_label_detailed)

    # Generate string-to-number and number-to-string mappings for label-intention pairs
    str2num = {f"{label} - {intention}": num for num, (label, intention) in enumerate(unique_label_intentions)}
    num2str = {num: f"{label} - {intention}" for num, (label, intention) in enumerate(unique_label_intentions)}
    label_detailed2num = {label_detailed: num for num, label_detailed in enumerate(unique_label_detailed)}
    num2label_detailed = {num: label_detailed for num, label_detailed in enumerate(unique_label_detailed)}

    sequences_modified = []  # This will store the modified sequences

    # Modify trajectories to fit the desired length and format
    for id_intention, trajectories in sequences.items():
        for trajectory in trajectories:
            if len(trajectory) < min_length:
                continue  # Skip trajectories shorter than the minimum length
            
            # Calculate how many chunks we will divide the trajectory into
            num_chunks = len(trajectory) // max_length
            if len(trajectory) % max_length > 0:
                num_chunks += 1  # Add an extra chunk for the remainder if needed
            
            total_length_needed = num_chunks * max_length
            total_padding_needed = total_length_needed - len(trajectory)
            
            # Calculate padding per chunk (try to distribute it evenly)
            padding_per_chunk, extra_padding = divmod(total_padding_needed, num_chunks)
            
            # Segment the trajectory and apply padding
            for i in range(num_chunks):
                start_index = i * (max_length - padding_per_chunk) - min(i, extra_padding)
                end_index = start_index + (max_length - padding_per_chunk) - (1 if i < extra_padding else 0)
                
                chunk = trajectory[start_index:end_index]
                
                label = chunk[0]['label']
                intention = chunk[0]['intention']
                label_detailed = chunk[0]['label_detailed']
                cls_num = str2num[f"{label} - {intention}"]
                label_detailed_num = label_detailed2num[label_detailed]
                
                if dimensions == 3:
                    coords = torch.zeros(max_length+1, 4)
                    coords[0] = torch.tensor([label_detailed_num, label_detailed_num, label_detailed_num, label_detailed_num])
                    for j, frame in enumerate(chunk):
                        coords[j+1] = torch.tensor([cls_num, *frame['coords'], frame['depth']])
                elif dimensions == 2:
                    coords = torch.zeros(max_length+1, 3)
                    coords[0] = torch.tensor([label_detailed_num, label_detailed_num, label_detailed_num])
                    for j, frame in enumerate(chunk):
                        coords[j+1] = torch.tensor([cls_num, *frame['coords']])
                else:
                    raise ValueError("Invalid number of dimensions. Must be 2 or 3.")
                    
                sequences_modified.append(coords)

    # Convert the modified sequences to a torch tensor
    data = torch.stack(sequences_modified)
    
    # Define the directory path to save the output files
    if dimensions == 3:
        save_path = f'/data/TGSSE/UpdatedIntentions/XYZ/{max_length}pad_{int(min_length / max_length * 100)}/'
    elif dimensions == 2:
        save_path = f'/data/TGSSE/UpdatedIntentions/XY/{max_length}pad_{int(min_length / max_length * 100)}/'
    else:
        raise ValueError("Invalid number of dimensions. Must be 2 or 3.")
    
    # Ensure the directory exists or create it
    os.makedirs(save_path, exist_ok=True)
    
    # Define the file name for the tensor data and save it
    save_file_name = f'{save_path}trajectory_with_intentions_{max_length}_pad_{min_length}_min_{save_signature}'
    torch.save(data, f'{save_file_name}.pt')
    
    # Define the file name for the yaml mapping and save it
    with open(f'{save_file_name}.yaml', 'w') as yaml_file:
        yaml.dump(num2str, yaml_file, default_flow_style=False)
        
    with open(f'{save_file_name}_label_detailed.yaml', 'w') as yaml_file:
        yaml.dump(num2label_detailed, yaml_file, default_flow_style=False)
    
#-----------------------------------------------------------------------------#
# Main entry point of the script
#-----------------------------------------------------------------------------#
    
if __name__ == "__main__":
    
    # Define data paths and save signatures
    data_paths = [
        "/data/TGSSE/UpdatedIntentions/173857.pickle",
        "/data/TGSSE/UpdatedIntentions/101108.pickle",
        "/data/TGSSE/UpdatedIntentions/095823.pickle",
        "/data/TGSSE/UpdatedIntentions/164544.pickle",
        "/data/TGSSE/UpdatedIntentions/151221.pickle"
    ]
    save_signatures = [
        "173857",
        "101108",
        "095823",
        "164544",
        "151221"
    ]
    # min_factors = [0, 1/3, 2/3]
    min_factors = [2/3]
    min_pad = 800
    max_pad = 900
    
    total_runs = len(data_paths) * len(min_factors) * len(range(min_pad, max_pad, 100))
    current_run = 1
    
    for i in range(min_pad, max_pad, 100):
        for min_factor in min_factors:
            for data_path, save_signature in zip(data_paths, save_signatures):
                print(f"Run {current_run}/{total_runs}: Processing {data_path} with max_length={i}, min_factor={min_factor}, dimensions=3")
                process_trajectories(data_path, save_signature, dimensions=3, max_length=i, min_factor=min_factor)
                current_run += 1

                # print(f"Run {current_run}/{total_runs}: Processing {data_path} with max_length={i}, min_factor={min_factor}, dimensions=2")
                # process_trajectories(data_path, save_signature, dimensions=2, max_length=i, min_factor=min_factor)
                # current_run += 1
