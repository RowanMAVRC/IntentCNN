"""
 _____       _             _                           
|_   _|     | |           | |    ____  _   _  _   _                           
  | |  _ __ | |_ ___ _ __ | |_  / ___|| \ | || \ | |
  | | | '_ \| __/ _ \ '_ \| __|| |   ||  \| ||  \| |
 _| |_| | | | ||  __/ | | | |_ | |___|| |\  || |\  |
|_____|_| |_|\__\___|_| |_|\__| \____||_| \_||_| \_|
"""
#-----------------------------------------------------------------------------#
# Imports
#-----------------------------------------------------------------------------#

# Standard library imports
import os
import zipfile
import pickle
import torch
import yaml

# Third-party imports
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------#
# Functions
#-----------------------------------------------------------------------------#

def plot_padding_statistics(padding_stats, data_file_path, max_length, min_length, save_signature, output_file=None):
    """
    Creates a bar graph of padding statistics for each intention group.

    Args:
        padding_stats (dict): Dictionary containing padding statistics for each intention.
        data_file_path (str): Path to the input pickle file containing trajectory sequences.
        max_length (int): Maximum trajectory length used for padding calculations.
        min_length (int): Minimum length of trajectories allowed.
        save_signature (str): Unique identifier for saving the output files.
        output_file (str, optional): File path to save the output plot. Defaults to None.
    """
    # Extract labels and statistics
    intentions = list(padding_stats.keys())
    total_paddings = [value['total_padding'] for value in padding_stats.values()]
    average_paddings = [value['total_padding'] / value['num_chunks'] for value in padding_stats.values()]
    num_traj = [value['num_chunks'] for value in padding_stats.values()]

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.barh(intentions, num_traj, color='skyblue', label='Num Trajectories')
    plt.bar_label(bars)
    bars = plt.barh(intentions, average_paddings, color='orange', alpha=0.5, label='Average Padding')
    plt.bar_label(bars)
    plt.xlabel('Padding')
    plt.ylabel('Intention')

    # Title and save the plot
    plt.title(f'Padding Statistics per Intention\nInput File: {os.path.basename(data_file_path)}, Max Length: {max_length}, Min Length: {min_length}')
    plt.legend()

    if output_file:
        plt.savefig(output_file)
    else:
        # Default output folder based on the input file
        base_name = os.path.basename(data_file_path).replace('.pickle', '')
        folder_path = f'graphs/{base_name}/'
        os.makedirs(folder_path, exist_ok=True)

        # Save plot to folder
        save_path = f'{folder_path}/{max_length}_{min_length}.png'
        plt.savefig(save_path)


def calculate_padding_statistics(sequences, max_length, min_length, data_file_path, save_signature):
    """
    Calculate padding statistics for trajectory sequences and plot them.

    Args:
        sequences (dict): Dictionary containing trajectory data.
        max_length (int): Maximum length for trajectories.
        min_length (int): Minimum length for trajectories.
        data_file_path (str): Path to the input pickle file containing sequences.
        save_signature (str): Unique identifier for saving the output files.

    Returns:
        dict: Dictionary containing padding statistics per intention group.
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

            # Process each chunk of trajectory and collect padding stats
            for i in range(num_chunks):
                padding = max_length - (len(trajectory[i * (max_length - padding_per_chunk): (i + 1) * (max_length - padding_per_chunk) - min(i, extra_padding)]))
                label = trajectory[0]['label_detailed'].split("|")[0]
                if label in ["HEX1", "QUAD"]:
                    label = "ROTARY"
                elif label in ["SWITCHBLADE", "BAYRAKTART2", "BAYRAKTAR"]:
                    label = "FIXEDWING"
                else:
                    print(label, "is Unknown in calculate_padding_statistics")
                    exit()
                intention = trajectory[0]['intention']
                key = f"{label} - {intention}"
                
                # Store padding stats
                if key not in padding_stats:
                    padding_stats[key] = {'total_padding': 0, 'num_chunks': 0}
                padding_stats[key]['total_padding'] += padding
                padding_stats[key]['num_chunks'] += 1

    # Plot the padding statistics
    plot_padding_statistics(padding_stats, data_file_path, max_length, min_length, save_signature)
    
    return padding_stats
        
        
def process_trajectories(data_file_path, save_signature, dimensions=3, max_length=100, min_factor=2/3):
    """
    Processes trajectory data, adjusts it according to maximum and minimum lengths, 
    and saves the results in a PyTorch tensor format along with padding statistics.

    Args:
        data_file_path (str): Path to the input pickle file containing sequences.
        save_signature (str): Unique identifier for saving the output files.
        dimensions (int): Number of dimensions for the trajectory data (2 for XY, 3 for XYZ).
        max_length (int): Maximum length for each trajectory.
        min_factor (float): Minimum length factor to calculate minimum trajectory length.
    """
    min_length = int(max_length * min_factor)
    print(f"Processing file: {data_file_path}")
    print(f"Max length: {max_length}, Min length: {min_length}")

    # Load the trajectory data from the pickle file
    with open(data_file_path, 'rb') as handle:
        sequences = pickle.load(handle)

    # Calculate and print padding statistics
    padding_stats = calculate_padding_statistics(sequences, max_length, min_length, data_file_path, save_signature)
    for key, value in padding_stats.items():
        total_padding = value['total_padding']
        num_chunks = value['num_chunks']
        average_padding = total_padding / num_chunks
        print(f"Intention: {key}, Total Padding: {total_padding}, Average Padding: {average_padding}")

    # Initialize label-intention mappings
    unique_label_intentions = set()
    unique_label_detailed = set()

    for id_intention, trajectories in sequences.items():
        for trajectory in trajectories:
            for frame in trajectory:
                label_detailed = frame['label_detailed']
                if label_detailed.split("|")[0] in ["HEX1", "QUAD"]:
                    label = "ROTARY"
                elif label_detailed.split("|")[0] in ["SWITCHBLADE", "BAYRAKTART2", "BAYRAKTAR"]:
                    label = "FIXEDWING"
                else:
                    print(label_detailed.split("|")[0], "is Unknown in process_trajectories")
                    exit()

                intention = frame['intention']
                unique_label_intentions.add((label, intention))
                unique_label_detailed.add(label_detailed)

    unique_label_intentions = sorted(unique_label_intentions)
    unique_label_detailed = sorted(unique_label_detailed)

    # Map string labels to numbers for the tensor
    str2num = {f"{label} - {intention}": num for num, (label, intention) in enumerate(unique_label_intentions)}
    num2str = {num: f"{label} - {intention}" for num, (label, intention) in enumerate(unique_label_intentions)}
    label_detailed2num = {label_detailed: num for num, label_detailed in enumerate(unique_label_detailed)}
    num2label_detailed = {num: label_detailed for num, label_detailed in enumerate(unique_label_detailed)}

    sequences_modified = []  
    # Adjust trajectories and apply padding
    for id_intention, trajectories in sequences.items():
        for trajectory in trajectories:
            if len(trajectory) < min_length:
                continue  # Skip short trajectories
            
            num_chunks = len(trajectory) // max_length
            if len(trajectory) % max_length > 0:
                num_chunks += 1  # Add extra chunk if needed
            
            total_length_needed = num_chunks * max_length
            total_padding_needed = total_length_needed - len(trajectory)
            padding_per_chunk, extra_padding = divmod(total_padding_needed, num_chunks)

            # Process each chunk of trajectory
            for i in range(num_chunks):
                start_index = i * (max_length - padding_per_chunk) - min(i, extra_padding)
                end_index = start_index + (max_length - padding_per_chunk) - (1 if i < extra_padding else 0)
                
                chunk = trajectory[start_index:end_index]
                label_detailed = chunk[0]['label_detailed']
                if label_detailed.split("|")[0] in ["HEX1", "QUAD"]:
                    label = "ROTARY"
                elif label_detailed.split("|")[0] in ["SWITCHBLADE", "BAYRAKTART2", "BAYRAKTAR"]:
                    label = "FIXEDWING"
                else:
                    print(label_detailed.split("|")[0], "is Unknown in process_trajectories")
                    exit()

                intention = chunk[0]['intention']
                cls_num = str2num[f"{label} - {intention}"]
                label_detailed_num = label_detailed2num[label_detailed]

                # Create tensors for trajectory data
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

    # Convert sequences to PyTorch tensor and save
    data = torch.stack(sequences_modified)
    if dimensions == 3:
        save_path = f'IntentCNN/Generated/XYZ/{max_length}pad_{int(min_length / max_length * 100)}/'
    elif dimensions == 2:
        save_path = f'IntentCNN/Generated/XY/{max_length}pad_{int(min_length / max_length * 100)}/'
    else:
        raise ValueError("Invalid number of dimensions. Must be 2 or 3.")
    
    os.makedirs(save_path, exist_ok=True)
    save_file_name = f'{save_path}trajectory_with_intentions_{max_length}_pad_{min_length}_min_{save_signature}'
    torch.save(data, f'{save_file_name}.pt')

    # Save the label-intention mappings to yaml files
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
        "IntentCNN/Segmented/173857.pickle",
        "IntentCNN/Segmented/101108.pickle",
        "IntentCNN/Segmented/095823.pickle",
        "IntentCNN/Segmented/164544.pickle",
        "IntentCNN/Segmented/151221.pickle"
    ]
    save_signatures = [
        "173857",
        "101108",
        "095823",
        "164544",
        "151221"
    ]
    dimensions = [2, 3]  # 2 for XY, 3 for XYZ
    min_factors = [0, 1/3, 2/3]  # Minimum length factors
    min_pad = 100
    max_pad = 900
    
    total_runs = len(data_paths) * len(min_factors) * len(range(min_pad, max_pad, 100)) * 2
    current_run = 1
    
    for i in [800]:  # Example run with max length of 800
        for min_factor in min_factors:
            for data_path, save_signature in zip(data_paths, save_signatures):
                for dim in dimensions:
                    print(f"Run {current_run}/{total_runs}: Processing {data_path} with max_length={i}, min_factor={min_factor}, dimensions={dim}")
                    
                    process_trajectories(
                        data_path, 
                        save_signature, 
                        dimensions=dim, 
                        max_length=i, 
                        min_factor=min_factor
                    )

                    current_run += 1
