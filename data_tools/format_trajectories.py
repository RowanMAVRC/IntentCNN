import os
import zipfile
import pickle
import torch
import yaml
import matplotlib.pyplot as plt

def calculate_padding_statistics(sequences, max_length, min_length):
    """
    Calculate padding statistics for trajectories.

    Args:
    sequences (dict): Dictionary containing trajectory data.
    max_length (int): Maximum length for trajectories.
    min_length (int): Minimum length for trajectories.

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

    return padding_stats

def trajectories_3d():
    # Define file paths
    # data_file_path = '/data/TGSSE/UpdatedIntentions/173857.pickle'
    # save_signature = "173857"
    data_file_path = '/data/TGSSE/UpdatedIntentions/151221.pickle'
    save_signature = "151221"

    # Load the dictionary back from the pickle file.
    with open(data_file_path, 'rb') as handle:
        sequences = pickle.load(handle)

    # Define maximum length and minimum length for trajectories
    max_length = 800
    min_factor = 2/3
    min_length = int(max_length * min_factor)
    print(data_file_path)
    print(f"Max length: {max_length}, Min length: {min_length}")

    # Calculate padding statistics
    padding_stats = calculate_padding_statistics(sequences, max_length, min_length)

    # Print padding statistics
    for key, value in padding_stats.items():
        total_padding = value['total_padding']
        num_chunks = value['num_chunks']
        average_padding = total_padding / num_chunks
        print(f"Intention: {key}, Total Padding: {total_padding}, Average Padding: {average_padding}")

    # Create a bar graph of padding statistics
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

    # Include input file name, max length, and minimum length in the title
    plt.title(f'Padding Statistics per Intention\nInput File: {os.path.basename(data_file_path)}, Max Length: {max_length}, Min Length: {min_length}')

    plt.legend()
    plt.savefig(f'graphs/padding_statistics_{os.path.basename(data_file_path)}_{max_length}_{min_length}_{save_signature}.png')

    # Define save paths and file names for modified data
    save_path = '/data/TGSSE/UpdatedIntentions/XYZ/' + str(max_length) + "pad_" + str(int(min_factor * 100)) +"/"
    save_file_name = f'{save_path}trajectory_with_intentions_{max_length}_pad_{min_length}_min_{save_signature}'

    # Initialize variables
    trajectory_list = []
    unique_label_intentions = set()

    # Extract unique label-intention pairs and generate mappings
    for id_intention, trajectories in sequences.items():
        for trajectory in trajectories:
            for frame in trajectory:
                label = frame['label']
                intention = frame['intention']
                unique_label_intentions.add((label, intention))
    unique_label_intentions = sorted(unique_label_intentions)

    # Generate string-to-number and number-to-string mappings for label-intention pairs
    str2num = {}
    num2str = {}
    for num, (label, intention) in enumerate(unique_label_intentions):
        pair_str = f"{label} - {intention}"  # Format: "HELICOPTER - Hover"
        str2num[pair_str] = num
        num2str[num] = pair_str

    sequences_modified = []  # This will store the modified sequences

    # Modify trajectories to fit the desired length and format
    for id_intention, trajectories in sequences.items():
        for trajectory in trajectories:
            if len(trajectory) < min_length:
                continue  # Skip trajectories shorter than 1/3 the max length
            
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
                cls_num = str2num[f"{label} - {intention}"]
                    
                coords = torch.zeros(max_length, 4)
                for j, frame in enumerate(chunk):
                    coords[j] = torch.tensor([cls_num, *frame['coords'], frame['depth']])
                    
                    
                sequences_modified.append(coords)
        
    # Convert the modified sequences to a torch tensor
    data = torch.stack(sequences_modified)

    # Save the tensor to a file
    torch.save(data, f'{save_file_name}.pt')

    # Save the dictionary to a YAML file
    with open(f'{save_file_name}.yaml', 'w') as yaml_file:
        yaml.dump(num2str, yaml_file, default_flow_style=False)

def trajectories_2d():
    # Define file paths
    data_file_path = '/data/TGSSE/UpdatedIntentions/151221.pickle'
    save_signature = "151221"

    # Load the dictionary back from the pickle file.
    with open(data_file_path, 'rb') as handle:
        sequences = pickle.load(handle)

    # Define maximum length and minimum length for trajectories
    max_length = 100
    min_length = int(max_length * 1/3)
    print(f"Max length: {max_length}, Min length: {min_length}")
    save_path = '/data/TGSSE/UpdatedIntentions/XY/' + str(max_length) + "pad_" + str(int(1/3 * 100)) +"/"
    save_file_name = f'{save_path}trajectory_with_intentions_{max_length}_pad_{min_length}_min_{save_signature}'

    # Initialize variables
    trajectory_list = []
    unique_label_intentions = set()

    # Extract unique label-intention pairs and generate mappings
    for id_intention, trajectories in sequences.items():
        for trajectory in trajectories:
            for frame in trajectory:
                label = frame['label']
                intention = frame['intention']
                unique_label_intentions.add((label, intention))
    unique_label_intentions = sorted(unique_label_intentions)

    # Generate string-to-number and number-to-string mappings for label-intention pairs
    str2num = {}
    num2str = {}
    for num, (label, intention) in enumerate(unique_label_intentions):
        pair_str = f"{label} - {intention}"  # Format: "HELICOPTER - Hover"
        str2num[pair_str] = num
        num2str[num] = pair_str

    sequences_modified = []  # This will store the modified sequences

    # Modify trajectories to fit the desired length and format
    for id_intention, trajectories in sequences.items():
        for trajectory in trajectories:
            if len(trajectory) < min_length:
                continue  # Skip trajectories shorter than 1/3 the max length
            
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
                cls_num = str2num[f"{label} - {intention}"]
                    
                coords = torch.zeros(max_length, 3)
                for j, frame in enumerate(chunk):
                    coords[j] = torch.tensor([cls_num, *frame['coords']])
                    
                    
                sequences_modified.append(coords)
        
    # Convert the modified sequences to a torch tensor
    data = torch.stack(sequences_modified)

    # Save the tensor to a file
    torch.save(data, f'{save_file_name}.pt')

    # Save the dictionary to a YAML file
    with open(f'{save_file_name}.yaml', 'w') as yaml_file:
        yaml.dump(num2str, yaml_file, default_flow_style=False)
    
if __name__ == "__main__":
    # trajectories_2d()
    trajectories_3d()
