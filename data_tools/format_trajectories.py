
import os
import zipfile
import pickle
import torch
import yaml

def trajectories_3d():
    # Define file paths
    # data_file_path = '/data/TGSSE/UpdatedIntentions/095823.pickle'
    # save_signature = "095823"
    # data_file_path = '/data/TGSSE/UpdatedIntentions/095823.pickle'
    # save_signature = "101108"
    data_file_path = '/data/TGSSE/UpdatedIntentions/173857.pickle'
    save_signature = "173857"

    # Load the dictionary back from the pickle file.
    with open(data_file_path, 'rb') as handle:
        sequences = pickle.load(handle)

    # Define maximum length for trajectories
    max_length = 100
    min_length = int(max_length * 0)
    print(f"Max length: {max_length}, Min length: {min_length}")
    save_path = '/data/TGSSE/UpdatedIntentions/' + str(max_length) + "pad_" + str(int(0 * 100)) +"/"
    save_file_name = f'{save_path}trajectory_with_intentions_{max_length}_pad_{min_length}_min_{save_signature}'

    # Initialize variables
    trajectory_list = []
    unique_label_intentions = set()

    # Step 1: Extract unique label-intention pairs from the sequences and sort them alphabetically
    for id_intention, trajectories in sequences.items():
        for trajectory in trajectories:
            for frame in trajectory:
                label = frame['label']
                intention = frame['intention']
                unique_label_intentions.add((label, intention))
    unique_label_intentions = sorted(unique_label_intentions)

    # Step 2: Generate string-to-number and number-to-string mappings for label-intention pairs
    str2num = {}
    num2str = {}
    for num, (label, intention) in enumerate(unique_label_intentions):
        pair_str = f"{label} - {intention}"  # Format: "HELICOPTER - Hover"
        str2num[pair_str] = num
        num2str[num] = pair_str

    # Display mappings for debugging
    print("String to Number mapping:", str2num)
    print("Number to String mapping:", num2str)

    sequences_modified = []  # This will store the modified sequences

    # Step 3: Modify trajectories to fit the desired length and format
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
            
            # Now segment the trajectory and apply padding
            for i in range(num_chunks):
                start_index = i * (max_length - padding_per_chunk) - min(i, extra_padding)
                end_index = start_index + (max_length - padding_per_chunk) - (1 if i < extra_padding else 0)
                
                chunk = trajectory[start_index:end_index]
                
                # # Apply zero-padding
                # chunk += [{'padding': 0}] * (max_length - len(chunk))  # Assuming a dict format for padding
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

    # # Create a ZIP file containing both the tensor file and the YAML file
    # with zipfile.ZipFile(f'{save_file_name}.zip', 'w') as zipf:
    #     zipf.write(f'{save_file_name}.pt')
    #     zipf.write(f'{save_file_name}.yaml')

    # # Clean up by removing the individual files if they are no longer needed
    # os.remove(f'{save_file_name}.pt')
    # os.remove(f'{save_file_name}.yaml')

def trajectories_2d():
    # Define file paths
    data_file_path = '/data/TGSSE/UpdatedIntentions/095823.pickle'
    save_signature = "095823"
    # data_file_path = '/data/TGSSE/UpdatedIntentions/095823.pickle'
    # save_signature = "101108"
    # data_file_path = '/data/TGSSE/UpdatedIntentions/173857.pickle'
    # save_signature = "173857"

    # Load the dictionary back from the pickle file.
    with open(data_file_path, 'rb') as handle:
        sequences = pickle.load(handle)

    # Define maximum length for trajectories
    max_length = 100
    min_length = int(max_length * 1/3)
    print(f"Max length: {max_length}, Min length: {min_length}")
    save_path = '/data/TGSSE/UpdatedIntentions/XY/' + str(max_length) + "pad_" + str(int(1/3 * 100)) +"/"
    save_file_name = f'{save_path}trajectory_with_intentions_{max_length}_pad_{min_length}_min_{save_signature}'

    # Initialize variables
    trajectory_list = []
    unique_label_intentions = set()

    # Step 1: Extract unique label-intention pairs from the sequences and sort them alphabetically
    for id_intention, trajectories in sequences.items():
        for trajectory in trajectories:
            for frame in trajectory:
                label = frame['label']
                intention = frame['intention']
                unique_label_intentions.add((label, intention))
    unique_label_intentions = sorted(unique_label_intentions)

    # Step 2: Generate string-to-number and number-to-string mappings for label-intention pairs
    str2num = {}
    num2str = {}
    for num, (label, intention) in enumerate(unique_label_intentions):
        pair_str = f"{label} - {intention}"  # Format: "HELICOPTER - Hover"
        str2num[pair_str] = num
        num2str[num] = pair_str

    # Display mappings for debugging
    print("String to Number mapping:", str2num)
    print("Number to String mapping:", num2str)

    sequences_modified = []  # This will store the modified sequences

    # Step 3: Modify trajectories to fit the desired length and format
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
            
            # Now segment the trajectory and apply padding
            for i in range(num_chunks):
                start_index = i * (max_length - padding_per_chunk) - min(i, extra_padding)
                end_index = start_index + (max_length - padding_per_chunk) - (1 if i < extra_padding else 0)
                
                chunk = trajectory[start_index:end_index]
                
                # # Apply zero-padding
                # chunk += [{'padding': 0}] * (max_length - len(chunk))  # Assuming a dict format for padding
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

    # # Create a ZIP file containing both the tensor file and the YAML file
    # with zipfile.ZipFile(f'{save_file_name}.zip', 'w') as zipf:
    #     zipf.write(f'{save_file_name}.pt')
    #     zipf.write(f'{save_file_name}.yaml')

    # # Clean up by removing the individual files if they are no longer needed
    # os.remove(f'{save_file_name}.pt')
    # os.remove(f'{save_file_name}.yaml')
    
if __name__ == "__main__":
    trajectories_2d()
    # trajectories_3d()