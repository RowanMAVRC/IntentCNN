'''
 __  __     __      _______   _____ 
|  \/  |   /\ \    / /  __ \ / ____|
| \  / |  /  \ \  / /| |__) | |     
| |\/| | / /\ \ \/ / |  _  /| |     
| |  | |/ ____ \  /  | | \ \| |____ 
|_|  |_/_/    \_\/   |_|  \_\\_____|

Python Script to Convert CSV and MP4 files to Trajectories

'''

#-----------------------------------------------------------------------------#
# Imports
#-----------------------------------------------------------------------------#

# Standard library imports
import argparse  # For parsing command-line arguments
import ast  # For safely evaluating strings containing Python expressions
import os  # For interacting with the operating system
import yaml  # For reading YAML files
import pickle  # For serializing and deserializing Python objects

# Third-party library imports
import cv2  # OpenCV library for video processing
import numpy as np  # NumPy library for numerical operations
import pandas as pd  # Pandas library for data manipulation
from tqdm import tqdm  # tqdm library for progress bars
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------#
# Functions
#-----------------------------------------------------------------------------#

def find_matching_pairs(data_path):
    # Initialize lists to hold the full paths of CSV and MP4 files
    csv_files = []
    mp4_files = []

    # Traverse through the directory structure
    for root, dirs, files in os.walk(data_path):
        # Append CSV files to the csv_files list
        csv_files.extend([os.path.join(root, file) for file in files if file.endswith('.csv')])
        
        # Check if 'VisualData' directory exists and append MP4 files accordingly
        if 'VisualData' in dirs:
            visual_data_path = os.path.join(root, 'VisualData')
            mp4_files.extend([os.path.join(visual_data_path, file) for file in os.listdir(visual_data_path) if file.endswith('.mp4')])
        else:
            mp4_files.extend([os.path.join(root, file) for file in files if file.endswith('.mp4')])

    # Find matching pairs of CSV and MP4 files based on their base name
    pairs = []
    for csv_file in csv_files:
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        for mp4_file in mp4_files:
            if os.path.splitext(os.path.basename(mp4_file))[0] == base_name:
                pairs.append((csv_file, mp4_file))
                break

    return pairs
                
#-----------------------------------------------------------------------------#
# Main entry point of the script
#-----------------------------------------------------------------------------#

if __name__ == "__main__":

    # Extract paths from the parsed arguments
    data_path = "/data/TGSSE/UpdatedIntentions/DyViR_DS_240410_173857_Optical_6D0A0B0H"
    save_path = '/data/TGSSE/UpdatedIntentions/173857.pickle'
    # data_path = "/data/TGSSE/UpdatedIntentions/DyViR_DS_240410_101108_Optical_6D0A0B0H"
    # save_path = '/data/TGSSE/UpdatedIntentions/101108.pickle'
    # data_path = "/data/TGSSE/UpdatedIntentions/DyViR_DS_240410_095823_Optical_6D0A0B0H"
    # save_path = '/data/TGSSE/UpdatedIntentions/095823.pickle'

    # Replace 'data_path' with the path to your dataset directory
    matching_pairs = find_matching_pairs(data_path)
    
    sequences = {}  

    # Process each pair
    for csv_file, mp4_file in tqdm(matching_pairs, desc="Processing file pairs", position=0):

        # Open the video file
        video_capture = cv2.VideoCapture(mp4_file)
        assert video_capture.isOpened(), "Error opening " + mp4_file

        # Retrieve video properties
        num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # Load and process the label data from CSV
        video_labels = pd.read_csv(csv_file)
        video_labels.dropna(subset=['label'], inplace=True)
        video_labels.dropna(subset=['intention'], inplace=True)
        
        current_sequences = {}  

        # Process each frame in the video
        for i in tqdm(range(num_frames), desc=f"Processing {os.path.basename(mp4_file).split('.')[0]}", position=1):
            frame_info = video_labels[video_labels['frame'] == i].reset_index(drop=True)

            # Prepare and write label information for each label in frame
            # file_path = f'{data_path}labels/{csv_file.split("/")[-1].split(".")[-2]}_{i}.txt'
            # with open(file_path, 'w') as file:
            for j in tqdm(range(frame_info.shape[0]), desc=f"Writing labels for frame {i}", position=2, disable=True):
                label = frame_info.loc[j, "label"]
                id = frame_info.loc[j, "id"]
                coords = ast.literal_eval(frame_info.loc[j, 'coords'])
                depth = frame_info.loc[j, "depth_meters"]
                intention = frame_info.loc[j, "intention"]

                key = (id, intention)  # Unique key for each combination of ID and intention
                if key not in current_sequences:
                    current_sequences[key] = [[{
                        'frame': i, 'label': label, 'id': id, 'coords': coords[:2], 'depth': depth, 'intention': intention
                    }]]
                else:
                    # Check if current frame continues the last sequence or starts a new one
                    last_sequence = current_sequences[key][-1]
                    if last_sequence[-1]['frame'] == i - 1:  # Continues the sequence
                        last_sequence.append({
                            'frame': i, 'label': label, 'id': id, 'coords': coords[:2], 'depth': depth, 'intention': intention
                        })
                    else:  # Starts a new sequence
                        current_sequences[key].append([{
                            'frame': i, 'label': label, 'id': id, 'coords': coords[:2], 'depth': depth, 'intention': intention
                        }])

        # After processing a pair, merge current sequences into global sequences
        for key, sequence_list in tqdm(current_sequences.items(), desc="Merging sequences", position=4, disable=True):
            if key in sequences:
                sequences[key].extend(sequence_list)
            else:
                sequences[key] = sequence_list

    print(sequences)
    # Calculate overall statistics
    sequence_lengths_all = [len(seq) for seq_list in sequences.values() for seq in seq_list]
    longest_sequence_all = max(sequence_lengths_all)
    shortest_sequence_all = min(sequence_lengths_all)
    average_sequence_length_all = sum(sequence_lengths_all) / len(sequence_lengths_all)

    # Calculate statistics for each intention group
    intention_stats = {}
    for key, seq_list in sequences.items():
        intention = key[1]
        seq_lengths = [len(seq) for seq in seq_list]
        longest_seq = max(seq_lengths)
        shortest_seq = min(seq_lengths)
        avg_seq_length = sum(seq_lengths) / len(seq_lengths)
        intention_stats[intention] = {
            'longest': longest_seq,
            'shortest': shortest_seq,
            'average': avg_seq_length
        }

    # Extract the base name of the input file
    input_file_name = os.path.basename(data_path)

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot statistics for each intention group
    intention_labels = list(intention_stats.keys())
    longest_values = [intention_stats[intention]['longest'] for intention in intention_labels]
    shortest_values = [intention_stats[intention]['shortest'] for intention in intention_labels]
    average_values = [intention_stats[intention]['average'] for intention in intention_labels]

    bar_width = 0.2
    index = range(len(intention_labels))
    plt.bar([i - bar_width for i in index], longest_values, bar_width, color='orange', label='Longest')
    plt.bar(index, shortest_values, bar_width, color='green', label='Shortest')
    plt.bar([i + bar_width for i in index], average_values, bar_width, color='red', label='Average')

    # Annotate each bar with its value
    for i, v in enumerate(longest_values):
        plt.text(i - bar_width, v + 0.1, str(v), ha='center', va='bottom')
    for i, v in enumerate(shortest_values):
        plt.text(i, v + 0.1, str(v), ha='center', va='bottom')
    for i, v in enumerate(average_values):
        plt.text(i + bar_width, v + 0.1, str(round(v, 2)), ha='center', va='bottom')

    # Annotate the plot with the input file name
    plt.text(0.5, -0.12, f'Input File: {input_file_name}', transform=plt.gca().transAxes, fontsize=10, ha='center')

    plt.xlabel('Statistics')
    plt.ylabel('Sequence Length')
    plt.title('Sequence Length Statistics')
    plt.xticks(index, intention_labels)
    plt.legend()

    # Save the chart to a file
    output_file = 'sequence_statistics.png'
    plt.savefig(output_file)
    
    with open(save_path, 'wb') as handle:
        pickle.dump(sequences, handle, protocol=pickle.HIGHEST_PROTOCOL)