'''
 __  __     __      _______   _____ 
|  \/  |   /\ \    / /  __ \ / ____|
| \  / |  /  \ \  / /| |__) | |     
| |\/| | / /\ \ \/ / |  _  /| |     
| |  | |/ ____ \  /  | | \ \| |____ 
|_|  |_/_/    \_\/   |_|  \_\______|


Python Script to Convert CSV and MP4 files to Trajectories

The main steps include:
1. Finding matching pairs of CSV and MP4 files based on their base names.
2. Processing each pair, extracting label information from CSV files, and associating it with corresponding frames in MP4 files.
3. Merging the extracted label sequences into global sequences, categorized by ID and intention.
4. Calculating statistics about the sequences, including the length of the longest and shortest sequences, as well as the average sequence length.
5. Plotting the sequence length statistics for each intention group and saving the plot to a file.
6. Saving the sequence data in a pickle file along with the calculated statistics.

'''

#-----------------------------------------------------------------------------#
# Imports
#-----------------------------------------------------------------------------#

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
    '''
    Find matching pairs of CSV and MP4 files based on their base names.

    Args:
        data_path (str): The path to the directory containing CSV and MP4 files.

    Returns:
        list: A list of tuples containing matching pairs of CSV and MP4 file paths.
    '''
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


def process_video_data(data_path: str, save_path: str) -> dict:
    """
    Processes video and label data to extract sequences of labeled frames and save the results to a pickle file.

    Args:
        data_path (str): Path to the directory containing video and CSV label data.
        save_path (str): Path to save the resulting sequences as a pickle file.

    Raises:
        AssertionError: If a video file cannot be opened.
    """
    # Find matching pairs of video and label files
    matching_pairs = find_matching_pairs(data_path)
    
    sequences = {}  # Initialize a dictionary to store sequences

    # Process each pair of video and label files
    for csv_file, mp4_file in tqdm(matching_pairs, desc="Processing file pairs", position=0):

        # Open the video file
        video_capture = cv2.VideoCapture(mp4_file)
        assert video_capture.isOpened(), "Error opening " + mp4_file

        # Retrieve video properties
        num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # Load and process the label data from CSV
        video_labels = pd.read_csv(csv_file)
        video_labels.dropna(subset=['label', 'label_detailed', 'intention'], inplace=True)
        
        current_sequences = {}  # Dictionary to store sequences for the current video

        # Process each frame in the video
        for i in tqdm(range(num_frames), desc=f"Processing {os.path.basename(mp4_file).split('.')[0]}", position=1):
            frame_info = video_labels[video_labels['frame'] == i].reset_index(drop=True)

            for j in range(frame_info.shape[0]):
                label = frame_info.loc[j, "label"]
                label_detailed = frame_info.loc[j, "label_detailed"]
                object_id = frame_info.loc[j, "id"]
                coords = ast.literal_eval(frame_info.loc[j, 'coords'])
                depth = frame_info.loc[j, "depth_meters"]
                intention = frame_info.loc[j, "intention"]

                key = (object_id, intention)  # Unique key for each combination of ID and intention
                if key not in current_sequences:
                    current_sequences[key] = [[{
                        'frame': i, 'label': label, 'label_detailed': label_detailed, 'id': object_id, 'coords': coords[:2], 'depth': depth, 'intention': intention
                    }]]
                else:
                    # Check if current frame continues the last sequence or starts a new one
                    last_sequence = current_sequences[key][-1]
                    if last_sequence[-1]['frame'] == i - 1:  # Continues the sequence
                        last_sequence.append({
                            'frame': i, 'label': label, 'label_detailed': label_detailed, 'id': object_id, 'coords': coords[:2], 'depth': depth, 'intention': intention
                        })
                    else:  # Starts a new sequence
                        current_sequences[key].append([{
                            'frame': i, 'label': label, 'label_detailed': label_detailed, 'id': object_id, 'coords': coords[:2], 'depth': depth, 'intention': intention
                        }])

        # Merge current sequences into global sequences
        for key, sequence_list in current_sequences.items():
            if key in sequences:
                sequences[key].extend(sequence_list)
            else:
                sequences[key] = sequence_list

    # Save the sequences to a pickle file
    with open(save_path, 'wb') as f:
        pickle.dump(sequences, f)

    print(sequences)
    print(f"Sequences saved to {save_path}")
    
    return sequences


def plot_sequence_statistics(sequences: dict, data_path: str, 
                             output_file: str='graphs/sequence_statistics.png') -> None:
    """
    Plots the sequence statistics and saves the plot to a file.

    Args:
        sequences (dict): Dictionary containing the sequences data.
        data_path (str): Path to the data used for naming the plot file.
        output_file (str, optional): Path to save the plot file.
    """
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
    plt.savefig(output_file)
    plt.close()
    print(f"Sequence statistics saved to {output_file}")
                
#-----------------------------------------------------------------------------#
# Main entry point of the script
#-----------------------------------------------------------------------------#

if __name__ == "__main__":

    # Define the paths to the data and the save locations
    data_paths = [
        "IntentCNN/Raw/DyViR_DS_240410_101108_Optical_6D0A0B0H",
        "IntentCNN/Raw/DyViR_DS_240410_173857_Optical_6D0A0B0H",
        "IntentCNN/Raw/DyViR_DS_240410_095823_Optical_6D0A0B0H",
        "IntentCNN/Raw/DyViR_DS_240423_164544_Optical_6D0A0B0H",
        "IntentCNN/Raw/DyViR_DS_240408_151221_Optical_6D0A0B0H"
    ]
    save_paths = [
        'IntentCNN/Generated/Segmented/101108.pickle',
        'IntentCNN/Generated/Segmented/173857.pickle',
        'IntentCNN/Generated/Segmented/095823.pickle',
        'IntentCNN/Generated/Segmented/164544.pickle',
        'IntentCNN/Generated/Segmented/151221.pickle'
    ]
    
    if len(data_paths) != len(save_paths):
        raise ValueError("Number of data paths and save paths must be equal.")

    for data_path, save_path in zip(data_paths, save_paths):
        # Process video data and save the sequences to a pickle file
        sequences = process_video_data(data_path, save_path)

        # Generate the path for the graph
        graph_path = save_path.replace('.pickle', '.png')
        
        # Plot and save sequence statistics
        plot_sequence_statistics(sequences, data_path, graph_path)
