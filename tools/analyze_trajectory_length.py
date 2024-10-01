"""
 _____       _             _                           
|_   _|     | |           | |    ____  _   _  _   _                           
  | |  _ __ | |_ ___ _ __ | |_  / ___|| \ | || \ | |
  | | | '_ \| __/ _ \ '_ \| __|| |   ||  \| ||  \| |
 _| |_| | | | ||  __/ | | | |_ | |___|| |\  || |\  |
|_____|_| |_|\__\___|_| |_|\__| \____||_| \_||_| \_|
                                                                 
Summary:
This script loads trajectory data from a pickle file, calculates various statistics on the trajectory lengths, 
and plots the statistics for each intention group. It also saves the plot as a PNG file and prints a summary of the statistics.

1. Load trajectory data from a pickle file.
2. Calculate overall statistics for all trajectories including the longest, shortest, and average sequence lengths.
3. Calculate statistics for each intention group including the longest, shortest, and average sequence lengths.
4. Plot the statistics for each intention group using a bar chart.
5. Annotate each bar with its corresponding value and annotate the plot with the input file name.
6. Save the plot as a PNG file.
7. Print a summary of the statistics including overall statistics and statistics by intention group.

"""

import os
import pickle
import matplotlib.pyplot as plt

def load_trajectory_data(file_path):
    """
    Load trajectory data from a pickle file.
    
    Args:
    file_path (str): Path to the pickle file.
    
    Returns:
    dict: Trajectory data loaded from the file.
    """
    with open(file_path, 'rb') as handle:
        sequences = pickle.load(handle)
    return sequences

# Define the path to the data file
data_path = "/data/TGSSE/SimData_2024-03-17__11-53-42_Optical/trajectory/trajectories_with_intentions.pickle"

# Load trajectory data from the pickle file
sequences = load_trajectory_data(data_path)

# Calculate overall statistics for all trajectories
sequence_lengths_all = [len(trajectory) for key in sequences.keys() for trajectory in sequences[key]]

# Compute overall statistics
longest_sequence_all = max(sequence_lengths_all)  # Longest sequence length
shortest_sequence_all = min(sequence_lengths_all)  # Shortest sequence length
average_sequence_length_all = sum(sequence_lengths_all) / len(sequence_lengths_all)  # Average sequence length

# Calculate statistics for each intention group
intention_stats = {}
for key, seq_list in sequences.items():
    intention = key[1]  # Extract the intention type from the key
    seq_lengths = [len(seq) for seq in seq_list]  # Lengths of all sequences in the group
    longest_seq = max(seq_lengths)  # Longest sequence in the group
    shortest_seq = min(seq_lengths)  # Shortest sequence in the group
    avg_seq_length = sum(seq_lengths) / len(seq_lengths)  # Average sequence length in the group
    # Store the statistics for each intention group
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
intention_labels = list(intention_stats.keys())  # List of intention group labels
longest_values = [intention_stats[intention]['longest'] for intention in intention_labels]  # Longest sequence per group
shortest_values = [intention_stats[intention]['shortest'] for intention in intention_labels]  # Shortest sequence per group
average_values = [intention_stats[intention]['average'] for intention in intention_labels]  # Average sequence per group

bar_width = 0.2  # Width of the bars in the plot
index = range(len(intention_labels))  # Index for the bars
plt.bar([i - bar_width for i in index], longest_values, bar_width, color='orange', label='Longest')  # Longest bars
plt.bar(index, shortest_values, bar_width, color='green', label='Shortest')  # Shortest bars
plt.bar([i + bar_width for i in index], average_values, bar_width, color='red', label='Average')  # Average bars

# Annotate each bar with its value
for i, v in enumerate(longest_values):
    plt.text(i - bar_width, v + 0.1, str(v), ha='center', va='bottom')
for i, v in enumerate(shortest_values):
    plt.text(i, v + 0.1, str(v), ha='center', va='bottom')
for i, v in enumerate(average_values):
    plt.text(i + bar_width, v + 0.1, str(round(v, 2)), ha='center', va='bottom')

# Annotate the plot with the input file name
plt.text(0.5, -0.12, f'Input File: {input_file_name}', transform=plt.gca().transAxes, fontsize=10, ha='center')

# Add labels and title to the plot
plt.xlabel('Intention Groups')
plt.ylabel('Sequence Length')
plt.title('Sequence Length Statistics by Intention Group')
plt.xticks(index, intention_labels)  # Set intention group labels on x-axis
plt.legend()  # Show the legend

# Save the chart to a file
output_file = 'graphs/sequence_statistics.png'
plt.savefig(output_file)

# Summary of statistics
print("Overall Statistics:")
print(f"Longest Sequence Length: {longest_sequence_all}")
print(f"Shortest Sequence Length: {shortest_sequence_all}")
print(f"Average Sequence Length: {average_sequence_length_all:.2f}")

print("\nStatistics by Intention Group:")
for intention, stats in intention_stats.items():
    print(f"\nIntention: {intention}")
    print(f"Longest Sequence Length: {stats['longest']}")
    print(f"Shortest Sequence Length: {stats['shortest']}")
    print(f"Average Sequence Length: {stats['average']:.2f}")

print(f"\nSequence statistics saved to {output_file}")
