import pandas as pd

# Specify the path to your CSV file
# Example path for the file you want to analyze (commented out for reference)
# file_path = '/data/TGSSE/DyViR Conference Paper 2024/Same POV Multi-Modality (300k)/SimData_2024-03-17__11-53-42_Optical/SimData_2024-03-17__11-53-42_Optical.csv'

# Actual file path used for loading the CSV file
file_path = "IntentCNN/Raw/DyViR_DS_240410_095823_Optical_6D0A0B0H/DyViR_DS_240410_095823_Optical_6D0A0B0H.csv"

# Load the CSV file into a DataFrame
data = pd.read_csv(file_path)

# Display the first 5 rows of the DataFrame for initial inspection
print(data.head())

# Print out the column names for clarity, since there are spaces after some column names
print(data.columns)

# Get unique values from the 'label_detailed' column (trailing spaces in column name need to be addressed)
# The column name has extra spaces, so we need to use the exact name as it appears
unique_label_detailed = data['label_detailed      '].unique()
print("Unique values in 'label_detailed' column:")
print(unique_label_detailed)

# Get unique values from the 'intention' column (also contains trailing spaces)
unique_intention = data['intention   '].unique()
print("Unique values in 'intention' column:")
print(unique_intention)

# Get unique combinations of 'label_detailed' and 'intention' columns
# This finds all distinct combinations of label_detailed and intention in the dataset
unique_combinations = data[['label_detailed      ', 'intention   ']].drop_duplicates()

# Count occurrences of each combination of 'label_detailed' and 'intention'
# This groups the data by label and intention and counts how often each combination appears
combination_counts = data.groupby(['label_detailed      ', 'intention   ']).size().reset_index(name='Count')

# Print the counts of each combination
print(combination_counts)
exit()

# Group the unique combinations by 'label_detailed' for further analysis
grouped_combinations = unique_combinations.groupby('label_detailed      ')

# Print the grouped combinations for each 'label_detailed'
print("Groups of unique combinations by 'label_detailed':")
for i, (name, group) in enumerate(grouped_combinations):
    print(f"\nLabel Detailed: {name}")
    print(group)
    print(f"Num: {combination_counts[i]}")
