import pandas as pd

# Specify the path to your CSV file
# file_path = '/data/TGSSE/DyViR Conference Paper 2024/Same POV Multi-Modality (300k)/SimData_2024-03-17__11-53-42_Optical/SimData_2024-03-17__11-53-42_Optical.csv'
file_path = "/data/TGSSE/UpdatedIntentions/DyViR_DS_240410_095823_Optical_6D0A0B0H/DyViR_DS_240410_095823_Optical_6D0A0B0H.csv"
# Load the CSV file into a DataFrame
data = pd.read_csv(file_path)

# Print the first 5 rows of the DataFrame
print(data.head())
print(data.columns)

# Get unique values from 'label_detailed' column
unique_label_detailed = data['label_detailed      '].unique()
print("Unique values in 'label_detailed' column:")
print(unique_label_detailed)

# Get unique values from 'intention   ' column
unique_intention = data['intention   '].unique()
print("Unique values in 'intention   ' column:")
print(unique_intention)

# Get unique combinations of 'label_detailed' and 'intention   ' columns
unique_combinations = data[['label_detailed      ', 'intention   ']].drop_duplicates()

combination_counts = data.groupby(['label_detailed      ', 'intention   ']).size().reset_index(name='Count')

print(combination_counts)
exit()

# Group the unique combinations by 'label_detailed'
grouped_combinations = unique_combinations.groupby('label_detailed      ')

# Print the groups
print("Groups of unique combinations by 'label_detailed':")
for i, (name, group) in enumerate(grouped_combinations):
    print(f"\nLabel Detailed: {name}")
    print(group)
    print(f"Num: {combination_counts[i]}")