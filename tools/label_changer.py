import pandas as pd

# Load the CSV file into a DataFrame
csv_file_path = "/data/TGSSE/DyViR Conference Paper 2024/Same POV Multi-Modality (300k)/SimData_2024-03-17__11-53-42_Optical/Temp.csv"
data = pd.read_csv(csv_file_path)

# Display the unique intention values before updating
print("Unique intention values before update:")
print(data['intention'].unique())

# Define the old intention and the new intention
old_intention = "Follow Path"
new_intention = "Travel"

# Update the intention values
data.loc[data['intention'] == old_intention, 'intention'] = new_intention

# Display the unique intention values after updating
print("\nUnique intention values after update:")
print(data['intention'].unique())

# Save the updated DataFrame back to the CSV file
data.to_csv(csv_file_path, index=False)
