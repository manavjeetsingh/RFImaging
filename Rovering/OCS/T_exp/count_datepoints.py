import os

# Get all folders in the current directory
current_path = os.getcwd()
folders = [f for f in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, f))]

total_files = 0
for folder in folders:
    dataframe_path = os.path.join(current_path, folder, "dataframes")
    if os.path.isdir(dataframe_path):
        file_count = len([f for f in os.listdir(dataframe_path) if os.path.isfile(os.path.join(dataframe_path, f))])
        print(f"  Files in {folder}: {file_count}")
        total_files+=file_count

print(f"\nTotal files: {total_files}")