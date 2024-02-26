import os
import re
# Define the directory path
directory_path = r"C:\Users\chenw\Downloads\evb\web\Failure Mode Images"

# List to hold file names
file_names = []

# Walk through the directory
for entry in os.listdir(directory_path):
    full_path = os.path.join(directory_path, entry)
    if os.path.isfile(full_path):
        file_names.append(entry)

# Now, file_names list contains all file names in the specified directory
print(file_names)
# Provided mapping
number_mapping = {
    1: 2, 2: 2, 3: 0, 4: 1, 5: 3, 6: 4, 7: 10, 8: 6, 9: 5,
    10: 11, 11: 7, 12: 9, 13: 13, 14: 8, 15: 12
}


def convert_and_rename_files(directory_path, mapping):
    for filename in os.listdir(directory_path):
        original_path = os.path.join(directory_path, filename)
        
        # Check if it's a file
        if os.path.isfile(original_path):
            # Attempt to extract the leading number and rest of the file name
            parts = filename.split(' ', 1)
            new_name = filename  # Default to original filename

            if parts[0].isdigit():
                number = int(parts[0])
                if number in mapping:
                    new_number = mapping[number]
                    if len(parts) > 1:
                        # Remove the specified suffix patterns
                        rest_of_name = re.sub(r'(-01-01|-01)$', '', parts[1])
                        new_name = f"{new_number} {rest_of_name}"
                    else:
                        new_name = str(new_number)
            
            # Construct full path for the new name
            new_path = os.path.join(directory_path, new_name)
            
            # Rename the file
            os.rename(original_path, new_path)
            print(f"Renamed '{original_path}' to '{new_path}'")

convert_and_rename_files(directory_path, number_mapping)