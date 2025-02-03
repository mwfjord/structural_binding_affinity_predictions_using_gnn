import os
import re


def find_and_rename_cif_files(root_dir):
    """
    Recursively finds .cif files in the given directory, extracts the sequence ID from `_entry.id`
    and renames the file accordingly.
    """
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".cif"):
                file_path = os.path.join(subdir, file)

                # Read the .cif file to find the sequence ID
                try:
                    with open(file_path, "r") as f:
                        for line in f:
                            match = re.match(r"^_entry\.id\s+(\S+)", line)
                            if match:
                                sequence_id = match.group(1)
                                break
                        else:
                            print(f"Skipping {file}: No '_entry.id' found.")
                            continue  # Skip if no sequence ID is found

                    # Generate new file name
                    new_file_name = f"{sequence_id}.cif"
                    new_file_path = os.path.join(subdir, new_file_name)

                    # Rename the file if necessary
                    if file_path != new_file_path:
                        os.rename(file_path, new_file_path)
                        print(f"Renamed: {file} -> {new_file_name}")
                    else:
                        print(f"Skipping {file}: Already named correctly.")

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")


# Set the root directory where .cif files are stored
root_directory = "./"  # Change to your target directory

# Run the renaming function
find_and_rename_cif_files(root_directory)
