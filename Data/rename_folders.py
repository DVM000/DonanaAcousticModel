import os

# Load mapping file and rename folders
def load_mapping_and_rename(mapping_file, audio_folder):
    # Parse the mapping file
    name2_to_name1 = {}
    try:
        with open(mapping_file, "r", encoding="utf-8") as file:
            for line in file:
              
                line = line.strip()
                if "_" in line:
                    parts = line.split("_")
                    if len(parts) == 3:
                        name1, name2, _ = parts
                        name2 = name2.strip()
                        name1 = name1.strip()
                        if name2 and name1:
                            name2_to_name1[name1] = name2
    except Exception as e:
        print(f"Error reading mapping file: {e}")
        return
    
    # Rename folders in the directory
    try:
        for folder in os.listdir(audio_folder):
            current_path = os.path.join(audio_folder, folder)
            if os.path.isdir(current_path):
                new_name = name2_to_name1.get(folder)
                if new_name and new_name != folder:
                    new_path = os.path.join(audio_folder, new_name)
                    try:
                        os.rename(current_path, new_path)
                        print(f"Renamed: {current_path} -> {new_path}")
                    except Exception as e:
                        print(f"Error renaming {current_path} to {new_path}: {e}")
    except Exception as e:
        print(f"Error accessing audio folder: {e}")

# Define paths
mapping_file = "Especies_dos_nombres_corregidov2.txt"
audio_folder = "AUDIOSTFM/val_imgs/"  # or "./xenocanto"
audio_folder = "AUDIOSTFM/WABAD/DATA_WABAD/DATA/"  


# Call the function
load_mapping_and_rename(mapping_file, audio_folder)

