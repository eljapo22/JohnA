import os
import shutil

def get_all_files_with_paths(file_list_path):
    with open(file_list_path, 'r') as file:
        return [line.strip() for line in file]

def remove_duplicates_from_text_file(file_list_path, dest_dir):
    file_paths_dict = {}
    
    # Read the file paths from the text file
    file_paths = get_all_files_with_paths(file_list_path)
    
    # Organize files by their relative paths
    for file_path in file_paths:
        relative_path = os.path.relpath(file_path, start=dest_dir)
        if relative_path not in file_paths_dict:
            file_paths_dict[relative_path] = []
        file_paths_dict[relative_path].append(file_path)
    
    # Remove duplicates and keep the first encountered file
    for relative_path, paths in file_paths_dict.items():
        dest_file_path = os.path.join(dest_dir, relative_path)
        if len(paths) > 1:
            # Remove all duplicate files
            for path in paths[1:]:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"Deleted duplicate file: {path}")
        # Ensure the remaining file is in the correct location
        if not os.path.exists(dest_file_path):
            shutil.copyfile(paths[0], dest_file_path)
    
    # Remove empty directories
    for root, dirs, files in os.walk(dest_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
                print(f"Deleted empty directory: {dir_path}")

# Paths
file_list_path = 'file_list.txt'
dest_dir = 'C:\\Users\\eljapo22\\gephi\\network_tool_project\\staticfiles'

# Remove duplicates
remove_duplicates_from_text_file(file_list_path, dest_dir)
