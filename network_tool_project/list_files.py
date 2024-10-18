import os

# Define the path to the staticfiles directory
staticfiles_dir = r'C:\Users\eljapo22\gephi\network_tool_project\staticfiles'

# List to store all file paths
all_files = []

# Traverse the staticfiles directory
for root, dirs, files in os.walk(staticfiles_dir):
    for file in files:
        file_path = os.path.join(root, file)
        all_files.append(file_path)

# Write the file list to a text file
with open(r'C:\Users\eljapo22\gephi\network_tool_project\allfiles_staticfiles.txt', 'w') as f:
    for file_path in all_files:
        f.write(file_path + '\n')

print("File list has been generated and saved to allfiles_staticfiles.txt.")
