import json

def print_json_structure(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    def print_structure(d, indent=0):
        if isinstance(d, dict):
            for key, value in d.items():
                print('  ' * indent + f"{key}: {type(value).__name__}")
                print_structure(value, indent + 1)
        elif isinstance(d, list):
            print('  ' * indent + f"List of {len(d)} items")
            if len(d) > 0:
                print_structure(d[0], indent + 1)
        else:
            print('  ' * indent + f"{d}: {type(d).__name__}")

    print(f"Structure of {file_path}:")
    print_structure(data)
    print("\n")

# List of file paths
file_paths = [
    r"C:\Users\eljapo22\gephi\Node2Edge2JSON\feeders\combined_structure_BottomLeft.json",
    r"C:\Users\eljapo22\gephi\Node2Edge2JSON\feeders\combined_structure_Middle.json",
    r"C:\Users\eljapo22\gephi\Node2Edge2JSON\feeders\combined_structure_TopLeft.json",
    r"C:\Users\eljapo22\gephi\Node2Edge2JSON\Node2Edge2JSON.json"
]

for file_path in file_paths:
    print_json_structure(file_path)