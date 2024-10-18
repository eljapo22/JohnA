import pickle
import gzip
import pprint

def load_pickle_file(file_path):
    """Load and return the contents of a pickle file."""
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)

def analyze_data_structure(data, depth=0, max_depth=3):
    """Recursively analyze and print the structure of the data."""
    if depth > max_depth:
        return
    
    if isinstance(data, dict):
        print(f"{'  ' * depth}Dictionary with {len(data)} keys:")
        for key, value in data.items():
            print(f"{'  ' * (depth + 1)}Key: {key} -> Type: {type(value).__name__}")
            analyze_data_structure(value, depth + 1, max_depth)
    elif isinstance(data, list):
        print(f"{'  ' * depth}List with {len(data)} elements:")
        if data:
            analyze_data_structure(data[0], depth + 1, max_depth)
    else:
        print(f"{'  ' * depth}Type: {type(data).__name__} -> Value: {pprint.pformat(data, width=60)}")

def main():
    file_path = 'node2edge_cache.gz'  # Replace with your actual file path
    try:
        data = load_pickle_file(file_path)
        print("Data loaded successfully. Analyzing structure...")
        analyze_data_structure(data)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()