import json

def load_json_data(json_file_path):
    """Load data from the JSON file."""
    with open(json_file_path, 'r') as file:
        return json.load(file)

def process_paths(data, substation_node, critical_intersections):
    paths = {}  # Dictionary to hold paths and their hierarchies

    for item in data:
        path = item['path']
        path_start = path[0]
        path_end = path[-1]

        if path_start not in paths:
            paths[path_start] = []

        # Find critical intersections along this path
        intersections_in_path = [node for node in path if node in critical_intersections]
        
        # Sort the intersections by their position in the path (closer to the substation = lower level)
        intersections_in_path_sorted = sorted(intersections_in_path, key=lambda node: path.index(node))

        # Add the intersections and their transformers to the path hierarchy
        for level, intersection in enumerate(intersections_in_path_sorted, start=1):
            transformers = get_transformers_for_intersection(path, intersection)
            paths[path_start].append({
                'Level': level,
                'Critical Intersection': intersection,
                'Transformers': transformers
            })

    return paths

def get_transformers_for_intersection(path, intersection):
    """Get transformers connected to the critical intersection."""
    transformers = []
    # Assuming transformers are the first nodes in the path
    for node in path:
        if node != intersection and 'N-' in node:  # Simple assumption transformers have N- prefix
            transformers.append(node)
    return transformers

def print_hierarchy(paths, substation_node):
    print(f"Substation Node: {substation_node}\n")
    
    for path_start, hierarchy in paths.items():
        print(f"Path starting from transformer node {path_start}:\n")
        for entry in hierarchy:
            print(f"  Level {entry['Level']}:")
            print(f"    Critical Intersection: {entry['Critical Intersection']}")
            print(f"    Transformers: {', '.join(entry['Transformers'])}")
        print("\n")  # Extra space between paths for clarity

if __name__ == "__main__":
    # Load the BottomLeft.json file
    data = load_json_data('BottomLeft.json')

    # Define the substation node and critical intersections
    substation_node = 'N-000023385'
    critical_intersections = ['N-000002126', 'N-000001872', 'N-000001480', 'N-000005434', 'N-000017676']

    # Process the paths and create the hierarchy
    paths = process_paths(data, substation_node, critical_intersections)

    # Print the hierarchy by path
    print_hierarchy(paths, substation_node)
