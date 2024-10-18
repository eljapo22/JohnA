import re

def get_last_newly_created_node(log_file='graph_changes.log'):
    # Pattern to match lines indicating added nodes
    pattern = re.compile(r"Added node (\d+): Coordinates \(([\d\.,-]+), ([\d\.,-]+)\), Label (.+)")

    last_node = None

    # Open the log file and read line by line
    with open(log_file, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                node_id = match.group(1)
                coordinates = (float(match.group(2)), float(match.group(3)))
                label = match.group(4)
                last_node = (node_id, coordinates, label)

    if last_node:
        node_id, coords, label = last_node
        print(f"Last newly created node:")
        print(f"Node ID: {node_id}, Coordinates: {coords}, Label: {label}")
    else:
        print("No newly created nodes found.")

# Run the script to get the last newly created node
get_last_newly_created_node()
