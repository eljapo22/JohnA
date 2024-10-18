import json
import networkx as nx
import os
from collections import Counter

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)






def analyze_file_structure(data):
    print("Top-level keys in the file:")
    for key in data.keys():
        print(f"- {key}")
        if isinstance(data[key], dict):
            print(f"  Type: dict, Length: {len(data[key])}")
            print(f"  Sample of keys: {list(data[key].keys())[:5]}")
        elif isinstance(data[key], list):
            print(f"  Type: list, Length: {len(data[key])}")
            print(f"  Sample of items: {data[key][:5]}")
        else:
            print(f"  Type: {type(data[key])}")









def analyze_bounding_box_contents(data):
    if 'boundingBoxContents' in data:
        bbc = data['boundingBoxContents']
        print(f"Number of bounding boxes: {len(bbc)}")
        print("Sample of boundingBoxContents:")
        for box_id, contents in list(bbc.items())[:1]:  # Look at only the first item
            print(f"  {box_id}:")
            print(f"    Type: {type(contents)}")
            if isinstance(contents, dict):
                print(f"    Number of items: {len(contents)}")
                print(f"    Sample keys: {list(contents.keys())[:5]}")
                sample_values = list(contents.values())[:5]
                print(f"    Sample values: {[type(v) for v in sample_values]}")
                if sample_values and isinstance(sample_values[0], (list, str)):
                    print(f"    First value: {sample_values[0][:100]}")  # Print first 100 chars of first value
            elif isinstance(contents, list):
                print(f"    Number of items: {len(contents)}")
                print(f"    Sample items: {contents[:5]}")
            elif isinstance(contents, str):
                print(f"    Length of string: {len(contents)}")
                print(f"    First 100 characters: {contents[:100]}")
        
        # Rest of the function remains the same

def extract_nodes_edges(data):
    nodes = set()
    edges = set()
    
    if 'nodes' in data:
        if isinstance(data['nodes'], dict):
            nodes = set(data['nodes'].keys())
        elif isinstance(data['nodes'], list):
            nodes = set(node['id'] for node in data['nodes'])
    elif 'boundingBoxes' in data:
        nodes = set(data['boundingBoxes'].keys())
    
    if 'edges' in data:
        for edge in data['edges']:
            edges.add((edge['source'], edge['target']))
    elif 'edge_list' in data:
        for edge in data['edge_list']:
            edges.add((edge['source'], edge['target']))
    elif 'boundingBoxContents' in data:
        for box_contents in data['boundingBoxContents'].values():
            if isinstance(box_contents, dict):
                for edge_id in box_contents.keys():
                    parts = edge_id.strip().split('-')
                    if len(parts) >= 3:
                        edges.add((f"N-{parts[1]}", f"N-{parts[2]}"))
            elif isinstance(box_contents, list):
                for edge_id in box_contents:
                    parts = edge_id.strip().split('-')
                    if len(parts) >= 3:
                        edges.add((f"N-{parts[1]}", f"N-{parts[2]}"))
            elif isinstance(box_contents, str):
                for edge_id in box_contents.split(','):
                    parts = edge_id.strip().split('-')
                    if len(parts) >= 3:
                        edges.add((f"N-{parts[1]}", f"N-{parts[2]}"))
    
    return nodes, edges


# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load all files
node2edge_data = load_json(os.path.join(script_dir, 'Node2Edge2JSON.json'))
combined_data = [
    load_json(os.path.join(script_dir, 'Node2Edge2JSON', 'feeders', 'combined_structure_BottomLeft.json')),
    load_json(os.path.join(script_dir, 'Node2Edge2JSON', 'feeders', 'combined_structure_Middle.json')),
    load_json(os.path.join(script_dir, 'Node2Edge2JSON', 'feeders', 'combined_structure_TopLeft.json'))
]



# Before extracting nodes and edges
print("Analyzing Node2Edge2JSON.json structure:")
analyze_file_structure(node2edge_data)
analyze_bounding_box_contents(node2edge_data)








# Extract nodes and edges
node2edge_nodes, node2edge_edges = extract_nodes_edges(node2edge_data)
combined_nodes, combined_edges = set(), set()
for data in combined_data:
    nodes, edges = extract_nodes_edges(data)
    combined_nodes.update(nodes)
    combined_edges.update(edges)

# Find unique elements in Node2Edge2JSON
unique_nodes = node2edge_nodes - combined_nodes
unique_edges = node2edge_edges - combined_edges

# Analyze special paths
special_nodes = ['N-000023033', 'N-000023029', 'N-000020297', 'N-000021825']
G = nx.Graph()
G.add_nodes_from(node2edge_nodes)
G.add_edges_from(node2edge_edges)

used_unique_nodes = set()
used_unique_edges = set()

for i in range(len(special_nodes) - 1):
    start_node, end_node = special_nodes[i], special_nodes[i + 1]
    if start_node in G.nodes and end_node in G.nodes:
        try:
            path = nx.shortest_path(G, source=start_node, target=end_node)
            for node in path:
                if node in unique_nodes:
                    used_unique_nodes.add(node)
            for edge in zip(path[:-1], path[1:]):
                if edge in unique_edges:
                    used_unique_edges.add(edge)
        except nx.NetworkXNoPath:
            print(f"No path between {start_node} and {end_node}")

print(f"Unique nodes in Node2Edge2JSON: {len(unique_nodes)}")
print(f"Unique nodes used in special paths: {len(used_unique_nodes)}")
print(f"Unique edges in Node2Edge2JSON: {len(unique_edges)}")
print(f"Unique edges used in special paths: {len(used_unique_edges)}")




def deep_inspect_json(data, max_depth=3, current_depth=0):
    if current_depth >= max_depth:
        return str(type(data))
    
    if isinstance(data, dict):
        return {k: deep_inspect_json(v, max_depth, current_depth + 1) for k, v in list(data.items())[:10]}
    elif isinstance(data, list):
        return [deep_inspect_json(item, max_depth, current_depth + 1) for item in data[:10]]
    else:
        return str(type(data))

# Use this function in your main script
print("Deep inspection of Node2Edge2JSON:")
print(json.dumps(deep_inspect_json(node2edge_data), indent=2))


  