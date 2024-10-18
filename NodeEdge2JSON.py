import pickle
import json
import networkx as nx

def create_json_from_pkl(pkl_path, json_path):
    # Load the PKL file
    with open(pkl_path, 'rb') as f:
        G = pickle.load(f)

    # Calculate bounding box
    min_lon = min(node[0] for node in G.nodes())
    max_lon = max(node[0] for node in G.nodes())
    min_lat = min(node[1] for node in G.nodes())
    max_lat = max(node[1] for node in G.nodes())

    # Create bounding box
    bounding_box = {
        "TOR-001": {
            "southwest": [min_lon, min_lat],
            "northeast": [max_lon, max_lat]
        }
    }

    # Prepare data structures
    nodes = []
    edges = []
    bounding_box_contents = {"TOR-001": {"nodes": [], "edges": []}}

    # Process nodes
    for i, (coord, data) in enumerate(G.nodes(data=True), 1):
        node_id = f"N-{i:09d}"
        nodes.append({
            "id": node_id,
            "longitude": coord[0],
            "latitude": coord[1],
            "type": data.get('type', 'normal'),
            "boundingBoxes": ["TOR-001"]
        })
        bounding_box_contents["TOR-001"]["nodes"].append(node_id)

    # Create a mapping of coordinates to node IDs
    coord_to_id = {(n['longitude'], n['latitude']): n['id'] for n in nodes}

    # Process edges
    for i, (u, v, data) in enumerate(G.edges(data=True), 1):
        edge_id = f"E-{i:09d}"
        edges.append({
            "id": edge_id,
            "source": coord_to_id[u],
            "target": coord_to_id[v],
            "boundingBoxes": ["TOR-001"]
        })
        bounding_box_contents["TOR-001"]["edges"].append(edge_id)

    # Create the final JSON structure
    graph_data = {
        "boundingBoxes": bounding_box,
        "nodes": nodes,
        "edges": edges,
        "boundingBoxContents": bounding_box_contents
    }

    # Write to JSON file
    with open(json_path, 'w') as f:
        json.dump(graph_data, f, indent=2)

    print(f"JSON file created at {json_path}")
    print(f"Total nodes: {len(nodes)}")
    print(f"Total edges: {len(edges)}")

# Usage
pkl_file = r"C:\Users\eljapo22\gephi\network_graph.pkl"
json_file = r"C:\Users\eljapo22\gephi\network_graph.json"
create_json_from_pkl(pkl_file, json_file)