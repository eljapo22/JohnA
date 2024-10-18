import json
import networkx as nx
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from collections import defaultdict
from tqdm import tqdm
import os

def load_combined_data(json_file_path):
    with open(json_file_path, 'r') as f:
        return json.load(f)

def save_combined_data(json_file_path, data):
    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=4)

def create_graph_from_combined_data(data):
    G = nx.Graph()
    print("Loading nodes and edges...")
    for node_id, node_data in tqdm(data['nodes'].items(), desc="Loading nodes"):
        G.add_node(node_id, pos=(float(node_data['longitude']), float(node_data['latitude'])), 
                   shape=node_data.get('shape', 'circle'), type=node_data['type'],
                   display=node_data.get('display', node_id))
    for edge in tqdm(data['edge_list'], desc="Loading edges"):
        G.add_edge(edge['source'], edge['target'], level=edge.get('level'), path_type=edge.get('path_type'))
    return G

def find_secondary_paths(data):
    secondary_paths = []
    for edge in data['edge_list']:
        if edge.get('path_type') == 'secondary':
            secondary_paths.append((edge['source'], edge['target']))
    print(f"Found {len(secondary_paths)} secondary paths")  # Debug statement
    return secondary_paths

def find_transformer_node(G):
    for node_id, node_data in G.nodes(data=True):
        if node_data.get('type') == 'Transformer':  # Ensure the type matches exactly
            return node_id
    raise ValueError("Transformer node not found")

def bfs_mark_nodes_to_intersection(G, start_node):
    visited = set()
    queue = [(start_node, 0)]
    bfs_list = []

    while queue:
        current_node, current_level = queue.pop(0)
        if current_node not in visited:
            visited.add(current_node)
            node_type = G.nodes[current_node].get('type', 'Unknown')
            bfs_list.append((current_node, current_level, node_type))
            if node_type == 'Intersection':
                break
            for neighbor in G.neighbors(current_node):
                if neighbor not in visited:
                    edge_data = G.get_edge_data(current_node, neighbor)
                    level = edge_data.get('level', current_level + 1)
                    queue.append((neighbor, level))
    
    return bfs_list

def update_json_with_bfs_and_level(data, bfs_results):
    # Find the level of the transformer node
    transformer_level = None
    for node_id, level, node_type in bfs_results:
        if node_type == 'Transformer':
            transformer_level = level
            break

    for node_id, level, node_type in bfs_results:
        if node_type != 'Intersection':
            if node_id in data['nodes']:
                data['nodes'][node_id]['first_breath_search'] = True
                data['nodes'][node_id]['level'] = transformer_level if node_type == 'Transformer' else level

def create_figure(graphs, secondary_paths):
    print("Creating visualization...")
    fig = go.Figure()

    for file_name, G in graphs.items():
        for edge in secondary_paths[file_name]:
            start_node, end_node = edge
            x0, y0 = G.nodes[start_node]['pos']
            x1, y1 = G.nodes[end_node]['pos']
            fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines',
                                     line=dict(width=5, color='blue'),
                                     hoverinfo='text',
                                     hovertext=f"File: {file_name}, Start: {start_node}, End: {end_node}"))

    fig.update_layout(showlegend=False, hovermode='closest',
                      margin=dict(b=20, l=5, r=5, t=40),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

    return fig

print("Starting main process...")
file_paths = [
    r"C:\Users\eljapo22\gephi\Node2Edge2JSON\feeders\combined_structure_BottomLeft.json",
    r"C:\Users\eljapo22\gephi\Node2Edge2JSON\feeders\combined_structure_Middle.json",
    r"C:\Users\eljapo22\gephi\Node2Edge2JSON\feeders\combined_structure_TopLeft.json"
]

color_mapping = {
    'combined_structure_BottomLeft.json': 'lightblue',
    'combined_structure_TopLeft.json': 'orange',
    'combined_structure_Middle.json': 'purple',
}

graphs = {}
combined_data = {}
secondary_paths = {}
bfs_results = {}

for file_path in file_paths:
    file_name = os.path.basename(file_path)
    combined_data[file_name] = load_combined_data(file_path)
    G = create_graph_from_combined_data(combined_data[file_name])
    graphs[file_name] = G
    secondary_paths[file_name] = find_secondary_paths(combined_data[file_name])
    
    # Automatically find the transformer node ID
    transformer_node_id = find_transformer_node(G)
    bfs_results[file_name] = bfs_mark_nodes_to_intersection(G, transformer_node_id)

    # Update JSON data with BFS and level information
    update_json_with_bfs_and_level(combined_data[file_name], bfs_results[file_name])
    
    # Save the updated JSON data back to the file
    save_combined_data(file_path, combined_data[file_name])

    # Print BFS results
    print(f"BFS results for {file_name}: {bfs_results[file_name]}")

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Secondary Paths Network Map"),
    dcc.Graph(id='graph', style={'height': '80vh'}, figure=create_figure(graphs, secondary_paths)),
])

if __name__ == '__main__':
    print("Starting server...")
    app.run_server(debug=True)