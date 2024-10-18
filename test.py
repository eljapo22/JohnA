import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_daq as daq
import json
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, filename='graph_changes.log', 
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger()

# Define the bounding box coordinates (north, south, east, west)
bbox = (43.6708, 43.6413, -79.3520, -79.3936)

# Function to save graph to cache
def save_graph_to_cache(graph, filename='graph_cache.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(graph, f)

# Function to load graph from cache
def load_graph_from_cache(filename='graph_cache.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Function to save transformer colors to cache
def save_transformer_colors(transformer_colors, filename='transformer_colors.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(transformer_colors, f)

# Function to load transformer colors from cache
def load_transformer_colors(filename='transformer_colors.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Check if cached graph exists
try:
    G_proj = load_graph_from_cache()
    transformer_colors = load_transformer_colors()
    print("Loaded graph and transformer colors from cache.")
except FileNotFoundError:
    # Create the graph using the bounding box
    G = ox.graph_from_bbox(*bbox, network_type='drive')
    # Project the graph to UTM (for accurate distance calculations)
    G_proj = ox.project_graph(G)
    transformer_colors = {}
    save_graph_to_cache(G_proj)
    save_transformer_colors(transformer_colors)
    print("Saved graph and transformer colors to cache.")

# Extract node coordinates and degree as a proxy for load demand
nodes = list(G_proj.nodes(data=True))
node_coords = np.array([(data['x'], data['y'], G_proj.degree(node)) for node, data in nodes])

# Use node degree as a proxy for demand, giving higher weights to high-degree nodes
threshold = np.percentile(node_coords[:, 2], 75)  # Use the 75th percentile as the threshold
high_demand_nodes = node_coords[node_coords[:, 2] >= threshold]

# Apply weighted K-Means clustering to identify high-demand areas
num_clusters = 10  # Adjust based on the size and density of your graph
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(high_demand_nodes[:, :2], sample_weight=high_demand_nodes[:, 2])
cluster_centers = kmeans.cluster_centers_

# Identify nodes closest to cluster centers for ground-mounted transformers
ground_transformer_nodes = [ox.distance.nearest_nodes(G_proj, coord[0], coord[1]) for coord in cluster_centers]

# Evenly distribute pole-mounted transformers in residential areas
def distribute_pole_transformers_evenly(graph, target_num):
    intersections = [node for node, degree in dict(graph.degree()).items() if degree > 2]
    if len(intersections) < target_num:
        target_num = len(intersections)
    evenly_spaced_nodes = np.linspace(0, len(intersections)-1, target_num, dtype=int)
    return [intersections[i] for i in evenly_spaced_nodes]

num_pole_transformers = 10  # Adjust based on the size and density of your graph
pole_transformer_nodes = distribute_pole_transformers_evenly(G_proj, num_pole_transformers)

# Mark these nodes in the graph with attributes
for node in ground_transformer_nodes:
    G_proj.nodes[node]['type'] = 'ground_transformer'
for node in pole_transformer_nodes:
    G_proj.nodes[node]['type'] = 'pole_transformer'

# Identify all transformer nodes
all_transformer_nodes = ground_transformer_nodes + pole_transformer_nodes

# Ensure the substation node is correctly placed
substation_coords = (630000, 4835000)  # Example coordinates
substation_node = ox.distance.nearest_nodes(G_proj, substation_coords[0], substation_coords[1])
G_proj.nodes[substation_node]['type'] = 'substation'
G_proj.nodes[substation_node]['color'] = 'red'

if substation_node in all_transformer_nodes:
    all_transformer_nodes.remove(substation_node)
else:
    logger.warning(f"Substation node {substation_node} is not in the list of transformer nodes.")

# Function to clean transformer_colors dictionary
def clean_transformer_colors(graph, transformer_colors):
    graph_nodes = set(graph.nodes())
    keys_to_delete = [key for key in transformer_colors.keys() if key not in graph_nodes]
    for key in keys_to_delete:
        del transformer_colors[key]
    return transformer_colors

# Clean the transformer colors dictionary
transformer_colors = clean_transformer_colors(G_proj, transformer_colors)

# Generate distinct colors for each transformer
def get_distinct_colors(num_colors):
    cmap = plt.get_cmap('tab20b')
    colors = cmap(np.linspace(0, 1, num_colors))
    return [f'rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})' for color in colors]

num_transformers = len(all_transformer_nodes)
colors = get_distinct_colors(num_transformers)

# Map transformers to colors
transformer_colors.update({node: colors[i] for i, node in enumerate(all_transformer_nodes)})

# Save the updated transformer colors
save_transformer_colors(transformer_colors)

# Generate labels for transformers
def generate_labels(num_transformers):
    labels = []
    for i in range(num_transformers):
        first = chr(65 + (i // 26))
        second = chr(65 + (i % 26))
        labels.append(f'DS T/F: {first}{second}')
    return labels

# Generate labels for transformers
labels = generate_labels(num_transformers)

# Function to find the nearest transformer and compute the shortest path
def compute_shortest_paths_to_transformers(graph, transformer_nodes):
    paths = {}
    for node in graph.nodes():
        if 'type' not in graph.nodes[node]:  # Blue nodes (not transformers)
            reachable_transformers = [t for t in transformer_nodes if nx.has_path(graph, node, t)]
            if reachable_transformers:
                closest_transformer = min(reachable_transformers, key=lambda t: nx.shortest_path_length(graph, node, t, weight='length'))
                # Compute the shortest path
                path = nx.shortest_path(graph, source=node, target=closest_transformer, weight='length')
                paths[node] = (path, transformer_colors[closest_transformer])
    return paths

# Compute shortest paths from all blue nodes to the nearest transformer
shortest_paths = compute_shortest_paths_to_transformers(G_proj, all_transformer_nodes)

# Initialize the figure
fig = make_subplots()

# Add edges
edge_x = []
edge_y = []
for u, v, data in G_proj.edges(data=True):
    x0, y0 = G_proj.nodes[u]['x'], G_proj.nodes[v]['y']
    x1, y1 = G_proj.nodes[v]['x'], G_proj.nodes[v]['y']
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                         line=dict(width=0.5, color='rgba(0,0,255,0.2)'),  # Make non-highlighted edges partially transparent
                         hoverinfo='none',
                         mode='lines'))

# Add nodes
node_x = []
node_y = []
node_color = []
node_text = []

for node in G_proj.nodes():
    x, y = G_proj.nodes[node]['x'], G_proj.nodes[node]['y']
    node_x.append(x)
    node_y.append(y)
    if G_proj.nodes[node].get('type') == 'ground_transformer':
        color = transformer_colors[node]
        node_color.append(color)
        node_text.append(f"Ground Transformer: {node}")
    elif G_proj.nodes[node].get('type') == 'pole_transformer':
        color = transformer_colors[node]
        node_color.append(color)
        node_text.append(f"Pole Transformer: {node}")
    elif G_proj.nodes[node].get('type') == 'substation':
        node_color.append('red')
        node_text.append(f"Substation: {node}")
    else:
        node_color.append('blue')
        node_text.append(f"Node: {node}")

fig.add_trace(go.Scatter(x=node_x, y=node_y,
                         mode='markers',
                         marker=dict(size=8, color=node_color),
                         text=node_text,
                         hoverinfo='text'))

# Highlight paths to transformers
for node, (path, color) in shortest_paths.items():
    path_x = [G_proj.nodes[node]['x'] for node in path]
    path_y = [G_proj.nodes[node]['y'] for node in path]
    fig.add_trace(go.Scatter(x=path_x, y=path_y,
                             line=dict(width=3, color=color),  # Thicker lines for highlighted paths
                             mode='lines',
                             hoverinfo='none',
                             legendgroup=labels[list(transformer_colors.values()).index(color)],
                             showlegend=False))

# Add legend
for i, (node, label) in enumerate(zip(all_transformer_nodes, labels)):
    fig.add_trace(go.Scatter(x=[G_proj.nodes[node]['x']], y=[G_proj.nodes[node]['y']],
                             mode='markers',
                             marker=dict(size=15, color=transformer_colors[node]),  # Larger marker size for transformers
                             legendgroup=label,
                             showlegend=True,
                             name=label))

fig.update_layout(
    showlegend=True,
    legend=dict(
        x=1,
        y=1,
        traceorder='normal',
        font=dict(
            family='sans-serif',
            size=12,
            color='black'
        ),
        bgcolor='LightSteelBlue',
        bordercolor='Black',
        borderwidth=2
    ),
    plot_bgcolor='white',
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False),
    autosize=True,
    width=1000,
    height=800
)

# Create the Dash app
app = dash.Dash(__name__)
server = app.server

# Define app layout
app.layout = html.Div([
    dcc.Graph(
        id='network-graph',
        figure=fig,
        config={'modeBarButtonsToAdd': ['drawcircle', 'drawopenpath', 'drawclosedpath']}
    ),
    html.Label('Node Label:'),
    dcc.Input(id='node-label', value='Substation', type='text'),
    html.Button('Save Nodes', id='save-button', n_clicks=0),
    html.Button('Delete Nodes', id='delete-button', n_clicks=0),
    dcc.Store(id='node-data', data=[]),
    html.Div(id='output-div'),
    html.Div(id='delete-output-div')
])

# Callback to capture click data
@app.callback(
    Output('node-data', 'data'),
    [Input('network-graph', 'relayoutData')],
    [State('node-data', 'data')]
)
def update_node_data(relayoutData, current_data):
    if relayoutData is not None and 'shapes' in relayoutData:
        new_data = relayoutData['shapes']
        return current_data + new_data
    return current_data

# Callback to save node data to a file and add new nodes to the graph
@app.callback(
    Output('output-div', 'children'),
    [Input('save-button', 'n_clicks')],
    [State('node-data', 'data'), State('node-label', 'value')]
)
def save_node_data(n_clicks, node_data, node_label):
    if n_clicks > 0:
        # Save node data to file
        with open('added_nodes.json', 'w') as f:
            json.dump(node_data, f)

        # Add new nodes to the graph
        for shape in node_data:
            if shape['type'] == 'circle':
                x = shape['x0']
                y = shape['y0']
                new_node_id = max(G_proj.nodes) + 1
                G_proj.add_node(new_node_id, x=x, y=y, type=node_label)
                logger.info(f"Added node {new_node_id}: Coordinates ({x}, {y}), Label {node_label}")
                # Optionally, connect the new node to the nearest existing node
                nearest_node = ox.distance.nearest_nodes(G_proj, x, y)
                G_proj.add_edge(new_node_id, nearest_node)

        # Save updated graph to cache
        save_graph_to_cache(G_proj)
        return html.Div('Nodes have been saved and added to the graph.')
    return html.Div()

# Callback to delete node data
@app.callback(
    Output('delete-output-div', 'children'),
    [Input('delete-button', 'n_clicks')],
    [State('node-data', 'data')]
)
def delete_node_data(n_clicks, node_data):
    if n_clicks > 0:
        nodes_to_delete = []
        target_x, target_y = None, None
        for shape in node_data:
            if shape['type'] == 'circle':
                target_x = shape['x0']
                target_y = shape['y0']
                for node, data in G_proj.nodes(data=True):
                    node_x = data['x']
                    node_y = data['y']
                    if np.isclose(node_x, target_x, atol=1e-2) and np.isclose(node_y, target_y, atol=1e-2):
                        nodes_to_delete.append(node)
                        if node in transformer_colors:
                            del transformer_colors[node]
        
        if nodes_to_delete:
            for node in nodes_to_delete:
                node_data = G_proj.nodes[node]
                logger.info(f"Deleted node {node}: Coordinates ({node_data['x']}, {node_data['y']}), Label {node_data.get('type', 'N/A')}")
            G_proj.remove_nodes_from(nodes_to_delete)

            # Save updated graph and transformer colors to cache
            save_graph_to_cache(G_proj)
            save_transformer_colors(transformer_colors)
            return html.Div(f'Selected node(s) {nodes_to_delete} have been deleted from the graph.')
        else:
            return html.Div('No matching nodes found to delete.')
    return html.Div()

if __name__ == '__main__':
    app.run_server(debug=True)
