import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Define the bounding box coordinates (north, south, east, west)
bbox = (43.6708, 43.6413, -79.3520, -79.3936)

# Check if cached graph exists
try:
    with open('graph_cache.pkl', 'rb') as f:
        G_proj = pickle.load(f)
    print("Loaded graph from cache.")
except FileNotFoundError:
    # Create the graph using the bounding box
    G = ox.graph_from_bbox(*bbox, network_type='drive')
    # Project the graph to UTM (for accurate distance calculations)
    G_proj = ox.project_graph(G)
    with open('graph_cache.pkl', 'wb') as f:
        pickle.dump(G_proj, f)
    print("Saved graph to cache.")

# Ensure the substation node is correctly placed
substation_coords = (630000, 4835000)  # Example coordinates
substation_node = ox.distance.nearest_nodes(G_proj, substation_coords[0], substation_coords[1])
G_proj.nodes[substation_node]['type'] = 'substation'

# Define transformer nodes (example nodes)
transformer_nodes = [node for node, data in G_proj.nodes(data=True) if data.get('type') in ['ground_transformer', 'pole_transformer']]

# Add transformer attributes
for node in transformer_nodes:
    G_proj.nodes[node]['type'] = 'transformer'

# Convert directed graph to undirected graph
G_undirected = G_proj.to_undirected()

# Compute Minimum Spanning Tree (MST)
mst = nx.minimum_spanning_tree(G_undirected, weight='length')

# Function to find the optimal loop by connecting endpoints of MST
def find_optimal_loop(mst, substation_node):
    edges = list(mst.edges())
    nodes = set(mst.nodes())
    endpoints = [node for node in nodes if len(list(mst.neighbors(node))) == 1]
    
    # Add an edge to form a loop
    if endpoints:
        mst.add_edge(endpoints[0], endpoints[1])
    return mst

optimal_mst_loop = find_optimal_loop(mst, substation_node)

# Visualize the Directed, Undirected Graphs, and the MST with the optimal loop
def plot_graphs_with_mst(G_dir, G_undir, mst, substation_node, transformer_nodes):
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    
    for ax, G, title in zip(axs, [G_dir, G_undir, mst], ['Directed Graph', 'Undirected Graph', 'MST with Optimal Loop']):
        pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}
        nx.draw(G, pos, ax=ax, node_size=10, node_color='blue', edge_color='gray')
        nx.draw_networkx_nodes(G, pos, nodelist=[substation_node], node_size=50, node_color='red', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=transformer_nodes, node_size=50, node_color='green', ax=ax)
        if title == 'MST with Optimal Loop':
            nx.draw(G, pos, ax=ax, node_size=10, node_color='blue', edge_color='black', width=2)
        ax.set_title(title)
    
    plt.show()

plot_graphs_with_mst(G_proj, G_undirected, optimal_mst_loop, substation_node, transformer_nodes)
