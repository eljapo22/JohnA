import json
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, filename='graph_changes.log', 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Define the cache path
cache_path = "C:/Users/eljapo22/gephi/cache/7644e20afe6c4a94db3dcb733771bb0d4d5a6fc6.json"

# Function to load and parse JSON data
def load_graph_from_custom_cache(filename=cache_path):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Log the structure of the JSON file for inspection
    logger.info(f"Loaded JSON structure: {type(data)}")
    
    if "elements" not in data:
        raise ValueError("The JSON data does not contain the 'elements' key.")
    
    elements = data["elements"]
    logger.info(f"First item: {elements[0]}")
    
    # Initialize a NetworkX MultiDiGraph
    G = nx.MultiDiGraph()
    
    # Add nodes to the graph
    for item in elements:
        if item['type'] == 'node':
            node_id = item['id']
            node_attrs = {
                'lat': item['lat'],
                'lon': item['lon']
            }
            # Include additional attributes if available
            if 'tags' in item:
                node_attrs.update(item['tags'])
            G.add_node(node_id, **node_attrs)
    
    # Add edges to the graph
    for item in elements:
        if item['type'] == 'way':
            nodes = item['nodes']
            for i in range(len(nodes) - 1):
                u = nodes[i]
                v = nodes[i + 1]
                G.add_edge(u, v)
    
    # Assign the CRS attribute
    G.graph['crs'] = {'init': 'epsg:32617'}  # Example CRS for UTM zone 17N
    
    return G

# Load the graph from the cache
try:
    G_proj = load_graph_from_custom_cache()
    logger.info("Loaded graph from cache.")
except (FileNotFoundError, ValueError) as e:
    logger.error(f"Error loading graph from cache: {e}")
    raise

# To convert node attributes to proper format for plotting with OSMnx
# The G_proj graph should have 'x' and 'y' attributes for each node
for node, data in G_proj.nodes(data=True):
    G_proj.nodes[node]['x'] = data['lon']
    G_proj.nodes[node]['y'] = data['lat']

# Visualize the graph using OSMnx
fig, ax = ox.plot_graph(G_proj, show=True, close=False)

# Save the figure if needed
fig.savefig("graph_visualization.png", dpi=300)
plt.show()
