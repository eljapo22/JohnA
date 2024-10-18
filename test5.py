import json
import networkx as nx
import plotly.graph_objects as go

# Load the graph from the cache
def load_graph_from_cache():
    with open("C:\\Users\\eljapo22\\gephi\\cache\\7644e20afe6c4a94db3dcb733771bb0d4d5a6fc6.json", 'r') as f:
        data = json.load(f)
    G = nx.Graph()
    for element in data['elements']:
        if element['type'] == 'node':
            G.add_node(element['id'], pos=(element['lon'], element['lat']))
        elif element['type'] == 'way':
            for i in range(len(element['nodes']) - 1):
                G.add_edge(element['nodes'][i], element['nodes'][i + 1])
    return G

G = load_graph_from_cache()

# Assign the CRS attribute manually
G.graph['crs'] = {'init': 'epsg:4326'}

# Define the nodes
substation = 24959509
transformer_bb = 34404246
transformer_cc = 34404249
node_112 = 344474250
node_48 = 3030997330

# Define the paths
path_1 = nx.shortest_path(G, source=substation, target=transformer_bb)
path_2 = nx.shortest_path(G, source=transformer_bb, target=transformer_cc)
path_3 = nx.shortest_path(G, source=transformer_bb, target=node_112)
path_4 = nx.shortest_path(G, source=transformer_cc, target=node_48)

# Combine all nodes in paths to render
highlighted_nodes = set(path_1 + path_2 + path_3 + path_4)

# Create the plot
fig = go.Figure()

# Add nodes in the highlighted paths to the plot
for node in highlighted_nodes:
    x, y = G.nodes[node]['pos']
    color = 'blue'
    size = 5
    text = None
    if node == substation:
        color = 'red'
        size = 10
        text = 'DS Substation: AA'
    elif node == transformer_bb:
        color = 'green'
        size = 10
        text = 'DS T/F BB'
    elif node == transformer_cc:
        color = 'green'
        size = 10
        text = 'DS T/F CC'
    elif node == node_112:
        color = 'red'
        size = 10
        text = '112'
    elif node == node_48:
        color = 'red'
        size = 10
        text = '48'
    fig.add_trace(go.Scatter(
        x=[x], y=[y],
        mode='markers+text',
        marker=dict(color=color, size=size),
        text=[text] if text else None,
        textposition='top center'
    ))

# Add all grid edges as a single trace with transparency
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
fig.add_trace(go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(color='rgba(0, 0, 255, 0.1)', width=1),
    mode='lines',
    hoverinfo='none',
    showlegend=False
))

# Function to add highlighted paths
def add_path_to_plot(paths, color, width=2):
    edge_x = []
    edge_y = []
    for path in paths:
        for i in range(len(path) - 1):
            x0, y0 = G.nodes[path[i]]['pos']
            x1, y1 = G.nodes[path[i + 1]]['pos']
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(color=color, width=width),
        mode='lines',
        showlegend=False
    ))

# Add paths to the plot
add_path_to_plot([path_1 + path_2[1:]], 'red', 3)  # Main path from substation to transformers
add_path_to_plot([path_3], 'green', 3)             # Path from transformer BB to node 112
add_path_to_plot([path_4], 'green', 3)             # Path from transformer CC to node 48

# Define the legend for highlighted nodes and paths only
legend_items = [
    {'color': 'red', 'name': 'DS Substation: AA'},
    {'color': 'green', 'name': 'DS T/F BB'},
    {'color': 'green', 'name': 'DS T/F CC'},
    {'color': 'red', 'name': '112'},
    {'color': 'red', 'name': '48'},
    {'color': 'red', 'name': 'Substation to TF BB and TF CC Path'},
    {'color': 'green', 'name': 'TF BB to 112 Path'},
    {'color': 'green', 'name': 'TF CC to 48 Path'}
]

for item in legend_items:
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=10, color=item['color']),
        legendgroup=item['name'],
        showlegend=True,
        name=item['name']
    ))

# Save the figure as an HTML file
fig.write_html("network_visualization.html")

# The generated file "network_visualization.html" can be opened in a web browser
