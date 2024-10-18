import json
import networkx as nx
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from tqdm import tqdm
import os

# Function to load JSON data from a file
def load_combined_data(json_file_path):
    with open(json_file_path, 'r') as f:
        return json.load(f)

# Function to create a graph from the combined data
def create_graph_from_combined_data(data):
    G = nx.Graph()
    print("Loading nodes and edges...")
    for node_id, node_data in tqdm(data['nodes'].items(), desc="Loading nodes"):
        G.add_node(node_id, pos=(float(node_data['longitude']), float(node_data['latitude'])), data=node_data)  # Store node attributes
    for edge in tqdm(data['edge_list'], desc="Loading edges"):
        G.add_edge(edge['source'], edge['target'])
    return G

# Function to create a graph from Node2Edge2JSON data (with node attributes)
def load_node2edge_graph(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    G = nx.Graph()
    for node in data['nodes']:
        G.add_node(node['id'], pos=(node['longitude'], node['latitude']), data=node)  # Store full node data
    for edge in data['edges']:
        G.add_edge(edge['source'], edge['target'])
    return G

# Function to filter out edges from Node2Edge2JSON that are also present in the feeder files
def filter_node2edge_graph(node2edge_graph, feeder_graphs):
    # Create a set of edges from all the feeder files to compare against Node2Edge2JSON
    feeder_edges = set()
    
    for G in feeder_graphs.values():
        for edge in G.edges():
            feeder_edges.add((edge[0], edge[1]))
            feeder_edges.add((edge[1], edge[0]))  # Add both directions for undirected edges
    
    # Filter edges in Node2Edge2JSON that are not in feeder_edges
    filtered_edges = []
    for edge in node2edge_graph.edges():
        if (edge[0], edge[1]) not in feeder_edges and (edge[1], edge[0]) not in feeder_edges:
            filtered_edges.append(edge)

    return filtered_edges

# Function to render the shortest path between two sets of nodes with thick dashed black lines
def render_shortest_path(fig, graph, node_list, color='black', line_dash='dash', line_width=5):
    for i in range(len(node_list) - 1):
        start_node = node_list[i]
        end_node = node_list[i + 1]
        if start_node in graph.nodes and end_node in graph.nodes:
            try:
                # Compute the shortest path
                path = nx.shortest_path(graph, source=start_node, target=end_node)
                path_edges = list(zip(path, path[1:]))

                # Render each edge in the path
                for edge in path_edges:
                    x0, y0 = graph.nodes[edge[0]]['pos']
                    x1, y1 = graph.nodes[edge[1]]['pos']
                    fig.add_trace(go.Scattergl(
                        x=[x0, x1], y=[y0, y1], mode='lines',
                        line=dict(width=line_width, color=color, dash=line_dash),  # Thick dashed black line
                        hoverinfo='text',
                        hovertext=f"Shortest Path: {edge[0]} -> {edge[1]}"
                    ))
            except nx.NetworkXNoPath:
                print(f"No path between {start_node} and {end_node}")

# Function to create the figure visualization
def create_figure(filtered_node2edge_edges, graphs, color_mapping, special_nodes, new_special_nodes, node2edge_graph):
    print("Creating visualization (filtered edges, special nodes, and transformers)...")
    fig = go.Figure()

    # Step 1: Render filtered edges from Node2Edge2JSON in grey
    edge_x, edge_y = [], []
    for edge in filtered_node2edge_edges:
        start_node, end_node = edge[0], edge[1]
        x0, y0 = node2edge_graph.nodes[start_node]['pos']
        x1, y1 = node2edge_graph.nodes[end_node]['pos']
        
        if None not in [x0, y0, x1, y1]:
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    fig.add_trace(go.Scattergl(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(width=1, color='grey'),  # Render filtered Node2Edge2JSON edges in grey
        hoverinfo='none'
    ))

    # Step 2: Render the edges from the three feeder files with their original colors
    for file_name, G in graphs.items():
        edge_x, edge_y = [], []
        edge_color = color_mapping[file_name]  # Get the color for each file

        for edge in G.edges():
            start_node, end_node = edge[0], edge[1]
            x0, y0 = G.nodes[start_node]['pos']
            x1, y1 = G.nodes[end_node]['pos']
            
            # Check if the edge connects a Transformer and a Meter
            if G.nodes[start_node]['data']['type'] == 'Transformer' and G.nodes[end_node]['data']['type'] == 'Meter':
                # Dashed line for Transformer to Meter connections
                fig.add_trace(go.Scattergl(
                    x=[x0, x1], y=[y0, y1], mode='lines',
                    line=dict(width=5, color='black', dash='dash'),  # Dashed line style
                    hoverinfo='text',
                    hovertext=f"{start_node} -> {end_node} (Transformer to Meter)"
                ))
            else:
                # Solid line for other connections
                fig.add_trace(go.Scattergl(
                    x=[x0, x1], y=[y0, y1], mode='lines',
                    line=dict(width=4, color=edge_color),  # Solid line style for regular edges
                    hoverinfo='text',
                    hovertext=f"{start_node} -> {end_node}"
                ))

    # Step 3: Render the shortest paths for the two sets of special nodes
    transformer_node = 'N-000023033'  # D/S T/F: EE transformer node
    render_shortest_path(fig, node2edge_graph, [transformer_node] + special_nodes[1:])

    # Render the second set of special nodes with the new path
    new_path_nodes = ['N-000021722', 'N-000021723', 'N-000021717', 'N-000021712']
    render_shortest_path(fig, node2edge_graph, new_path_nodes)

    # Step 4: Render the special nodes from Node2Edge2JSON in purple
    special_node_x, special_node_y = [], []
    
    for node in special_nodes:
        if node in node2edge_graph.nodes:
            x, y = node2edge_graph.nodes[node]['pos']
            special_node_x.append(x)
            special_node_y.append(y)

    # Add a trace for the special nodes
    fig.add_trace(go.Scattergl(
        x=special_node_x, y=special_node_y, mode='markers',
        marker=dict(color='purple', size=10, symbol='circle'),
        hoverinfo='text',
        text=special_nodes  # Display node IDs when hovered
    ))

    # Render the new special nodes in purple
    new_special_node_x, new_special_node_y = [], []
    for node in new_special_nodes:
        if node in node2edge_graph.nodes:
            x, y = node2edge_graph.nodes[node]['pos']
            new_special_node_x.append(x)
            new_special_node_y.append(y)

    fig.add_trace(go.Scattergl(
        x=new_special_node_x, y=new_special_node_y, mode='markers',
        marker=dict(color='purple', size=10, symbol='circle'),
        hoverinfo='text',
        text=new_special_nodes  # Display node IDs when hovered
    ))

    # Step 5: Render important nodes (Substation, Transformer, Intersection, Meter) with specific styles
    node_x, node_y, node_color, node_size, node_symbol, node_text = [], [], [], [], [], []
    for file_name, G in graphs.items():
        edge_color = color_mapping[file_name]  # Use the edge color for transformer nodes
        for node, data in G.nodes(data=True):
            x, y = data['pos']
            node_type = data['data']['type']

            # Customize styles based on node type
            if node_type == 'Substation':
                node_color.append('red')
                node_size.append(20)
                node_symbol.append('square')
                node_text.append(data['data'].get('display', node))
            elif node_type == 'Transformer':
                node_color.append(edge_color)  # Match transformer color to its path
                node_size.append(18)
                node_symbol.append('triangle-up')
                node_text.append(data['data'].get('display', node))
            elif node_type == 'Intersection':
                node_color.append('red')
                node_size.append(10)
                node_symbol.append('circle')
                node_text.append(data['data'].get('display', node))
            elif node_type == 'Meter':
                node_color.append('black')  # Change meter nodes to black
                node_size.append(8)
                node_symbol.append('diamond')
                node_text.append(data['data'].get('display', node))
            else:
                continue  # Skip any other node types

            node_x.append(x)
            node_y.append(y)

    # Add the important nodes to the figure
    fig.add_trace(go.Scattergl(
        x=node_x, y=node_y, mode='markers+text',
        marker=dict(color=node_color, size=node_size, symbol=node_symbol),
        text=node_text,
        textposition="top center",
        hoverinfo='text'
    ))

    fig.update_layout(showlegend=False, hovermode='closest',
                      margin=dict(b=20, l=5, r=5, t=40),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

    return fig


# Main process to load data and create the Dash app
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

# Load the three feeder files
for file_path in file_paths:
    file_name = os.path.basename(file_path)
    combined_data[file_name] = load_combined_data(file_path)
    graphs[file_name] = create_graph_from_combined_data(combined_data[file_name])

# Load Node2Edge2JSON data
node2edge_graph = load_node2edge_graph(r"C:\Users\eljapo22\gephi\Node2Edge2JSON.json")

# Special nodes to highlight in purple
special_nodes = ['N-000023033', 'N-000023029', 'N-000020297', 'N-000021825']

# New set of special nodes for the second path
new_special_nodes = ['N-000021722', 'N-000021723', 'N-000021717', 'N-000021712']

# Filter out edges in Node2Edge2JSON that are also in the feeder files
filtered_node2edge_edges = filter_node2edge_graph(node2edge_graph, graphs)

# Initialize the Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Combined Network Map (With Node Types and Two Shortest Paths)"),
    dcc.Graph(id='graph', style={'height': '80vh'}, figure=create_figure(filtered_node2edge_edges, graphs, color_mapping, special_nodes, new_special_nodes, node2edge_graph)),
])

if __name__ == '__main__':
    print("Starting server...")
    app.run_server(debug=True)
