import json
import networkx as nx
import plotly.graph_objects as go
import dash
from dash import Dash, dcc, html
from scipy.spatial import distance




# Load the graph from the JSON file
def load_graph_from_json():
    json_file_path = r"C:\Users\eljapo22\gephi\Node2Edge2JSON\Node2Edge2JSON.json"
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    G = nx.Graph()
    for node in data['nodes']:
        G.add_node(node['id'], pos=(float(node['longitude']), float(node['latitude'])))
    for edge in data['edges']:
        G.add_edge(edge['source'], edge['target'])
    return G

G = load_graph_from_json()

# Substation coordinates
substation_coords = (-79.3858484, 43.6608075)

# Find the nearest node to the substation
def find_nearest_node(G, coords):
    return min(G.nodes(), key=lambda n: distance.euclidean(G.nodes[n]['pos'], coords))

substation_node = find_nearest_node(G, substation_coords)

# Load paths from JSON files
def load_paths(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

bottom_left_paths = load_paths(r"C:\Users\eljapo22\gephi\BottomLeft.json")
top_left_paths = load_paths(r"C:\Users\eljapo22\gephi\TopLeft.json")
middle_paths = load_paths(r"C:\Users\eljapo22\gephi\Middle.json")

# Create the plot
def create_figure():
    fig = go.Figure()

    # Add all grid edges as a single trace
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(color='rgba(0, 0, 0, 0.1)', width=1),
        mode='lines',
        hoverinfo='none',
        showlegend=False
    ))

    # Add all nodes as scatter points
    node_x, node_y, node_color, node_size, node_ids, node_symbol = [], [], [], [], [], []

    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        node_ids.append(str(node))
        
        if node == substation_node:
            node_color.append('red')
            node_size.append(15)
            node_symbol.append('square')
        else:
            node_color.append('rgba(0, 0, 0, 0.1)')
            node_size.append(5)
            node_symbol.append('circle')

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(color=node_color, size=node_size, symbol=node_symbol),
        text=node_ids,
        hoverinfo='text',
        showlegend=False
    ))

    # Function to add paths and nodes from a JSON file
    def add_paths_and_nodes(paths, color):
        path_x, path_y = [], []
        for selection in paths:
            path = selection['path']
            for node in path:
                x, y = G.nodes[node]['pos']
                path_x.append(x)
                path_y.append(y)
            path_x.append(None)
            path_y.append(None)
        
        fig.add_trace(go.Scatter(
            x=path_x, y=path_y,
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False
        ))

        node_x, node_y, node_ids = [], [], []
        for selection in paths:
            node_id = selection['node_id']
            x, y = G.nodes[node_id]['pos']
            node_x.append(x)
            node_y.append(y)
            node_ids.append(str(node_id))
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(color=color, size=10, symbol='triangle-up'),
            text=node_ids,
            hoverinfo='text',
            showlegend=False
        ))

    # Add paths and nodes from each JSON file
    add_paths_and_nodes(bottom_left_paths, 'lightblue')
    add_paths_and_nodes(top_left_paths, 'orange')
    add_paths_and_nodes(middle_paths, 'purple')

    return fig

# Define the layout of the Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Network Map with Path Finding to Substation"),
    dcc.Graph(figure=create_figure(), id='graph', style={'height': '90vh'})
])

if __name__ == '__main__':
    app.run_server(debug=True)