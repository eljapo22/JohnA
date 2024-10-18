import json
import networkx as nx
import plotly.graph_objects as go
import dash
from dash import Dash, dcc, html, Input, Output, State
from scipy.spatial import distance
import os

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

# Save paths to JSON file
def save_paths(file_path, paths):
    with open(file_path, 'w') as f:
        json.dump(paths, f, indent=4)

# Color mapping for clusters
color_mapping = {
    'BottomLeft.json': 'lightblue',
    'TopLeft.json': 'orange',
    'Middle.json': 'purple',
    'Unique Path': 'lime'
}

# Create the plot
def create_figure(cluster_paths, unique_path_nodes, unique_path):
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
        elif node in unique_path_nodes:
            node_color.append('red')
            node_size.append(10)
            node_symbol.append('circle')
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

    # Add paths and nodes from each cluster
    for file_name, paths in cluster_paths.items():
        add_paths_and_nodes(paths, color_mapping[file_name])

    # Add unique path
    if unique_path:
        path_x, path_y = [], []
        for node in unique_path:
            x, y = G.nodes[node]['pos']
            path_x.append(x)
            path_y.append(y)
        
        fig.add_trace(go.Scatter(
            x=path_x, y=path_y,
            mode='lines',
            line=dict(color='lime', width=3),
            showlegend=False
        ))

    return fig

# Define the layout of the Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Network Map with Path Finding to Substation"),
    dcc.Graph(id='graph', style={'height': '80vh'}),
    html.Div([
        dcc.Dropdown(
            id='cluster-dropdown',
            options=[{'label': f, 'value': f} for f in color_mapping.keys()],
            value=list(color_mapping.keys())[0],
            style={'width': '200px'}
        ),
        dcc.Input(id='node-input', type='text', placeholder='Enter node ID'),
        html.Button('Add Node', id='add-button'),
        html.Button('Delete Node', id='delete-button'),
        html.Button('Complete Unique Path', id='complete-unique-path-button'),
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'width': '800px', 'margin': '20px 0'}),
    html.Div([
        dcc.Textarea(
            id='node-list-input',
            placeholder='Enter comma-separated list of node IDs for unique path',
            style={'width': '100%', 'height': 100}
        ),
        html.Button('Process Node List', id='process-node-list-button'),
    ], style={'width': '800px', 'margin': '20px 0'}),
    html.Div(id='unique-path-nodes', children='Unique Path Nodes: '),
    html.Div(id='output-message')
])

# Load initial cluster paths
cluster_paths = {f: load_paths(f) for f in color_mapping.keys() if f != 'Unique Path'}
unique_path_nodes = []
unique_path = []

@app.callback(
    Output('node-input', 'value'),
    [Input('graph', 'clickData')]
)
def update_node_input(clickData):
    if clickData:
        return clickData['points'][0]['text']
    return ''

@app.callback(
    [Output('graph', 'figure'),
     Output('output-message', 'children'),
     Output('unique-path-nodes', 'children')],
    [Input('add-button', 'n_clicks'),
     Input('delete-button', 'n_clicks'),
     Input('complete-unique-path-button', 'n_clicks'),
     Input('process-node-list-button', 'n_clicks')],
    [State('cluster-dropdown', 'value'),
     State('node-input', 'value'),
     State('node-list-input', 'value'),
     State('graph', 'figure')]
)
def update_graph(add_clicks, delete_clicks, complete_unique_path_clicks, process_node_list_clicks,
                 selected_cluster, node_id, node_list, current_fig):
    global unique_path_nodes, unique_path
    ctx = dash.callback_context
    if not ctx.triggered:
        return create_figure(cluster_paths, unique_path_nodes, unique_path), "", f"Unique Path Nodes: {', '.join(map(str, unique_path_nodes))}"

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'process-node-list-button':
        if not node_list:
            return current_fig, "Please enter a list of node IDs.", f"Unique Path Nodes: {', '.join(map(str, unique_path_nodes))}"
        
        new_nodes = [node.strip() for node in node_list.split(',')]
        valid_nodes = [node for node in new_nodes if node in G.nodes()]
        invalid_nodes = set(new_nodes) - set(valid_nodes)
        
        unique_path_nodes = valid_nodes
        unique_path = []  # Reset the unique path
        
        if invalid_nodes:
            message = f"Invalid nodes ignored: {', '.join(invalid_nodes)}. Valid nodes added to Unique Path."
        else:
            message = "All nodes successfully added to Unique Path."
        
        return create_figure(cluster_paths, unique_path_nodes, unique_path), message, f"Unique Path Nodes: {', '.join(map(str, unique_path_nodes))}"
    
    if not node_id and button_id != 'complete-unique-path-button':
        return current_fig, "Please enter a node ID.", f"Unique Path Nodes: {', '.join(map(str, unique_path_nodes))}"

    if node_id not in G.nodes() and button_id != 'complete-unique-path-button':
        return current_fig, f"Node {node_id} does not exist in the graph.", f"Unique Path Nodes: {', '.join(map(str, unique_path_nodes))}"

    if selected_cluster == 'Unique Path':
        if button_id == 'add-button':
            if node_id not in unique_path_nodes:
                unique_path_nodes.append(node_id)
                message = f"Node {node_id} added to Unique Path"
            else:
                message = f"Node {node_id} is already in the Unique Path"
        elif button_id == 'delete-button':
            if node_id in unique_path_nodes:
                unique_path_nodes.remove(node_id)
                message = f"Node {node_id} removed from Unique Path"
            else:
                message = f"Node {node_id} is not in the Unique Path"
        elif button_id == 'complete-unique-path-button':
            if len(unique_path_nodes) < 2:
                return current_fig, "Please select at least two nodes for the Unique Path.", f"Unique Path Nodes: {', '.join(map(str, unique_path_nodes))}"
            unique_path = optimize_path(unique_path_nodes)
            message = "Unique Path completed"
        
        return create_figure(cluster_paths, unique_path_nodes, unique_path), message, f"Unique Path Nodes: {', '.join(map(str, unique_path_nodes))}"
    else:
        paths = cluster_paths[selected_cluster]

        if button_id == 'add-button':
            if any(selection['node_id'] == node_id for selection in paths):
                return current_fig, f"Node {node_id} is already in the cluster.", f"Unique Path Nodes: {', '.join(map(str, unique_path_nodes))}"
            
            try:
                new_path = nx.shortest_path(G, source=node_id, target=substation_node)
                paths.append({
                    'node_id': node_id,
                    'coordinates': G.nodes[node_id]['pos'],
                    'path': new_path
                })
                save_paths(selected_cluster, paths)
                message = f"Node {node_id} added to {selected_cluster}"
            except nx.NetworkXNoPath:
                return current_fig, f"No path found from node {node_id} to the substation.", f"Unique Path Nodes: {', '.join(map(str, unique_path_nodes))}"

        elif button_id == 'delete-button':
            paths = [selection for selection in paths if selection['node_id'] != node_id]
            save_paths(selected_cluster, paths)
            message = f"Node {node_id} removed from {selected_cluster}"

        cluster_paths[selected_cluster] = paths
        return create_figure(cluster_paths, unique_path_nodes, unique_path), message, f"Unique Path Nodes: {', '.join(map(str, unique_path_nodes))}"

def optimize_path(nodes):
    if len(nodes) < 2:
        return []
    
    result_path = []
    for i in range(len(nodes) - 1):
        try:
            path = nx.shortest_path(G, nodes[i], nodes[i+1])
            result_path.extend(path[:-1])  # Don't add the last node to avoid duplication
        except nx.NetworkXNoPath:
            return []  # Return empty path if no connection found
    
    # Add the last node
    result_path.append(nodes[-1])
    
    # Add path to substation from the last node
    try:
        final_path = nx.shortest_path(G, nodes[-1], substation_node)
        result_path.extend(final_path[1:])  # Skip the first node as it's already in result_path
    except nx.NetworkXNoPath:
        return []  # Return empty path if no connection to substation found
    
    return result_path

if __name__ == '__main__':
    app.run_server(debug=True)