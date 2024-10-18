import json
import networkx as nx
import plotly.graph_objects as go
import dash
from dash import Dash, dcc, html, Input, Output, State
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
        line=dict(color='rgba(0, 255, 0, 0.5)', width=1),
        mode='lines',
        hoverinfo='none',
        showlegend=False
    ))

    # Add all nodes as scatter points
    node_x, node_y, node_color, node_size, node_ids = [], [], [], [], []

    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        node_ids.append(str(node))
        
        if node == substation_node:
            node_color.append('red')
            node_size.append(10)
        else:
            node_color.append('blue')
            node_size.append(5)

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(color=node_color, size=node_size),
        text=node_ids,
        hoverinfo='text',
        showlegend=False
    ))

    return fig

# Define the layout of the Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Network Map with Path Finding to Substation"),
    dcc.Graph(figure=create_figure(), id='graph', style={'height': '90vh'}),
    html.Div(id='node-info', style={'whiteSpace': 'pre-line'}),
    html.Div(id='path-info', style={'whiteSpace': 'pre-line'}),
    dcc.Input(id='file-name', type='text', placeholder='Enter file name'),
    html.Button('Save Paths', id='save-button'),
    html.Div(id='save-info', style={'whiteSpace': 'pre-line'})
])

# State to keep track of selections
selections = []
path_color_flag = 0

# Callback to update the graph and path information when a node is double-clicked
@app.callback(
    [Output('graph', 'figure'),
     Output('node-info', 'children'),
     Output('path-info', 'children')],
    [Input('graph', 'clickData')],
    [State('graph', 'figure')]
)
def update_path(clickData, current_fig):
    global selections, path_color_flag
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_fig, "Double-click on a node to find path to substation.", ""

    point = clickData['points'][0]
    node_id = point['text']
    x, y = point['x'], point['y']

    try:
        path = nx.shortest_path(G, source=node_id, target=substation_node)
        
        # Create a new trace for the path
        path_x, path_y = [], []
        for node in path:
            x, y = G.nodes[node]['pos']
            path_x.append(x)
            path_y.append(y)
        
        color = 'red' if path_color_flag == 0 else 'green'
        new_trace = go.Scatter(
            x=path_x, y=path_y,
            mode='lines',
            line=dict(color=color, width=2),
            name=f'Path from {node_id}'
        )
        
        current_fig['data'].append(new_trace)
        
        # Save the selection
        selections.append({
            'node_id': node_id,
            'coordinates': (x, y),
            'path': path
        })
        
        return current_fig, f"Selected Node ID: {node_id}\nCoordinates: ({x}, {y})", f"Path found: {' -> '.join(map(str, path))}"
    except nx.NetworkXNoPath:
        return current_fig, f"Selected Node ID: {node_id}\nCoordinates: ({x}, {y})", "No path found to the substation."

# Callback to save the paths to a JSON file
@app.callback(
    Output('save-info', 'children'),
    [Input('save-button', 'n_clicks')],
    [State('file-name', 'value')]
)
def save_paths(n_clicks, file_name):
    global selections, path_color_flag
    if n_clicks is None or not file_name:
        return "Enter a file name and click 'Save Paths' to save the paths."

    file_path = f"{file_name}.json"
    with open(file_path, 'w') as f:
        json.dump(selections, f, indent=4)
    
    # Clear selections and switch path color
    selections = []
    path_color_flag = 1 - path_color_flag
    
    return f"Paths saved to {file_path}"

if __name__ == '__main__':
    app.run_server(debug=True)