import json
import networkx as nx
import plotly.graph_objects as go
import dash
from dash import Dash, dcc, html, Input, Output, State

# Load the graph from the cache
def load_graph_from_cache():
    with open("C:\\Users\\eljapo22\\gephi\\cache\\7644e20afe6c4a94db3dcb733771bb0d4d5a6fc6.json", 'r') as f:
        data = json.load(f)
    G = nx.Graph()
    for element in data['elements']:
        if element['type'] == 'node':
            G.add_node(element['id'], pos=(float(element['lon']), float(element['lat'])))
        elif element['type'] == 'way':
            for i in range(len(element['nodes']) - 1):
                G.add_edge(element['nodes'][i], element['nodes'][i + 1])
    return G

G = load_graph_from_cache()

# Define IDs for the special nodes
ds_substation_aa_id = 24959509
ds_tf_bb_id = 34404246

# Create the plot
def create_figure():
    fig = go.Figure()

    # Add all grid edges as a single trace
    edge_x = []
    edge_y = []
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
    node_x = []
    node_y = []
    node_color = []
    node_size = []
    node_ids = []

    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        node_ids.append(str(node))
        
        if node == ds_substation_aa_id:
            node_color.append('red')
            node_size.append(10)
        elif node == ds_tf_bb_id:
            node_color.append('blue')
            node_size.append(10)
        else:
            node_color.append('blue')
            node_size.append(5)

    fig.add_trace(go.Scatter(
        x=node_x, 
        y=node_y,
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
    html.H1("Network Map with Clickable Nodes and Path Finding"),
    dcc.Input(id='start-node', type='text', placeholder='Start Node ID'),
    dcc.Input(id='end-node', type='text', placeholder='End Node ID'),
    html.Button('Find Path', id='find-path-button'),
    dcc.Graph(figure=create_figure(), id='graph', style={'height': '90vh'}),
    html.Div(id='node-info', style={'whiteSpace': 'pre-line'}),
    html.Div(id='path-info', style={'whiteSpace': 'pre-line'})
])

# Callback to update the node information when a node is clicked
@app.callback(
    Output('node-info', 'children'),
    [Input('graph', 'clickData')]
)
def display_node_info(clickData):
    if clickData:
        point = clickData['points'][0]
        x = point['x']
        y = point['y']
        node_id = point['text']
        return f"Clicked Node ID: {node_id}\nCoordinates: ({x}, {y})"
    return "Click on a node to see its information."

# Callback for path finding
@app.callback(
    [Output('graph', 'figure'),
     Output('path-info', 'children')],
    [Input('find-path-button', 'n_clicks')],
    [State('start-node', 'value'),
     State('end-node', 'value'),
     State('graph', 'figure')]
)
def update_path(n_clicks, start_node, end_node, current_fig):
    if n_clicks is None:
        return current_fig, "Enter start and end nodes and click 'Find Path'"
    
    try:
        path = nx.shortest_path(G, source=int(start_node), target=int(end_node))
        
        # Create a new trace for the path
        path_x, path_y = [], []
        for node in path:
            x, y = G.nodes[node]['pos']
            path_x.append(x)
            path_y.append(y)
        
        new_trace = go.Scatter(
            x=path_x, y=path_y,
            mode='lines',
            line=dict(color='red', width=1),
            name='Path'
        )
        
        # Highlight start and end nodes
        start_x, start_y = G.nodes[int(start_node)]['pos']
        end_x, end_y = G.nodes[int(end_node)]['pos']
        
        highlight_trace = go.Scatter(
            x=[start_x, end_x],
            y=[start_y, end_y],
            mode='markers',
            marker=dict(
                color=['green', 'purple'],
                size=[15, 15],
                symbol=['star', 'star-triangle-up']
            ),
            text=['Start Node', 'End Node'],
            hoverinfo='text',
            name='Start/End Nodes'
        )
        
        current_fig['data'].append(new_trace)
        current_fig['data'].append(highlight_trace)
        
        return current_fig, f"Path found: {' -> '.join(map(str, path))}"
    except nx.NetworkXNoPath:
        return current_fig, "No path found between the specified nodes."
    except ValueError:
        return current_fig, "Please enter valid node IDs."

if __name__ == '__main__':
    app.run_server(debug=True)