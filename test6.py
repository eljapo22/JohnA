import json
import networkx as nx
import plotly.graph_objects as go
import dash
from dash import Dash, dcc, html, Input, Output

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

# Add all nodes as scatter points, differentiating DS Substation AA and DS T/F BB
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

# Define the layout of the Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Network Map with Clickable Nodes"),
    dcc.Graph(figure=fig, id='graph', style={'height': '90vh'}),
    html.Div(id='node-info', style={'whiteSpace': 'pre-line'})
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

if __name__ == '__main__':
    app.run_server(debug=True)
