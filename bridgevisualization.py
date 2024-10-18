import json
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import networkx as nx
import plotly.graph_objs as go
from flask import Flask

# Load the graph from the cache
cache_path = "C:/Users/eljapo22/gephi/cache/7644e20afe6c4a94db3dcb733771bb0d4d5a6fc6.json"
saved_coordinates_path = "saved_coordinates.json"

def load_graph_from_custom_cache(filename=cache_path):
    with open(filename, 'r') as f:
        data = json.load(f)

    if "elements" not in data:
        raise ValueError("The JSON data does not contain the 'elements' key.")

    elements = data["elements"]

    G = nx.MultiDiGraph()

    for item in elements:
        if item['type'] == 'node':
            node_id = item['id']
            node_attrs = {
                'lat': item['lat'],
                'lon': item['lon']
            }
            if 'tags' in item:
                node_attrs.update(item['tags'])
            G.add_node(node_id, **node_attrs)

    for item in elements:
        if item['type'] == 'way':
            nodes = item['nodes']
            for i in range(len(nodes) - 1):
                u = nodes[i]
                v = nodes[i + 1]
                G.add_edge(u, v)

    G.graph['crs'] = {'init': 'epsg:32617'}

    for node, data in G.nodes(data=True):
        G.nodes[node]['x'] = data['lon']
        G.nodes[node]['y'] = data['lat']

    return G

def save_coordinate(node_id, coordinates):
    try:
        with open(saved_coordinates_path, 'r') as f:
            saved_coords = json.load(f)
    except FileNotFoundError:
        saved_coords = []

    saved_coords.append({'node_id': node_id, 'coordinates': coordinates})

    with open(saved_coordinates_path, 'w') as f:
        json.dump(saved_coords, f, indent=2)

# Load the graph
G_proj = load_graph_from_custom_cache()

# Create the Dash app
app = dash.Dash(__name__)
server = app.server

# Extract node coordinates
node_x = [data['lon'] for node, data in G_proj.nodes(data=True)]
node_y = [data['lat'] for node, data in G_proj.nodes(data=True)]
node_ids = list(G_proj.nodes())

# Layout
app.layout = html.Div([
    html.H1("Click a node to get its details"),
    dcc.Graph(
        id='network-graph',
        figure={
            'data': [go.Scattergl(
                x=node_x,
                y=node_y,
                mode='markers',
                marker=dict(size=5, color='blue'),
                text=node_ids,
                hoverinfo='text'
            )],
            'layout': go.Layout(
                title='Interactive Network Map',
                dragmode='select',
                margin=dict(l=0, r=0, t=0, b=0),
                hovermode='closest'
            )
        }
    ),
    html.Div(id='node-info'),
    html.Button('Save Selection', id='save-button', n_clicks=0),
    dcc.Store(id='selected-nodes', data=[])
])

# Callbacks
@app.callback(
    Output('node-info', 'children'),
    Output('save-button', 'style'),
    Input('network-graph', 'selectedData')
)
def display_selected_data(selectedData):
    if selectedData:
        selected_nodes = [point['text'] for point in selectedData['points']]
        return f"Selected Nodes: {', '.join(selected_nodes)}", {'display': 'block'}
    return "No nodes selected", {'display': 'none'}

@app.callback(
    Output('selected-nodes', 'data'),
    Input('save-button', 'n_clicks'),
    State('network-graph', 'selectedData'),
    State('selected-nodes', 'data')
)
def save_selected_nodes(n_clicks, selectedData, stored_data):
    if n_clicks > 0 and selectedData:
        selected_nodes = [point['text'] for point in selectedData['points']]
        for node_id in selected_nodes:
            node_data = G_proj.nodes[int(node_id)]
            coordinates = {'lat': node_data['lat'], 'lon': node_data['lon']}
            save_coordinate(node_id, coordinates)
        return selected_nodes
    return stored_data

if __name__ == '__main__':
    app.run_server(debug=True)
