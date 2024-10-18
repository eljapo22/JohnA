import json
import networkx as nx
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import pickle
import os
from tqdm import tqdm
import gzip

def load_and_process_data(file_path):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    G = nx.Graph()
    
    print("Processing nodes...")
    for node in tqdm(data['nodes']):
        G.add_node(node['id'], **node)
    
    print("Processing edges...")
    for edge in tqdm(data['edges']):
        G.add_edge(edge['source'], edge['target'], **edge)
    
    return G

def calculate_bounds(G):
    longitudes = [node['longitude'] for node in G.nodes.values()]
    latitudes = [node['latitude'] for node in G.nodes.values()]
    return min(longitudes), max(longitudes), min(latitudes), max(latitudes)

def save_compressed_cache(data, cache_file):
    print(f"Saving compressed cache to {cache_file}...")
    with gzip.open(cache_file, 'wb') as f:
        pickle.dump(data, f)

def load_compressed_cache(cache_file):
    if os.path.exists(cache_file):
        print(f"Loading compressed cache from {cache_file}...")
        with gzip.open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None

def create_edge_trace(G):
    print("Creating edge trace...")
    edge_x, edge_y = [], []
    for edge in tqdm(G.edges()):
        x0, y0 = G.nodes[edge[0]]['longitude'], G.nodes[edge[0]]['latitude']
        x1, y1 = G.nodes[edge[1]]['longitude'], G.nodes[edge[1]]['latitude']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    return go.Scattergl(x=edge_x, y=edge_y, mode='lines',
                        line=dict(width=1, color='rgba(0,0,255,0.2)'),
                        hoverinfo='none')

def create_node_trace(G):
    print("Creating node trace...")
    node_x = [node['longitude'] for node in G.nodes.values()]
    node_y = [node['latitude'] for node in G.nodes.values()]
    node_text = [f"Node ID: {node}" for node in G.nodes()]
    return go.Scattergl(x=node_x, y=node_y, mode='markers',
                        marker=dict(size=2, color='red'),
                        hoverinfo='text',
                        text=node_text)



# Main process
print("Starting main process...")
file_path = r"C:\Users\eljapo22\gephi\Node2Edge2JSON\Node2Edge2JSON.json"
cache_file = 'node2edge_cache.gz'

cached_data = load_compressed_cache(cache_file)

if cached_data is None or 'bounds' not in cached_data:
    print("Cache not found or outdated. Processing data...")
    G = load_and_process_data(file_path)
    edge_trace = create_edge_trace(G)
    node_trace = create_node_trace(G)
    bounds = calculate_bounds(G)
    cached_data = {
        'edge_trace': edge_trace,
        'node_trace': node_trace,
        'bounds': bounds
    }
    save_compressed_cache(cached_data, cache_file)
else:
    print("Data loaded from cache.")

edge_trace = cached_data['edge_trace']
node_trace = cached_data['node_trace']
min_longitude, max_longitude, min_latitude, max_latitude = cached_data['bounds']

# Create Dash app
app = Dash(__name__)

# Define the app layout only once
app.layout = html.Div([
    html.H1("Node2Edge2JSON Network Visualization"),
    dcc.Graph(id='graph', style={'height': '90vh'}, config={'displayModeBar': True, 'scrollZoom': True}),
    html.Div(id='click-data')
])

@app.callback(
    Output('graph', 'figure'),
    Input('graph', 'relayoutData'),
    State('graph', 'figure')
)
def update_figure(relayoutData, current_figure):
    print("Updating figure...")
    
    if current_figure is None:
        # Initial render
        fig = go.Figure(data=[edge_trace, node_trace])
        layout = dict(
            showlegend=False, 
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                range=[min_longitude, max_longitude],
                autorange=False,
                scaleanchor="y",
                scaleratio=1,
            ),
            yaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                range=[min_latitude, max_latitude],
                autorange=False,
                scaleanchor="x",
                scaleratio=1,
            ),
            uirevision='constant'
        )
        fig.update_layout(layout)
    else:
        # Subsequent updates
        fig = go.Figure(data=current_figure['data'], layout=current_figure['layout'])
        
        if relayoutData is not None:
            # Update only specific parts of the layout if changed
            updates = {}
            if 'xaxis.range[0]' in relayoutData and 'xaxis.range[1]' in relayoutData:
                updates['xaxis.range'] = [relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']]
            if 'yaxis.range[0]' in relayoutData and 'yaxis.range[1]' in relayoutData:
                updates['yaxis.range'] = [relayoutData['yaxis.range[0]'], relayoutData['yaxis.range[1]']]
            
            fig.update_layout(updates)
    
    return fig

@app.callback(
    Output('click-data', 'children'),
    Input('graph', 'clickData')
)
def display_click_data(clickData):
    if clickData is None:
        return "Click a node to see its details"
    point = clickData['points'][0]
    return f"Clicked Point: Longitude {point['x']}, Latitude {point['y']}"

if __name__ == '__main__':
    print("Starting server...")
    app.run_server(debug=True)