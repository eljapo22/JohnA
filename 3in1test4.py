import networkx as nx
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import os
from tqdm import tqdm
import pickle
import gzip
import logging
import json

# Cache file path
CACHE_FILE = 'graph_cache.pkl.gz'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pickle_cache(file_path):
    """Load and return the contents of a pickle file."""
    if os.path.exists(file_path):
        with gzip.open(file_path, 'rb') as f:
            return pickle.load(f)
    return None

def save_pickle_cache(data, file_path):
    """Save data to a pickle file."""
    try:
        with gzip.open(file_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info("Cache saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")

def create_graph_from_combined_data(data):
    G = nx.Graph()
    print("Loading nodes and edges...")
    nodes = data.get('nodes', {})
    for node_id, node_data in tqdm(nodes.items(), desc="Loading nodes"):
        if isinstance(node_data, dict):
            G.add_node(node_id, pos=(float(node_data['longitude']), float(node_data['latitude'])), 
                       shape=node_data.get('shape', 'circle'), type=node_data['type'],
                       display=node_data.get('display', node_id))
    edges = data.get('edges', {})
    for edge_id, edge_data in tqdm(edges.items(), desc="Loading edges"):
        if isinstance(edge_data, dict):
            G.add_edge(edge_data['source'], edge_data['target'], level=edge_data.get('level'), path_type=edge_data.get('path_type'))
    return G

def load_graphs():
    file_paths = [
        r"C:\Users\eljapo22\gephi\Node2Edge2JSON\feeders\combined_structure_BottomLeft.json",
        r"C:\Users\eljapo22\gephi\Node2Edge2JSON\feeders\combined_structure_Middle.json",
        r"C:\Users\eljapo22\gephi\Node2Edge2JSON\feeders\combined_structure_TopLeft.json"
    ]
    graphs = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)
        G = create_graph_from_combined_data(data)
        graphs.append((G, data))
    return graphs

def load_all_edges_graph(file_path):
    """Load a graph with all nodes and edges from a separate file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    G = nx.Graph()
    print("Loading nodes and edges from Node2Edge2JSON...")
    nodes = data.get('nodes', {})
    for node_id, node_data in tqdm(nodes.items(), desc="Loading nodes"):
        if isinstance(node_data, dict):
            G.add_node(node_id, pos=(float(node_data['longitude']), float(node_data['latitude'])), 
                       shape=node_data.get('shape', 'circle'), type=node_data['type'],
                       display=node_data.get('display', node_id))
    edges = data.get('edges', {})
    for edge_id, edge_data in tqdm(edges.items(), desc="Loading edges"):
        if isinstance(edge_data, dict):
            G.add_edge(edge_data['source'], edge_data['target'], level=edge_data.get('level'), path_type=edge_data.get('path_type'))
    return G

def create_edge_trace(G, color='rgba(0,0,255,0.2)'):
    print("Creating edge trace...")
    edge_x, edge_y = [], []
    for edge in tqdm(G.edges()):
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    return go.Scattergl(x=edge_x, y=edge_y, mode='lines',
                        line=dict(width=1, color=color),
                        hoverinfo='none')

def create_specific_node_traces_for_graph(G):
    """Create node traces for specific node types in a given graph."""
    node_traces = {
        'Substation': {'x': [], 'y': [], 'text': []},
        'Transformer': {'x': [], 'y': [], 'text': []},
        'Intersection': {'x': [], 'y': [], 'text': []},
        'Other': {'x': [], 'y': [], 'text': []}
    }
    for node, node_data in G.nodes(data=True):
        x, y = node_data['pos']
        node_type = node_data.get('type', 'Other')
        if node_type in node_traces:
            node_traces[node_type]['x'].append(x)
            node_traces[node_type]['y'].append(y)
            node_traces[node_type]['text'].append(f"Node ID: {node}, Type: {node_type}, Display: {node_data.get('display', node)}")
    
    scatter_traces = []
    for node_type, trace_data in node_traces.items():
        scatter = go.Scattergl(
            x=trace_data['x'],
            y=trace_data['y'],
            mode='markers',
            name=node_type,
            marker=dict(
                size=20 if node_type == 'Substation' else 12 if node_type == 'Transformer' else 10,
                color='red' if node_type == 'Substation' else 'blue' if node_type == 'Transformer' else 'green' if node_type == 'Intersection' else 'gray',
                symbol='square' if node_type == 'Substation' else 'triangle-up' if node_type == 'Transformer' else 'circle',
                opacity=0.3 if node_type == 'Other' else 1.0  # Set opacity for "Other" nodes
            ),
            text=trace_data['text'],
            hoverinfo='text'
        )
        scatter_traces.append(scatter)
    
    return scatter_traces

def calculate_bounds(G):
    logger.info("Calculating bounds...")
    pos = nx.get_node_attributes(G, 'pos')
    if not pos:
        logger.warning("No position data found for nodes.")
        return -180, 180, -90, 90  # Default to global bounds
    xs, ys = zip(*pos.values())
    bounds = {
        'min_longitude': min(xs),
        'max_longitude': max(xs),
        'min_latitude': min(ys),
        'max_latitude': max(ys)
    }
    logger.info(f"Bounds calculated: {bounds}")
    return bounds['min_longitude'], bounds['max_longitude'], bounds['min_latitude'], bounds['max_latitude']

def create_highlighted_path_trace(G, paths):
    edge_x, edge_y = [], []
    for path in paths:
        for start, end in zip(path[:-1], path[1:]):
            x0, y0 = G.nodes[start]['pos']
            x1, y1 = G.nodes[end]['pos']
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
    return go.Scattergl(x=edge_x, y=edge_y, mode='lines', name='Highlighted Path',
                        line=dict(width=3, color='yellow'), hoverinfo='none')

def find_paths_between_nodes(G, source, targets):
    """Find paths between a source node and multiple target nodes."""
    paths = []
    for target in targets:
        try:
            path = nx.shortest_path(G, source, target)
            paths.append(path)
        except nx.NetworkXNoPath:
            print(f"No path found between {source} and {target}")
    return paths

# Load graphs
cached_data = load_pickle_cache(CACHE_FILE)

if cached_data is None or 'node_traces_list' not in cached_data:
    print("Cache not found or outdated. Processing data...")
    graphs = load_graphs()
    combined_graph = nx.compose_all([G for G, _ in graphs])  # Combine all graphs

    # Load all edges graph
    all_edges_graph = load_all_edges_graph(r"C:\Users\eljapo22\gephi\Node2Edge2JSON\Node2Edge2JSON.json")
    all_edges_trace = create_edge_trace(all_edges_graph, color='rgba(0,0,0,0.1)')  # Light grey for all edges

    # Create traces for Dash
    edge_traces = [create_edge_trace(G, color=f'rgba(0,0,255,{0.2 + i*0.2})') for i, (G, _) in enumerate(graphs)]
    combined_edge_trace = create_edge_trace(combined_graph)
    node_traces_list = [create_specific_node_traces_for_graph(G) for G, _ in graphs]
    combined_node_traces = create_specific_node_traces_for_graph(combined_graph)  # Create combined node traces
    min_longitude, max_longitude, min_latitude, max_latitude = calculate_bounds(combined_graph)

    # Find paths for highlighting
    special_nodes = ['N-000023033', 'N-000023029', 'N-000020297', 'N-000021825']
    source_node = 'N-000021825'
    paths = find_paths_between_nodes(combined_graph, source_node, special_nodes[:-1])
    highlighted_path_trace = create_highlighted_path_trace(combined_graph, paths)

    # Save cache
    cached_data = {
        'edge_traces': edge_traces,
        'combined_edge_trace': combined_edge_trace,
        'node_traces_list': node_traces_list,
        'combined_node_traces': combined_node_traces,
        'all_edges_trace': all_edges_trace,
        'bounds': (min_longitude, max_longitude, min_latitude, max_latitude),
        'highlighted_path_trace': highlighted_path_trace
    }
    logger.info(f"Data to be cached: {cached_data}")
    save_pickle_cache(cached_data, CACHE_FILE)
else:
    print("Data loaded from cache.")
    edge_traces = cached_data['edge_traces']
    combined_edge_trace = cached_data['combined_edge_trace']
    node_traces_list = cached_data['node_traces_list']
    combined_node_traces = cached_data['combined_node_traces']
    all_edges_trace = cached_data['all_edges_trace']
    min_longitude, max_longitude, min_latitude, max_latitude = cached_data['bounds']
    highlighted_path_trace = cached_data['highlighted_path_trace']

# Create Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Network Visualization"),
    dcc.Dropdown(
        id='graph-selector',
        options=[
            {'label': 'Combined Graph', 'value': 'combined'},
            {'label': 'Bottom Left Graph', 'value': 'bottom_left'},
            {'label': 'Middle Graph', 'value': 'middle'},
            {'label': 'Top Left Graph', 'value': 'top_left'}
        ],
        value='combined'
    ),
    dcc.Checklist(
        id='show-node-types',
        options=[
            {'label': 'Show Substations', 'value': 'Substation'},
            {'label': 'Show Transformers', 'value': 'Transformer'},
            {'label': 'Show Intersections', 'value': 'Intersection'},
            {'label': 'Show Other Nodes', 'value': 'Other'},
        ],
        value=['Substation', 'Transformer', 'Intersection', 'Other']
    ),
    dcc.Checklist(
        id='show-all-edges',
        options=[{'label': 'Show All Edges', 'value': 'all_edges'}],
        value=[]
    ),
    dcc.Checklist(
        id='show-highlighted-path',
        options=[{'label': 'Show Highlighted Path', 'value': 'show'}],
        value=[]
    ),
    dcc.Graph(id='graph', style={'height': '90vh'}, config={'displayModeBar': True, 'scrollZoom': True}),
    html.Div(id='click-data')
])

@app.callback(
    Output('graph', 'figure'),
    [Input('graph-selector', 'value'),
     Input('show-node-types', 'value'),
     Input('show-all-edges', 'value'),
     Input('show-highlighted-path', 'value'),
     Input('graph', 'relayoutData')],
    State('graph', 'figure')
)
def update_figure(selected_graph, show_node_types, show_all_edges, show_highlighted_path, relayoutData, current_figure):
    print("Updating figure...")
    
    if selected_graph == 'combined':
        edge_data = [combined_edge_trace]
        node_data = combined_node_traces
    elif selected_graph == 'bottom_left':
        edge_data = [edge_traces[0]]
        node_data = node_traces_list[0]
    elif selected_graph == 'middle':
        edge_data = [edge_traces[1]]
        node_data = node_traces_list[1]
    elif selected_graph == 'top_left':
        edge_data = [edge_traces[2]]
        node_data = node_traces_list[2]
    
    node_data = [trace for trace in node_data if trace.name in show_node_types]
    
    data = edge_data + node_data
    
    if 'all_edges' in show_all_edges:
        data.insert(0, all_edges_trace)  # Add all edges first to ensure they are in the background
    
    if 'show' in show_highlighted_path:
        data.append(highlighted_path_trace)  # Add highlighted path last to overlay

    if current_figure is None or current_figure['data'] != data:
        # Initial render or graph change
        fig = go.Figure(data=data)
        layout = dict(
            showlegend=True, 
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
        fig = go.Figure(data=data, layout=current_figure['layout'])
        
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
    return point['text']

if __name__ == '__main__':
    print("Starting server...")
    app.run_server(debug=True)