import json
import networkx as nx
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from tqdm import tqdm
import os
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

def load_combined_data(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    print(f"Keys in {os.path.basename(json_file_path)}: {list(data.keys())}")
    return data

def create_graph_from_combined_data(data, file_name, k_value, progress_callback=None):
    G = nx.Graph()
    print(f"Loading data from {file_name}...")
    
    if file_name == 'combined_structure_Middle.json':
        if isinstance(data['nodes'], list):
            for node_data in tqdm(data['nodes'], desc="Loading nodes"):
                G.add_node(node_data['id'], **node_data)
        elif isinstance(data['nodes'], dict):
            for node_id, node_data in tqdm(data['nodes'].items(), desc="Loading nodes"):
                G.add_node(node_id, **node_data)
        else:
            raise ValueError("Unexpected 'nodes' structure in JSON data")

        edge_key = 'edge_list' if 'edge_list' in data else 'edges'
        for edge in tqdm(data[edge_key], desc="Loading edges"):
            G.add_edge(edge['source'], edge['target'], **edge)
        
        if 'paths' in data and 'main_paths' in data['paths']:
            for path_key, paths in data['paths']['main_paths'].items():
                if paths and isinstance(paths, list) and paths[0]:
                    main_path = paths[0]
                    for i in range(len(main_path) - 1):
                        start, end = main_path[i], main_path[i+1]
                        if G.has_edge(start, end):
                            G[start][end]['is_middle_path'] = True
    
    elif file_name == 'Node2Edge2JSON.json':
        for node in tqdm(data['nodes'], desc="Loading nodes"):
            G.add_node(node['id'], longitude=node['longitude'], latitude=node['latitude'])
        for edge in tqdm(data['edges'], desc="Loading edges"):
            G.add_edge(edge['source'], edge['target'], **edge)
        
        # Compute edge importance scores
        importance_scores = compute_edge_importance(G, k_value, progress_callback)
        nx.set_edge_attributes(G, importance_scores, 'importance')
    
    return G

def compute_edge_importance(graph, k_value, progress_callback=None):
    if progress_callback:
        progress_callback(15, f"Starting edge importance computation with k={k_value}")
    
    print(f"Computing edge importance with k={k_value}...")
    edge_betweenness = nx.edge_betweenness_centrality(graph, k=k_value)
    
    if progress_callback:
        progress_callback(20, "Edge betweenness centrality computed")
    
    adjacency_matrix = nx.to_scipy_sparse_array(graph)
    mst = minimum_spanning_tree(adjacency_matrix)
    mst_edges = set(map(tuple, np.array(mst.nonzero()).T))
    
    if progress_callback:
        progress_callback(25, "Minimum spanning tree created")
    
    importance_scores = {}
    for edge, betweenness in edge_betweenness.items():
        importance = betweenness
        if edge in mst_edges or (edge[1], edge[0]) in mst_edges:
            importance += 1  # Boost importance for MST edges
        importance_scores[edge] = importance
    
    if progress_callback:
        progress_callback(30, "Edge importance scores assigned")
    
    return importance_scores

def create_adaptive_lod(graph, threshold):
    print(f"Creating adaptive LOD with threshold {threshold}")
    importance_scores = nx.get_edge_attributes(graph, 'importance')
    edges = sorted(graph.edges(data=True), key=lambda x: importance_scores.get((x[0], x[1]), 0), reverse=True)
    num_edges_to_keep = int(len(edges) * threshold)
    
    reduced_graph = nx.Graph()
    reduced_graph.add_nodes_from(graph.nodes(data=True))
    for edge in edges[:num_edges_to_keep]:
        reduced_graph.add_edge(edge[0], edge[1], **edge[2])
    
    print(f"Reduced graph has {reduced_graph.number_of_edges()} edges")
    return reduced_graph

def print_network_summary(combined_data):
    print("\nNetwork Summary:")
    print(f"Total nodes: {len(combined_data['nodes'])}")
    
    edge_keys = ['edge_list', 'edges', 'links']
    edge_count = sum(len(combined_data.get(key, [])) for key in edge_keys)
    print(f"Total edges: {edge_count}")
    
    if 'metadata' in combined_data:
        metadata = combined_data['metadata']
        print(f"Max hierarchy level: {metadata.get('max_level', 'N/A')}")
        print(f"Number of transformers: {len(metadata.get('roots', []))}")
        print(f"Number of intersections: {len(metadata.get('intersections', []))}")
    
    if 'paths' in combined_data:
        paths = combined_data['paths']
        print(f"Number of main paths: {len(paths.get('main_paths', []))}")
        secondary_paths = paths.get('secondary_paths', {})
        print(f"Number of secondary paths: {sum(len(p) for p in secondary_paths.values())}")

print("Starting main process...")
file_paths = [
    r"C:\Users\eljapo22\gephi\Node2Edge2JSON\Node2Edge2JSON.json",
    r"C:\Users\eljapo22\gephi\Node2Edge2JSON\feeders\combined_structure_Middle.json"
]

color_mapping = {
    'combined_structure_Middle.json': 'purple',
    'Node2Edge2JSON.json': 'orange',
}

graphs = {}
combined_data = {}

for file_path in file_paths:
    file_name = os.path.basename(file_path)
    combined_data[file_name] = load_combined_data(file_path)
    graphs[file_name] = create_graph_from_combined_data(combined_data[file_name], file_name, k_value=1000)
    print_network_summary(combined_data[file_name])

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Combined Network Map"),
    dcc.Dropdown(
        id='level-dropdown',
        options=[{'label': f'Level {i}', 'value': i} for i in range(1, 8)],
        value=None,
        clearable=True,
        placeholder="Select a level"
    ),
    dcc.Slider(
        id='detail-slider',
        min=0,
        max=100,
        step=1,
        value=100,
        marks={i: f'{i}%' for i in range(0, 101, 10)},
        tooltip={"placement": "bottom", "always_visible": True}
    ),
    html.Div("Edge Betweenness 'k' value:"),
    dcc.Slider(
        id='k-value-slider',
        min=100,
        max=5000,
        step=100,
        value=1000,
        marks={i: str(i) for i in range(100, 5001, 1000)},
        tooltip={"placement": "bottom", "always_visible": True}
    ),
    dcc.Graph(id='graph', style={'height': '80vh'}),
    html.Div(id='click-data', style={'margin-top': '20px'}),
    html.Div(id='detail-info', style={'margin-top': '20px'}),
    dcc.Store(id='progress-store')
])

@app.callback(
    [Output('graph', 'figure'),
     Output('detail-info', 'children'),
     Output('progress-store', 'data')],
    [Input('detail-slider', 'value'),
     Input('level-dropdown', 'value'),
     Input('k-value-slider', 'value')]
)
def update_graph(detail_level, selected_level, k_value):
    print(f"Updating graph with detail level: {detail_level}, selected level: {selected_level}, k value: {k_value}")
    threshold = detail_level / 100
    
    progress = {'progress': 0, 'status': 'Starting graph update'}
    
    def update_progress(value, status):
        nonlocal progress
        progress = {'progress': value, 'status': status}
    
    update_progress(10, f"Starting graph creation with k={k_value}")
    
    # Recreate the graph with the new k value
    graphs['Node2Edge2JSON.json'] = create_graph_from_combined_data(combined_data['Node2Edge2JSON.json'], 'Node2Edge2JSON.json', k_value, update_progress)
    
    update_progress(40, "Graph created, starting adaptive LOD")
    
    node2edge_graph = create_adaptive_lod(graphs['Node2Edge2JSON.json'], threshold)
    
    update_progress(50, "Adaptive LOD created")
    
    fig = go.Figure()
    
    # Add edges from Node2Edge2JSON graph
    for edge in node2edge_graph.edges():
        start_node, end_node = edge
        x0, y0 = node2edge_graph.nodes[start_node]['longitude'], node2edge_graph.nodes[start_node]['latitude']
        x1, y1 = node2edge_graph.nodes[end_node]['longitude'], node2edge_graph.nodes[end_node]['latitude']
        
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines',
                                 line=dict(width=1, color=color_mapping['Node2Edge2JSON.json']),
                                 hoverinfo='skip'))
    
    update_progress(70, "Node2Edge2JSON edges added")
    
    # Add edges from combined_structure_Middle graph
    for edge in graphs['combined_structure_Middle.json'].edges(data=True):
        if selected_level is not None and edge[2].get('level', 0) != selected_level:
            continue
        start_node, end_node = edge[0], edge[1]
        x0, y0 = graphs['combined_structure_Middle.json'].nodes[start_node]['longitude'], graphs['combined_structure_Middle.json'].nodes[start_node]['latitude']
        x1, y1 = graphs['combined_structure_Middle.json'].nodes[end_node]['longitude'], graphs['combined_structure_Middle.json'].nodes[end_node]['latitude']
        
        color = color_mapping['combined_structure_Middle.json'] if edge[2].get('is_middle_path', False) else 'rgba(112, 128, 144, 0.3)'
        width = 3 if edge[2].get('is_middle_path', False) else 1
        
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines',
                                 line=dict(width=width, color=color),
                                 hoverinfo='skip'))
    
    update_progress(80, "Combined structure edges added")
    
    # Add nodes from combined_structure_Middle graph
    node_data = {'x': [], 'y': [], 'color': [], 'size': [], 'symbol': [], 'text': [], 'ids': []}
    for node, data in graphs['combined_structure_Middle.json'].nodes(data=True):
        if data['type'].lower() != 'regular' and (selected_level is None or data.get('level', 0) == selected_level):
            node_data['x'].append(data['longitude'])
            node_data['y'].append(data['latitude'])
            node_data['ids'].append(node)
            
            if data['type'].lower() == 'substation':
                node_data['color'].append('red')
                node_data['size'].append(20)
                node_data['symbol'].append('square')
            elif data['type'].lower() == 'transformer':
                node_data['color'].append('blue')
                node_data['size'].append(15)
                node_data['symbol'].append('triangle-up')
            elif data['type'].lower() == 'meter':
                node_data['color'].append('green')
                node_data['size'].append(10)
                node_data['symbol'].append('diamond')
            else:
                node_data['color'].append('purple')
                node_data['size'].append(8)
                node_data['symbol'].append('circle')
            
            node_data['text'].append(data.get('display', node))
    
    fig.add_trace(go.Scatter(
        x=node_data['x'], y=node_data['y'],
        mode='markers+text',
        marker=dict(color=node_data['color'], size=node_data['size'], symbol=node_data['symbol']),
        text=node_data['text'],
        textposition="top center",
        hoverinfo='text',
        customdata=node_data['ids'],
    ))
    
    update_progress(90, "Nodes added")
    
    fig.update_layout(showlegend=False, hovermode='closest',
                      margin=dict(b=20, l=5, r=5, t=40),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    
    update_progress(100, "Layout updated")
    
    detail_info = f"Current detail level: {detail_level}%, k value: {k_value}"
    return fig, detail_info, progress

@app.callback(
    Output('click-data', 'children'),
    Input('graph', 'clickData')
)
def display_click_data(clickData):
    if clickData is None:
        return "Click a node to see its details"
    
    point = clickData['points'][0]
    node_id = point['customdata']
    
    for file_name, data in combined_data.items():
        node_data = next((node for node in data['nodes'] if node['id'] == node_id), None)
        if node_data:
            return f"Node ID: {node_id}, Type: {node_data['type']}, Display: {node_data.get('display', 'N/A')}, File: {file_name}"
    
    return "Node not found in any file"

if __name__ == '__main__':
    print("Starting server...")
    app.run_server(debug=True)