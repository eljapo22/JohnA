import json
import networkx as nx
import plotly.graph_objects as go
import plotly.colors as pc
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from collections import defaultdict
from tqdm import tqdm
import os

def load_combined_data(json_file_path):
    with open(json_file_path, 'r') as f:
        return json.load(f)

def create_graph_from_combined_data(data):
    G = nx.Graph()
    print("Loading nodes and edges...")
    for node_id, node_data in tqdm(data['nodes'].items(), desc="Loading nodes"):
        G.add_node(node_id, pos=(float(node_data['longitude']), float(node_data['latitude'])), 
                   shape=node_data.get('shape', 'circle'), type=node_data['type'],
                   display=node_data.get('display', node_id))
    for edge in tqdm(data['edge_list'], desc="Loading edges"):
        G.add_edge(edge['source'], edge['target'], level=edge.get('level'), path_type=edge.get('path_type'))
    return G

def generate_custom_labels(G, combined_data):
    transformer_count = 0
    for node, data in G.nodes(data=True):
        if data['type'] == 'Transformer':
            transformer_count += 1
            label = f"D/S T/F: {chr(65 + transformer_count - 1) * 2}"
            G.nodes[node]['display'] = label
            combined_data['nodes'][node]['display'] = label  # Update combined_data

def save_combined_data(combined_data, file_path):
    with open(file_path, 'w') as f:
        json.dump(combined_data, f, indent=2)
    print(f"Updated combined structure saved to: {file_path}")

def generate_distinct_colors(num_levels):
    colors = [
        '#FF0000', '#006400', '#0000FF', '#FF8C00', '#FF00FF', '#000000', '#800000'
    ]
    return colors[:num_levels]

def create_figure(G, combined_data):
    print("Creating visualization...")
    fig = go.Figure()

    max_level = combined_data['metadata']['max_level']
    distinct_colors = generate_distinct_colors(max_level)

    for edge in G.edges(data=True):
        start_node, end_node = edge[0], edge[1]
        level = edge[2].get('level', 1)
        x0, y0 = G.nodes[start_node]['pos']
        x1, y1 = G.nodes[end_node]['pos']
        
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines',
                                 line=dict(width=2, color=distinct_colors[level-1]),
                                 hoverinfo='text',
                                 hovertext=f"Level: {level}, Start: {start_node}, End: {end_node}"))

    node_x, node_y, node_color, node_size, node_symbol, node_text, node_ids = [], [], [], [], [], [], []
    for node, data in G.nodes(data=True):
        x, y = data['pos']
        node_x.append(x)
        node_y.append(y)
        node_ids.append(node)
        
        if data['type'] == 'Substation':
            node_color.append('red')
            node_size.append(20)
            node_symbol.append('square')
            node_text.append(data['display'])
        elif data['type'] == 'Transformer':
            node_color.append('green')
            node_size.append(12)
            node_symbol.append('triangle-up')
            node_text.append(data['display'])
        elif data['type'] == 'Intersection':
            node_color.append('red')
            node_size.append(10)
            node_symbol.append('circle')
            node_text.append(data['display'])
        else:
            node_color.append('lightblue')
            node_size.append(5)
            node_symbol.append('circle')
            node_text.append('')

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(color=node_color, size=node_size, symbol=node_symbol),
        text=node_text,
        textposition="top center",
        hoverinfo='none',
        customdata=node_ids,
    ))

    fig.update_layout(showlegend=False, hovermode='closest',
                      margin=dict(b=20, l=5, r=5, t=40),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

    return fig

def print_network_summary(combined_data):
    print("\nNetwork Summary:")
    print(f"Total nodes: {len(combined_data['nodes'])}")
    print(f"Total edges: {len(combined_data['edge_list'])}")
    print(f"Max hierarchy level: {combined_data['metadata']['max_level']}")
    print(f"Number of transformers: {len(combined_data['metadata']['roots'])}")
    print(f"Number of intersections: {len(combined_data['metadata']['intersections'])}")
    print(f"Number of main paths: {len(combined_data['paths']['main_paths'])}")
    print(f"Number of secondary paths: {sum(len(paths) for paths in combined_data['paths']['secondary_paths'].values())}")

print("Starting main process...")
file_path = r"C:\Users\eljapo22\gephi\Node2Edge2JSON\feeders\combined_structure_Middle.json"
combined_data = load_combined_data(file_path)
G = create_graph_from_combined_data(combined_data)
generate_custom_labels(G, combined_data)
save_combined_data(combined_data, file_path)

print_network_summary(combined_data)

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Network Map with Identified Intersections, Roots, and Common End"),
    dcc.Graph(figure=create_figure(G, combined_data), id='graph', style={'height': '80vh'}),
    html.Div(id='click-data', style={'margin-top': '20px'})
])

@app.callback(
    Output('click-data', 'children'),
    Input('graph', 'clickData')
)
def display_click_data(clickData):
    if clickData is None:
        return "Double-click a transformer node to see its unique ID"
    
    point = clickData['points'][0]
    node_id = point['customdata']
    node_data = combined_data['nodes'].get(node_id)
    
    if node_data and node_data['type'] == 'Transformer':
        return f"Transformer Unique ID: {node_id}"
    else:
        return "Double-click a transformer node to see its unique ID"

if __name__ == '__main__':
    print("Starting server...")
    app.run_server(debug=True)