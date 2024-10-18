import json
import networkx as nx
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from collections import defaultdict
from tqdm import tqdm
import os

def load_combined_data(json_file_path):
    with open(json_file_path, 'r') as f:
        return json.load(f)

def load_graph_from_json(json_file_path):
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

def create_figure(graphs, color_mapping, selected_level=None):
    print("Creating visualization...")
    fig = go.Figure()

    node_x, node_y, node_color, node_size, node_symbol, node_text, node_ids = [], [], [], [], [], [], []
    meter_positions = []

    for file_name, G in graphs.items():
        edge_color = color_mapping[file_name]
        
        for edge in G.edges(data=True):
            start_node, end_node = edge[0], edge[1]
            x0, y0 = G.nodes[start_node]['pos']
            x1, y1 = G.nodes[end_node]['pos']
            
            # Check edge level and assign color based on selected level
            if selected_level is not None:
                edge_level = edge[2].get('level')
                if edge_level == selected_level:
                    line_color = 'yellow'
                elif edge_level > selected_level:
                    line_color = 'red'
                elif edge_level < selected_level:
                    line_color = 'green'
                else:
                    line_color = edge_color
            else:
                line_color = edge_color
            
            # Modify line style for transformer to meter connections
            if G.nodes[start_node]['type'] == 'Transformer' and G.nodes[end_node]['type'] == 'Meter':
                fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines',
                                         line=dict(width=1, color='black', dash='dash'),
                                         hoverinfo='text',
                                         hovertext=f"File: {file_name}, Start: {start_node}, End: {end_node}, Level: {edge[2].get('level')}"))
            else:
                fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines',
                                         line=dict(width=2, color=line_color),
                                         hoverinfo='text',
                                         hovertext=f"File: {file_name}, Start: {start_node}, End: {end_node}, Level: {edge[2].get('level')}"))

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
                node_color.append(edge_color)
                node_size.append(12)
                node_symbol.append('triangle-up')
                node_text.append(data['display'])
            elif data['type'] == 'Intersection':
                node_color.append('red')
                node_size.append(10)
                node_symbol.append('circle')
                node_text.append(data['display'])
            elif data['type'] == 'Meter':
                node_color.append('black')  # Change meter nodes to black
                node_size.append(8)
                node_symbol.append('diamond')
                node_text.append(data['display'])
                meter_positions.append((x, y))  # Collect meter positions
            else:
                node_color.append('lightgray')
                node_size.append(5)
                node_symbol.append('circle')
                node_text.append('')

    # Add a circumference around meter nodes
    if meter_positions:
        avg_x = sum(x for x, y in meter_positions) / len(meter_positions)
        avg_y = sum(y for x, y in meter_positions) / len(meter_positions)
        radius = max(((x - avg_x)**2 + (y - avg_y)**2)**0.5 for x, y in meter_positions) * 1.5
        circle_x = [avg_x + radius * nx for nx in range(-1, 2)]
        circle_y = [avg_y + radius * ny for ny in range(-1, 2)]
        fig.add_trace(go.Scatter(
            x=circle_x, y=circle_y,
            mode='lines',
            line=dict(color='lightslategrey', width=2, dash='dash'),
            fill='toself',
            fillcolor='rgba(119,136,153,0.2)',
            hoverinfo='none'
        ))

    # Add dashed dotted lines for specific connections using Node2Edge2JSON.json
    special_nodes = ['N-000023033', 'N-000023029', 'N-000020297', 'N-000021825']
    node2edge_data = load_graph_from_json(r"C:\Users\eljapo22\gephi\Node2Edge2JSON.json")
    
    G = nx.Graph()
    for node in node2edge_data['nodes']:
        G.add_node(node['id'], pos=(node['longitude'], node['latitude']))
    for edge in node2edge_data['edges']:
        G.add_edge(edge['source'], edge['target'])
    
    for i in range(len(special_nodes) - 1):
        start_node = special_nodes[i]
        end_node = special_nodes[i + 1]
        
        if start_node in G.nodes and end_node in G.nodes:
            try:
                path = nx.shortest_path(G, source=start_node, target=end_node)
                path_edges = list(zip(path, path[1:]))
                
                for edge in path_edges:
                    x0, y0 = G.nodes[edge[0]]['pos']
                    x1, y1 = G.nodes[edge[1]]['pos']
                    fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines',
                                             line=dict(width=5, color='purple', dash='dashdot'),
                                             hoverinfo='text',
                                             hovertext=f"Special Connection: {edge[0]} to {edge[1]}"))
            except nx.NetworkXNoPath:
                print(f"No path between {start_node} and {end_node} in Node2Edge2JSON.json")
        else:
            if start_node not in G.nodes:
                print(f"Node {start_node} not found in Node2Edge2JSON.json")
            if end_node not in G.nodes:
                print(f"Node {end_node} not found in Node2Edge2JSON.json")

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
file_paths = [
    r"C:\Users\eljapo22\gephi\Node2Edge2JSON\feeders\combined_structure_BottomLeft.json",
    r"C:\Users\eljapo22\gephi\Node2Edge2JSON\feeders\combined_structure_Middle.json",
    r"C:\Users\eljapo22\gephi\Node2Edge2JSON\feeders\combined_structure_TopLeft.json"
]

color_mapping = {
    'combined_structure_BottomLeft.json': 'lightblue',
    'combined_structure_TopLeft.json': 'orange',
    'combined_structure_Middle.json': 'purple',
}

graphs = {}
combined_data = {}

for file_path in file_paths:
    file_name = os.path.basename(file_path)
    combined_data[file_name] = load_combined_data(file_path)
    graphs[file_name] = create_graph_from_combined_data(combined_data[file_name])
    print_network_summary(combined_data[file_name])

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Combined Network Map"),
    dcc.Dropdown(
        id='level-dropdown',
        options=[{'label': f'Level {i}', 'value': i} for i in range(1, 8)],
        value=None,
        clearable=False
    ),
    html.Button('Update', id='update-button', n_clicks=0),
    dcc.Graph(id='graph', style={'height': '80vh'}, figure=create_figure(graphs, color_mapping)),
    html.Div(id='click-data', style={'margin-top': '20px'})
])

@app.callback(
    Output('graph', 'figure'),
    Input('update-button', 'n_clicks'),
    State('level-dropdown', 'value')
)
def update_graph(n_clicks, selected_level):
    return create_figure(graphs, color_mapping, selected_level)

@app.callback(
    Output('click-data', 'children'),
    Input('graph', 'clickData')
)
def display_click_data(clickData):
    if clickData is None:
        return "Double-click a transformer node to see its unique ID"
    
    point = clickData['points'][0]
    node_id = point['customdata']
    
    for file_name, data in combined_data.items():
        node_data = data['nodes'].get(node_id)
        if node_data and node_data['type'] == 'Transformer':
            return f"Transformer Unique ID: {node_id} (File: {file_name})"
    
    return "Double-click a transformer node to see its unique ID"

if __name__ == '__main__':
    print("Starting server...")
    app.run_server(debug=True)