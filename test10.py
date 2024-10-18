import json
import networkx as nx
import plotly.graph_objs as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, ALL
import numpy as np
from scipy.spatial import distance
import colorsys
import base64







COLOR_PALETTE = [
    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
    '#800000', '#008000', '#000080', '#808000', '#800080', '#008080',
    '#FFA500', '#FFC0CB', '#A52A2A', '#DEB887', '#5F9EA0',
    '#7FFF00', '#D2691E', '#FF7F50', '#6495ED', '#DC143C', '#00008B'
]








# Load JSON file
with open(r"C:\Users\eljapo22\gephi\Node2Edge2JSON\Node2Edge2JSON.json", 'r') as f:
    data = json.load(f)

# Create NetworkX graph
G = nx.Graph()

# Add nodes
for node in data['nodes']:
    G.add_node(node['id'], pos=(node['longitude'], node['latitude']), type=node['type'])

# Add edges
for edge in data['edges']:
    G.add_edge(edge['source'], edge['target'])

# Function to select transformer nodes
def select_transformer_nodes(G, num_transformers=100, num_sectors=4):
    nodes = list(G.nodes(data=True))
    positions = np.array([node[1]['pos'] for node in nodes])
    
    # Calculate centroid
    centroid = np.mean(positions, axis=0)
    
    # Assign sectors
    angles = np.arctan2(positions[:, 1] - centroid[1], positions[:, 0] - centroid[0])
    sector_size = 2 * np.pi / num_sectors
    sectors = (angles + np.pi) // sector_size
    
    transformers = []
    for sector in range(num_sectors):
        sector_nodes = [nodes[i] for i, s in enumerate(sectors) if s == sector]
        sector_transformers = max(1, int(num_transformers * len(sector_nodes) / len(nodes)))
        
        # Sort nodes by distance from centroid
        sector_nodes.sort(key=lambda n: distance.euclidean(n[1]['pos'], centroid))
        
        step = max(1, len(sector_nodes) // sector_transformers)
        for i in range(0, len(sector_nodes), step):
            candidates = sector_nodes[i:i+step]
            if candidates:
                # Select the node with highest degree among candidates
                transformer = max(candidates, key=lambda n: G.degree(n[0]))
                transformers.append(transformer[0])
                if len(transformers) == num_transformers:
                    return transformers
    
    # If we haven't selected enough transformers, add more from the most connected nodes
    if len(transformers) < num_transformers:
        remaining = sorted(set(G.nodes()) - set(transformers), key=lambda n: G.degree(n), reverse=True)
        transformers.extend(remaining[:num_transformers - len(transformers)])
    
    return transformers[:num_transformers]

# Select transformer nodes
transformer_nodes = select_transformer_nodes(G)

# Identify the substation node
substation_coords = (-79.3858484, 43.6608075)
substation_node = min(G.nodes(), key=lambda n: 
    ((G.nodes[n]['pos'][0] - substation_coords[0])**2 + 
     (G.nodes[n]['pos'][1] - substation_coords[1])**2))

# Find shortest paths from transformers to substation
paths = {}
for transformer in transformer_nodes:
    try:
        path = nx.shortest_path(G, transformer, substation_node)
        paths[transformer] = path
    except nx.NetworkXNoPath:
        print(f"No path found from transformer {transformer} to substation")

# Initialize clusters dictionary
clusters = {}

# Function to generate distinct colors
def generate_colors(n):
    HSV_tuples = [(x * 1.0 / n, 0.5, 0.5) for x in range(n)]
    return ['#%02x%02x%02x' % tuple(int(x*255) for x in colorsys.hsv_to_rgb(*hsv)) for hsv in HSV_tuples]

# Function to create figure
def create_figure(selected_clusters=None):
    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Create node trace
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_opacities = []
    node_text = []
    for node in G.nodes(data=True):
        x, y = node[1]['pos']
        node_x.append(x)
        node_y.append(y)
        
        if node[0] == substation_node:
            color = 'yellow'
            size = 20
            opacity = 1
        elif node[0] in transformer_nodes:
            color = 'red'
            size = 12
            opacity = 0.5
            for cluster_id, cluster_nodes in clusters.items():
                if node[0] in cluster_nodes and (selected_clusters is None or cluster_id in selected_clusters):
                    color = clusters[cluster_id]['color']
                    opacity = 1
                    break
        else:
            color = 'blue'
            size = 7
            opacity = 0.1
        
        node_colors.append(color)
        node_sizes.append(size)
        node_opacities.append(opacity)
        
        node_type = 'Substation' if node[0] == substation_node else ('Transformer' if node[0] in transformer_nodes else 'Regular')
        node_text.append(f"ID: {node[0]}<br>Type: {node_type}<br># of connections: {G.degree(node[0])}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            color=node_colors,
            size=node_sizes,
            opacity=node_opacities,
            line_width=2))

    # Create path traces
    path_traces = []
    if selected_clusters:
        for cluster_id in selected_clusters:
            cluster_color = clusters[cluster_id]['color']
            for node in clusters[cluster_id]['nodes']:
                if node in paths:
                    path = paths[node]
                    x, y = [], []
                    for path_node in path:
                        pos = G.nodes[path_node]['pos']
                        x.append(pos[0])
                        y.append(pos[1])
                    path_traces.append(go.Scatter(
                        x=x, y=y,
                        mode='lines',
                        line=dict(width=2, color=cluster_color),
                        opacity=0.7,
                        hoverinfo='none'
                    ))

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace] + path_traces,
                    layout=go.Layout(
                        title='Network Graph with Transformers and Paths to Substation',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 ) ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig







# Create Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='network-graph', figure=create_figure(), style={'height': '80vh'}),
    html.Div([
        html.P("Click and drag on the graph to use the lasso selection tool", style={'font-style': 'italic'}),
        html.Button('Reset Selection', id='reset-selection', n_clicks=0),
        html.Button('Clear Selection', id='clear-selection', n_clicks=0),
        html.Button('Save Cluster', id='save-cluster', n_clicks=0),
        html.Div("Save the current selection as a cluster", style={'font-size': '12px', 'color': 'gray'}),
        dcc.Input(id='cluster-name-input', type='text', placeholder='Enter cluster name'),
        dcc.Dropdown(id='cluster-dropdown', multi=True, placeholder='Select clusters'),
        html.Label("Select Color:"),
        html.Div([
            html.Button(
                style={
                    'backgroundColor': color,
                    'width': '30px',
                    'height': '30px',
                    'margin': '2px',
                    'border': '1px solid black',
                    'cursor': 'pointer'
                },
                id={'type': 'color-button', 'index': color}
            )
            for color in COLOR_PALETTE
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'maxWidth': '300px'}),
        html.Div([
            dcc.Input(id='color-picker', type='text', placeholder='Enter color (e.g., #FF0000)', value='#000000', style={'width': '200px'}),
            html.Div(id='color-preview', style={'width': '30px', 'height': '30px', 'border': '1px solid black', 'display': 'inline-block', 'marginLeft': '10px'})
        ]),
        html.Button('Save Configuration', id='save-config', n_clicks=0),
        html.Div("Save all clusters and settings to a file", style={'font-size': '12px', 'color': 'gray'}),
        html.Button('Load Configuration', id='load-config', n_clicks=0),
        dcc.Upload(id='upload-config', children=html.Button('Upload Configuration')),
        dcc.Download(id='download-config')
    ], style={'margin': '10px'}),
    html.Div(id='cluster-info', style={'margin-top': '20px', 'padding': '10px', 'background-color': '#f0f0f0'}),
    html.Div(id='node-info', style={'margin-top': '20px', 'padding': '10px', 'background-color': '#f0f0f0'})
])



@app.callback(
    Output('color-picker', 'value'),
    Output('network-graph', 'figure'),
    Output('cluster-dropdown', 'options'),
    Output('cluster-dropdown', 'value'),
    Input({'type': 'color-button', 'index': ALL}, 'n_clicks'),
    Input('reset-selection', 'n_clicks'),
    Input('clear-selection', 'n_clicks'),
    Input('save-cluster', 'n_clicks'),
    Input('cluster-dropdown', 'value'),
    Input('color-picker', 'value'),
    Input('upload-config', 'contents'),
    Input('load-config', 'n_clicks'),
    State('network-graph', 'selectedData'),
    State('cluster-name-input', 'value'),
    State({'type': 'color-button', 'index': ALL}, 'id'),
    State('upload-config', 'filename'),
    State('cluster-dropdown', 'options')
)
def update_graph_and_clusters(color_clicks, reset_clicks, clear_clicks, save_clicks, selected_clusters, color, 
                              upload_contents, load_clicks, selectedData, cluster_name, color_button_ids, upload_filename, available_clusters):
    global clusters, transformer_nodes, substation_node, paths
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, create_figure(), [], None

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if 'color-button' in button_id:
        color = json.loads(button_id)['index']
        return color, dash.no_update, dash.no_update, dash.no_update
    
    if button_id == 'reset-selection':
        return dash.no_update, create_figure(), [{'label': k, 'value': k} for k in clusters.keys()], None
    elif button_id == 'clear-selection':
        return dash.no_update, create_figure(), [{'label': k, 'value': k} for k in clusters.keys()], None
    elif button_id == 'save-cluster':
        if selectedData and cluster_name:
            selected_nodes = [point['text'].split('<br>')[0].split(': ')[1] for point in selectedData['points']]
            selected_transformers = [node for node in selected_nodes if node in transformer_nodes]
            if selected_transformers:
                clusters[cluster_name] = {
                    'nodes': selected_transformers,
                    'color': color or '#000000'
                }
                if selected_clusters is None:
                    selected_clusters = [cluster_name]
                elif cluster_name not in selected_clusters:
                    selected_clusters.append(cluster_name)
        elif selected_clusters:
            for cluster_id in selected_clusters:
                if cluster_id in clusters:
                    clusters[cluster_id]['color'] = color
        return dash.no_update, create_figure(selected_clusters), [{'label': k, 'value': k} for k in clusters.keys()], selected_clusters
    elif button_id == 'cluster-dropdown':
        if len(selected_clusters) < len(available_clusters):
            removed_cluster = next(cluster for cluster in available_clusters if cluster['value'] not in selected_clusters)
            del clusters[removed_cluster['value']]
            updated_options = [cluster for cluster in available_clusters if cluster['value'] in selected_clusters]
            return dash.no_update, create_figure(selected_clusters), updated_options, selected_clusters
        return dash.no_update, create_figure(selected_clusters), [{'label': k, 'value': k} for k in clusters.keys()], selected_clusters
    elif button_id == 'color-picker' and selected_clusters:
        for cluster_id in selected_clusters:
            if cluster_id in clusters:
                clusters[cluster_id]['color'] = color
        return dash.no_update, create_figure(selected_clusters), [{'label': k, 'value': k} for k in clusters.keys()], selected_clusters
    elif button_id == 'upload-config' or button_id == 'load-config':
        if upload_contents is not None:
            content_type, content_string = upload_contents.split(',')
            decoded = base64.b64decode(content_string)
            config = json.loads(decoded.decode('utf-8'))
            
            clusters = config['clusters']
            transformer_nodes = set(config['transformer_nodes'])
            substation_node = config['substation_node']
            paths = {eval(k): v for k, v in config['paths'].items()}
            
            return dash.no_update, create_figure(), [{'label': k, 'value': k} for k in clusters.keys()], list(clusters.keys())
    
    return dash.no_update, create_figure(selected_clusters), [{'label': k, 'value': k} for k in clusters.keys()], selected_clusters

@app.callback(
    Output('color-preview', 'style'),
    Input('color-picker', 'value')
)
def update_color_preview(color):
    return {'backgroundColor': color, 'width': '30px', 'height': '30px', 'border': '1px solid black', 'display': 'inline-block', 'marginLeft': '10px'}

@app.callback(
    Output('node-info', 'children'),
    Input('network-graph', 'clickData'),
    State('network-graph', 'clickData')
)
def display_node_info(clickData, prevClickData):
    if clickData is None:
        return "Double-click a node to display its information."
    
    if clickData == prevClickData:  # Double click detected
        point = clickData['points'][0]
        node_id = point['text'].split('<br>')[0].split(': ')[1]
        node_data = next((node for node in data['nodes'] if node['id'] == node_id), None)
        
        if node_data:
            node_type = 'Substation' if node_id == substation_node else ('Transformer' if node_id in transformer_nodes else node_data['type'])
            info = f"""
            Node ID: {node_data['id']}
            Type: {node_type}
            Longitude: {node_data['longitude']}
            Latitude: {node_data['latitude']}
            Bounding Boxes: {', '.join(node_data['boundingBoxes'])}
            """
            return html.Pre(info)
    
    return "Double-click a node to display its information."

@app.callback(
    Output('cluster-info', 'children'),
    Input('cluster-dropdown', 'value')
)
def update_cluster_info(selected_clusters):
    if not selected_clusters:
        return "No clusters selected"
    
    info = []
    for cluster_id in selected_clusters:
        if cluster_id in clusters:
            cluster = clusters[cluster_id]
            info.append(html.Div([
                html.H4(f"Cluster: {cluster_id}"),
                html.P(f"Number of nodes: {len(cluster['nodes'])}"),
                html.P(f"Color: {cluster['color']}"),
                html.P(f"Nodes: {', '.join(cluster['nodes'][:10])}{'...' if len(cluster['nodes']) > 10 else ''}")
            ]))
    
    return info

@app.callback(
    Output('download-config', 'data'),
    Input('save-config', 'n_clicks'),
    State('cluster-dropdown', 'options'),
    prevent_initial_call=True
)
def save_configuration(n_clicks, cluster_options):
    if n_clicks > 0:
        config = {
            'clusters': {},
            'transformer_nodes': list(transformer_nodes),
            'substation_node': substation_node,
            'paths': {str(k): v for k, v in paths.items()}
        }
        
        for cluster_option in cluster_options:
            cluster_name = cluster_option['value']
            if cluster_name in clusters:
                config['clusters'][cluster_name] = {
                    'nodes': clusters[cluster_name]['nodes'],
                    'color': clusters[cluster_name]['color']
                }
        
        return dict(content=json.dumps(config, indent=2), filename="network_config.json")
    
    return dash.no_update



if __name__ == '__main__':
    app.run_server(debug=True)