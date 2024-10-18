import networkx as nx
import json
import plotly.graph_objects as go
import dash
import random
import hashlib
import pickle
from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import math
import os
import logging
from shapely.geometry import Point, Polygon
import numpy as np
import dash_daq as daq
from dash_extensions import Keyboard

# 2. Global variable declarations
global G, transformer_nodes, highlight_mode, stored_meter_nodes, stored_transformer_nodes
global undo_redo_store, is_placing_meter_mode, is_placing_transformer_mode, is_delete_mode, lasso_store, meter_connections
global cluster_info

# Initialize global variables
G = None
transformer_nodes = None
highlight_mode = True
stored_meter_nodes = []
stored_transformer_nodes = []
undo_redo_store = {'undo': [], 'redo': []}
is_placing_meter_mode = False
is_placing_transformer_mode = False
is_delete_mode = False
lasso_store = []
meter_connections = {
    961011153: [961011152, 31258725, 3252904691],
    502815960: [71723092, 11359207136, 9008674764]
}

cluster_info = {}  # Initialize as an empty dictionary


# Define IDs for the special nodes
ds_substation_aa_id = 24959509



logging.basicConfig(level=logging.DEBUG)



def get_existing_cluster_colors():
    colors = set()
    if G is not None:
        for node, data in G.nodes(data=True):
            if data.get('is_transformer'):
                colors.add(data.get('color'))
    return list(colors) if colors else ['red', 'blue', 'green', 'yellow', 'purple', 'orange']



def generate_graph_id(G):
    graph_data = sorted((node, sorted(neighbors)) for node, neighbors in G.adjacency())
    graph_str = str(graph_data).encode('utf-8')
    return hashlib.md5(graph_str).hexdigest()





    




# Add this new function after get_existing_cluster_colors() and before other main functions



def assign_transformer_label(transformer_id, color):
    if color not in cluster_info:
        cluster_info[color] = {'transformers': [], 'next_label': 'AA'}
    
    if transformer_id not in cluster_info[color]['transformers']:
        label = f"DS TF {cluster_info[color]['next_label']}"
        cluster_info[color]['transformers'].append(transformer_id)
        
        # Update to next label
        current_label = cluster_info[color]['next_label']
        if current_label == 'ZZ':
            # Reset to 'AA' if we've reached 'ZZ'
            cluster_info[color]['next_label'] = 'AA'
        else:
            # Increment both letters
            next_label = chr(ord(current_label[0]) + 1) + chr(ord(current_label[1]) + 1)
            cluster_info[color]['next_label'] = next_label
    else:
        # If transformer already exists, find its label
        index = cluster_info[color]['transformers'].index(transformer_id)
        label = f"DS TF {chr(ord('A') + index // 26)}{chr(ord('A') + index % 26)}"
    
    return label



















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


def hide_node(G, node_id):
    if node_id in G.nodes:
        G.nodes[node_id]['hidden'] = True



# Add 100 transformer nodes
# Modify add_transformer_nodes function
def add_transformer_nodes(G, num_transformers=100):
    graph_id = generate_graph_id(G)
    transformer_file = f'transformer_nodes_{graph_id}.pkl'
    
    try:
        if os.path.exists(transformer_file):
            with open(transformer_file, 'rb') as f:
                transformer_nodes = pickle.load(f)
            logging.info(f"Loaded transformer nodes from {transformer_file}")
        else:
            random.seed(graph_id)
            all_nodes = list(G.nodes())
            transformer_nodes = random.sample(all_nodes, num_transformers)
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(transformer_file), exist_ok=True)
            
            with open(transformer_file, 'wb') as f:
                pickle.dump(transformer_nodes, f)
            logging.info(f"Created and saved new transformer nodes to {transformer_file}")
        
        default_colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        existing_colors = get_existing_cluster_colors()
        colors_to_use = existing_colors if existing_colors else default_colors








        for node in transformer_nodes:
            G.nodes[node]['is_transformer'] = True
            if 'color' not in G.nodes[node]:
                color = random.choice(colors_to_use)
                G.nodes[node]['color'] = color
            else:
                color = G.nodes[node]['color']
            label = assign_transformer_label(node, color)
            G.nodes[node]['label'] = label
        
        return transformer_nodes












    
    except Exception as e:
        logging.error(f"Error in add_transformer_nodes: {str(e)}")
        return []







def add_meter_nodes(G, transformer_nodes, num_meter_per_transformer=3, min_distance=0.0005):
    new_G = G.copy()
    processed_edges = set()

    for transformer in transformer_nodes:
        connected_edges = list(new_G.edges(transformer))
        random.shuffle(connected_edges)  # Randomize edge selection
        meter_count = 0

        for edge in connected_edges:
            if meter_count >= num_meter_per_transformer:
                break
            if edge in processed_edges or tuple(reversed(edge)) in processed_edges:
                continue
            
            pos1 = new_G.nodes[edge[0]]['pos']
            pos2 = new_G.nodes[edge[1]]['pos']
            edge_length = math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
            
            for _ in range(20):  # Try 20 times to place a node
                t = random.uniform(0.2, 0.8)  # Place node between 20% and 80% along the edge
                px = pos1[0] + t * (pos2[0] - pos1[0])
                py = pos1[1] + t * (pos2[1] - pos1[1])
                
                too_close = any(math.sqrt((px-node['pos'][0])**2 + (py-node['pos'][1])**2) < min_distance 
                                for node in new_G.nodes.values())
                
                if not too_close:
                    node_id = max(new_G.nodes()) + 1
                    new_G.add_node(node_id, pos=(px, py), is_meter=True)
                    new_G.add_edge(edge[0], node_id)
                    new_G.add_edge(node_id, edge[1])
                    new_G.remove_edge(edge[0], edge[1])
                    processed_edges.add(edge)
                    meter_count += 1
                    break
    
    return new_G









def initialize_graph():
    global G, transformer_nodes, cluster_info, meter_connections
    G = load_graph_from_cache()
    transformer_nodes = add_transformer_nodes(G)
    
    # Add meter nodes for transformers not in meter_connections
    transformers_to_process = set(transformer_nodes) - set(meter_connections.keys())
    G = add_meter_nodes(G, transformers_to_process)
    
    # Update meter_connections with new meter nodes
    for node, data in G.nodes(data=True):
        if data.get('is_meter'):
            for transformer in G.neighbors(node):
                if G.nodes[transformer].get('is_transformer'):
                    if transformer not in meter_connections:
                        meter_connections[transformer] = []
                    if node not in meter_connections[transformer]:
                        meter_connections[transformer].append(node)
    
    # Add hardcoded meter nodes and set colors
    for transformer, meters in meter_connections.items():
        if transformer in G.nodes:
            G.nodes[transformer]['color'] = 'green'  # Set a default color
            for meter in meters:
                if meter not in G.nodes:
                    # Use the transformer's position as a fallback
                    new_pos = G.nodes[transformer]['pos']
                    G.add_node(meter, pos=new_pos, is_meter=True)
                    G.add_edge(transformer, meter)
                    logging.info(f"Added missing meter node {meter} to graph at position {new_pos}")
                G.nodes[meter]['is_meter'] = True
        else:
            logging.warning(f"Transformer {transformer} not found in graph")
    
    return G, transformer_nodes, cluster_info
    




        



















































def create_figure(highlight_paths=False, stored_meter_nodes=None, stored_transformer_nodes=None, lasso_selected=None, transformer_colors=None):
    print("Creating figure...")
    global G, ds_substation_aa_id, meter_connections

   

    logging.debug(f"Creating figure with highlight_paths={highlight_paths}")
    logging.debug(f"Number of nodes in G: {len(G.nodes())}")
    logging.debug(f"Number of transformer nodes: {sum(1 for n in G.nodes if G.nodes[n].get('is_transformer'))}")
    logging.debug(f"Number of meter nodes: {sum(1 for n in G.nodes if G.nodes[n].get('is_meter'))}")
    logging.debug(f"Substation node {ds_substation_aa_id} exists: {ds_substation_aa_id in G.nodes}")
    logging.debug(f"meter_connections: {meter_connections}")


    fig = go.Figure()
    
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

    node_x, node_y, node_text, node_color, node_size, node_symbol = [], [], [], [], [], []
    substation_x, substation_y = None, None

    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        if node == ds_substation_aa_id:
            substation_x, substation_y = x, y
            continue
        node_x.append(x)
        node_y.append(y)
        if G.nodes[node].get('is_transformer'):
            node_text.append(f"{node}<br>{G.nodes[node].get('label', '')}")  # Add label to hover text
            node_color.append(G.nodes[node].get('color', 'green'))
            node_size.append(17)  # Larger size for transformers
            node_symbol.append('triangle-up')

        elif G.nodes[node].get('is_meter'):
            logging.debug(f"Processing meter node: {node}")
            associated_transformer = None
            for transformer, meters in meter_connections.items():
                if isinstance(meters, list) and len(meters) > 0:
                    if str(node) in meters or int(node) in meters:
                        associated_transformer = transformer
                        break

            if associated_transformer and associated_transformer in G.nodes:
                transformer_label = G.nodes[associated_transformer].get('label', '')
                node_text.append(f"{node}<br>Meter<br>T/F: {transformer_label}")
                node_color.append(G.nodes[associated_transformer].get('color', 'yellow'))
            else:
                node_text.append(f"{node}<br>Meter")
                node_color.append('yellow')  # default color if no associated transformer found
            node_size.append(12)
            node_symbol.append('circle')
        else:
            node_text.append(str(node))
            node_color.append('lightgrey')
            node_size.append(3)
            node_symbol.append('circle')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        hovertext=node_text,
        text=node_text,  # Add this line
        customdata=node_text,  # And this line
        marker=dict(
            color=node_color,
            size=node_size,
            symbol=node_symbol,
            line_width=2
        )
    )













    # Create a separate trace for the substation
    if substation_x is not None and substation_y is not None:
        substation_trace = go.Scatter(
            x=[substation_x], y=[substation_y],
            mode='markers',
            hoverinfo='text',
            text=['Substation'],
            marker=dict(
                color='red',
                size=15,
                symbol='square',
                line_width=2
            )
        )

    # Add traces to the figure
    fig.add_trace(edge_trace)
    fig.add_trace(node_trace)
    
    if highlight_paths:
        logging.debug("Highlighting paths...")
        for node in G.nodes():
            if G.nodes[node].get('is_transformer'):
                logging.debug(f"Processing transformer node: {node}")

                try:
                    path = nx.shortest_path(G, source=node, target=ds_substation_aa_id)
                    logging.debug(f"Path for transformer {node}: {path}")
                    path_x = []
                    path_y = []
                    for path_node in path:
                        x, y = G.nodes[path_node]['pos']
                        path_x.append(x)
                        path_y.append(y)
                    fig.add_trace(go.Scatter(
                        x=path_x, y=path_y,
                        mode='lines',
                        line=dict(color=G.nodes[node].get('color', 'green'), width=2),
                        hoverinfo='none'
                    ))
                except nx.NetworkXNoPath:
                    logging.warning(f"No path found from transformer {node} to substation {ds_substation_aa_id}")
                except Exception as e:
                    logging.error(f"Error highlighting path for transformer {node}: {str(e)}")

    # Add dotted lines for meter connections
    for transformer, meters in meter_connections.items():
        transformer_color = G.nodes[transformer].get('color', 'green')
        for meter in meters:
            path = nx.shortest_path(G, source=transformer, target=meter)
            path_x, path_y = [], []
            for node in path:
                x, y = G.nodes[node]['pos']
                path_x.append(x)
                path_y.append(y)
            fig.add_trace(go.Scatter(
                x=path_x, y=path_y,
                mode='lines',
                line=dict(color=transformer_color, width=1, dash='dot'),
                showlegend=False
            ))

    # Highlight lasso selected area
    if lasso_selected and isinstance(lasso_selected, dict) and 'x' in lasso_selected and 'y' in lasso_selected:
        fig.add_trace(go.Scatter(
            x=lasso_selected['x'],
            y=lasso_selected['y'],
            mode='lines',
            line=dict(color='rgba(255, 0, 0, 0.5)', width=2),
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.1)',
            showlegend=False
        ))

    # Add substation trace last to ensure it's on top
    if substation_x is not None and substation_y is not None:
        fig.add_trace(substation_trace)

    # Set the layout
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    return fig

# Specify the exact path to your existing app_state.json file
APP_STATE_PATH = r"C:\Users\eljapo22\gephi\app_state.json"

def save_state(nodes, highlight_mode, lasso_selections, transformer_colors, meter_connections, cluster_info):
    state = {
        'meter_nodes': nodes,
        'highlight_mode': highlight_mode,
        'lasso_selections': lasso_selections,
        'transformer_colors': transformer_colors,
        'meter_connections': meter_connections,
        'cluster_info' : cluster_info
        
    }
    with open(APP_STATE_PATH, 'w') as f:
        json.dump(state, f)

def load_state():
    logging.info(f"Attempting to load state from {APP_STATE_PATH}")
    if os.path.exists(APP_STATE_PATH):
        try:
            with open(APP_STATE_PATH, 'r') as f:
                state = json.load(f)
            global cluster_info
            cluster_info = state.get('cluster_info', {})    
            logging.info("State loaded successfully")
            logging.info(f"Loaded state: {state}")
            return (
                state.get('meter_nodes', []),
                state.get('highlight_mode', False),
                state.get('lasso_selections', []),
                state.get('transformer_colors', {}),
                state.get('meter_connections', {}),
                state.get('cluster_info', {}),
            )
        except json.JSONDecodeError:
            logging.error("Error decoding JSON from app_state.json")
        except Exception as e:
            logging.error(f"Error loading state: {str(e)}")
    else:
        logging.warning(f"app_state.json not found at {APP_STATE_PATH}")
    state.get('cluster_info', {}),

# Load any saved state
initial_meter_nodes, initial_highlight_mode, initial_lasso_selections, initial_transformer_colors, loaded_meter_connections, loaded_cluster_info = load_state()

# Call initialization function
initialize_graph()

for node, color in initial_transformer_colors.items():
    if int(node) in G.nodes and G.nodes[int(node)].get('is_transformer'):
        G.nodes[int(node)]['color'] = color
        # Update cluster_info and assign label
        label = assign_transformer_label(int(node), color)
        G.nodes[int(node)]['label'] = label





    # Update cluster_info if loaded
if loaded_cluster_info:
    cluster_info = loaded_cluster_info
else:
    # If no cluster_info was loaded, initialize it based on the current graph state
    existing_colors = get_existing_cluster_colors()
    cluster_info = {color: {'transformers': [], 'next_label': 'AA'} for color in existing_colors}
    for node, data in G.nodes(data=True):
        if data.get('is_transformer'):
            color = data.get('color', 'default')
            if color not in cluster_info:
                cluster_info[color] = {'transformers': [], 'next_label': 'AA'}
            cluster_info[color]['transformers'].append(node)
    
    # Ensure all transformers have labels
    for node, data in G.nodes(data=True):
        if data.get('is_transformer') and 'label' not in data:
            color = data.get('color', 'default')
            label = assign_transformer_label(node, color)
            G.nodes[node]['label'] = label


    # Add this debug logging
    for node, data in G.nodes(data=True):
        if data.get('is_meter'):
            logging.debug(f"Meter node {node}: {data}")






# Define the layout of the Dash app
app = Dash(__name__)

app.layout = html.Div([
    daq.ColorPicker(
        id='node-color-picker',
        label='Select Node Color',
                value=dict(hex='#1E90FF')
    ),
    html.H1("Network Map with Clickable Nodes and Path Finding"),
    dcc.Input(id='start-node', type='text', placeholder='Start Node ID'),
    dcc.Input(id='end-node', type='text', placeholder='End Node ID'),
    html.Button('Find Path', id='find-path-button'),
    html.Button('Place Meter Node', id='place-meter-node-button'),
    html.Button('Place Transformer Node', id='place-transformer-node-button'),
    html.Button('Delete Node', id='delete-node-button'),
    html.Button('Highlight Meter to Transformer Paths', id='highlight-paths-button'),
    html.Button('Reset Meter Nodes', id='reset-meter-nodes-button'),
    html.Button('Reset Transformer Nodes', id='reset-transformer-nodes-button'),
    html.Button('Enable Lasso', id='lasso-button'),
    html.Button('Reset Lasso', id='reset-lasso-button'),
    html.Button('Undo', id='undo-button'),
    html.Button('Redo', id='redo-button'),
    html.Button('Save Clusters', id='save-clusters-button'),
    html.Button('Load Clusters', id='load-clusters-button'),
    html.Button('Export Data', id='export-data-button'),
    daq.ColorPicker(
        id='color-picker',
        label='Select Color',
        value=dict(hex='#FF0000')
    ),
    dcc.Dropdown(
        id='selection-mode',
        options=[
            {'label': 'Lasso', 'value': 'lasso'},
            {'label': 'Rectangle', 'value': 'rect'},
            {'label': 'Circle', 'value': 'circle'}
        ],
        value='lasso',
        clearable=False
    ),
    dcc.Graph(
        id='graph',
        figure=create_figure(highlight_paths=True, stored_meter_nodes=initial_meter_nodes, transformer_colors={}),
        config={
            'modeBarButtonsToAdd': ['drawclosedpath', 'eraseshape'],
            'scrollZoom': True,
            'doubleClick': 'reset+autosize'
        },
        style={'height': '80vh'}
    ),
    html.Div(id='node-info', style={'whiteSpace': 'pre-line'}),
    html.Div(id='path-info', style={'whiteSpace': 'pre-line'}),
    html.Div(id='lasso-info', style={'whiteSpace': 'pre-line'}),
    html.Div(id='locked-node-info', style={'whiteSpace': 'pre-line', 'fontWeight': 'bold'}),
    dcc.Store(id='meter-node-mode', data=False),
    dcc.Store(id='transformer-node-mode', data=False),
    dcc.Store(id='delete-node-mode', data=False),
    dcc.Store(id='meter-nodes-store', data=initial_meter_nodes),
    dcc.Store(id='transformer-nodes-store', data=[]),
    dcc.Store(id='highlight-paths-mode', data=initial_highlight_mode),
    dcc.Store(id='lasso-store', data=[]),
    dcc.Store(id='undo-redo-store', data={'undo': [], 'redo': []}),
    dcc.Store(id='zoom-pan-lock', data={'locked': False}),
    Keyboard(id="keyboard")
])

def update_node_colors(selected_ids, color):
    changed_nodes = []
    for node_id in selected_ids:
        if G.nodes[node_id].get('is_transformer'):
            old_color = G.nodes[node_id].get('color')
            if old_color != color:
                # Remove from old color cluster if exists
                if old_color in cluster_info:
                    cluster_info[old_color]['transformers'].remove(node_id)
                
                # Assign new label for new color
                new_label = assign_transformer_label(node_id, color)
                G.nodes[node_id]['label'] = new_label
                G.nodes[node_id]['color'] = color
                changed_nodes.append(node_id)
    return changed_nodes

def handle_lasso_selection(selectedData, selection_mode, node_color):
    global stored_meter_nodes, stored_transformer_nodes, undo_redo_store, is_placing_meter_mode, is_placing_transformer_mode, is_delete_mode, highlight_mode, lasso_store

    logging.debug(f"Handling lasso selection: {selectedData}")
    selected_points = selectedData['points']
    selected_ids = [int(point['text']) for point in selected_points]
    changed_nodes = update_node_colors(selected_ids, node_color['hex'])
    new_fig = create_figure(highlight_paths=highlight_mode, 
                            stored_meter_nodes=stored_meter_nodes, 
                            stored_transformer_nodes=stored_transformer_nodes)
    if changed_nodes:
        undo_redo_store['undo'].append(('color_nodes', changed_nodes, node_color['hex']))
        undo_redo_store['redo'] = []
    return new_fig, stored_meter_nodes, stored_transformer_nodes, is_placing_meter_mode, is_placing_transformer_mode, is_delete_mode, highlight_mode, f"{len(changed_nodes)} transformer nodes colored", lasso_store, undo_redo_store

def handle_node_deletion(selectedData):
    global G, stored_meter_nodes, stored_transformer_nodes, undo_redo_store, is_placing_meter_mode, is_placing_transformer_mode, is_delete_mode, highlight_mode, lasso_store

    logging.debug(f"Handling node deletion: {selectedData}")
    deleted_node = int(selectedData['points'][0]['text'])
    if deleted_node in G.nodes:
        node_data = G.nodes[deleted_node].copy()
        G.remove_node(deleted_node)
        stored_meter_nodes = [node for node in stored_meter_nodes if node['id'] != str(deleted_node)]
        stored_transformer_nodes = [node for node in stored_transformer_nodes if node['id'] != str(deleted_node)]
        undo_redo_store['undo'].append(('delete_node', deleted_node, node_data))
        undo_redo_store['redo'] = []
        new_fig = create_figure(highlight_paths=highlight_mode, 
                                stored_meter_nodes=stored_meter_nodes, 
                                stored_transformer_nodes=stored_transformer_nodes)
        return new_fig, stored_meter_nodes, stored_transformer_nodes, is_placing_meter_mode, is_placing_transformer_mode, is_delete_mode, highlight_mode, f"Node {deleted_node} deleted", lasso_store, undo_redo_store
    return dash.no_update

@app.callback(
    Output('graph', 'figure'),
    Output('meter-nodes-store', 'data'),
    Output('transformer-nodes-store', 'data'),
    Output('meter-node-mode', 'data'),
    Output('transformer-node-mode', 'data'),
    Output('delete-node-mode', 'data'),
    Output('highlight-paths-mode', 'data'),
    Output('path-info', 'children'),
    Output('lasso-store', 'data'),
    Output('undo-redo-store', 'data'),
    Input('node-color-picker', 'value'),
    Input('graph', 'selectedData'),
    Input('place-meter-node-button', 'n_clicks'),
    Input('place-transformer-node-button', 'n_clicks'),
    Input('delete-node-button', 'n_clicks'),
    Input('find-path-button', 'n_clicks'),
    Input('highlight-paths-button', 'n_clicks'),
    Input('reset-meter-nodes-button', 'n_clicks'),
    Input('reset-transformer-nodes-button', 'n_clicks'),
    Input('lasso-button', 'n_clicks'),
    Input('reset-lasso-button', 'n_clicks'),
    Input('undo-button', 'n_clicks'),
    Input('redo-button', 'n_clicks'),
    Input('save-clusters-button', 'n_clicks'),
    Input('load-clusters-button', 'n_clicks'),
    Input('export-data-button', 'n_clicks'),
    Input('color-picker', 'value'),
    Input('keyboard', 'n_keydowns'),
    State('find-path-button', 'id'),
    State('place-meter-node-button', 'id'),
    State('meter-nodes-store', 'data'),
    State('transformer-nodes-store', 'data'),
    State('meter-node-mode', 'data'),
    State('transformer-node-mode', 'data'),
    State('delete-node-mode', 'data'),
    State('highlight-paths-mode', 'data'),
    State('start-node', 'value'),
    State('end-node', 'value'),
    State('lasso-store', 'data'),
    State('undo-redo-store', 'data'),
    State('zoom-pan-lock', 'data'),
    State('keyboard', 'keydown'),
    State('selection-mode', 'value')
)


def update_graph(
    node_color, selectedData, 
    place_meter_n_clicks, place_transformer_n_clicks, delete_n_clicks, 
    find_n_clicks, highlight_n_clicks, reset_meter_n_clicks, reset_transformer_n_clicks, 
    lasso_n_clicks, reset_lasso_n_clicks, undo_n_clicks, redo_n_clicks,
    save_clusters_n_clicks, load_clusters_n_clicks, export_data_n_clicks, 
    color_value, n_keydowns,
    find_path_button_id, place_meter_node_button_id,
    stored_meter_nodes, stored_transformer_nodes, 
    is_placing_meter_mode, is_placing_transformer_mode, is_delete_mode, 
    is_highlight_mode, start_node, end_node, 
    lasso_store, undo_redo_store, zoom_pan_lock, last_key_pressed, selection_mode
):
    
    
    global G, transformer_nodes, meter_connections



    print(f"Callback triggered. Button ID: {button_id}")
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    try:
        # Handle keyboard shortcuts
        if button_id == 'keyboard':
            if last_key_pressed == 'l':
                button_id = 'lasso-button'
            elif last_key_pressed == 'u':
                button_id = 'undo-button'
            elif last_key_pressed == 'r':
                button_id = 'redo-button'
        
        if button_id == 'place-meter-node-button':
            print("Placing meter node...")
            return create_figure(highlight_paths=is_highlight_mode, stored_meter_nodes=stored_meter_nodes, stored_transformer_nodes=stored_transformer_nodes), stored_meter_nodes, stored_transformer_nodes, not is_placing_meter_mode, False, False, is_highlight_mode, "Meter node placement mode " + ("activated" if not is_placing_meter_mode else "deactivated"), lasso_store, undo_redo_store
            print(f"Meter node placed. New stored_meter_nodes: {stored_meter_nodes}")

        if button_id == 'place-transformer-node-button':
            
            return create_figure(highlight_paths=is_highlight_mode, stored_meter_nodes=stored_meter_nodes, stored_transformer_nodes=stored_transformer_nodes), stored_meter_nodes, stored_transformer_nodes, False, not is_placing_transformer_mode, False, is_highlight_mode, "Transformer node placement mode " + ("activated" if not is_placing_transformer_mode else "deactivated"), lasso_store, undo_redo_store
        
        if button_id == 'delete-node-button':
            return create_figure(highlight_paths=is_highlight_mode, stored_meter_nodes=stored_meter_nodes, stored_transformer_nodes=stored_transformer_nodes), stored_meter_nodes, stored_transformer_nodes, False, False, not is_delete_mode, is_highlight_mode, "Delete node mode " + ("activated" if not is_delete_mode else "deactivated"), lasso_store, undo_redo_store    
        
        if button_id == 'highlight-paths-button':
            is_highlight_mode = not is_highlight_mode
            new_fig = create_figure(highlight_paths=is_highlight_mode, stored_meter_nodes=stored_meter_nodes, stored_transformer_nodes=stored_transformer_nodes)
            save_state(stored_meter_nodes, is_highlight_mode, lasso_store, {node: data['color'] for node, data in G.nodes(data=True) if data.get('is_transformer')}, meter_connections)
            return new_fig, stored_meter_nodes, stored_transformer_nodes, is_placing_meter_mode, is_placing_transformer_mode, is_delete_mode, is_highlight_mode, "Paths " + ("highlighted" if is_highlight_mode else "hidden"), lasso_store, undo_redo_store
        
        if button_id == 'graph' and selectedData:
            if selection_mode in ['rect', 'circle', 'lasso']:
                return handle_lasso_selection(selectedData, selection_mode, node_color)
            elif is_delete_mode:
                return handle_node_deletion(selectedData)
        
        if button_id == 'reset-meter-nodes-button':
            new_fig = create_figure(highlight_paths=is_highlight_mode, stored_meter_nodes=[], stored_transformer_nodes=stored_transformer_nodes)
            save_state([], is_highlight_mode, lasso_store, {node: data['color'] for node, data in G.nodes(data=True) if data.get('is_transformer')}, meter_connections)
            undo_redo_store['undo'].append(('reset_meter', stored_meter_nodes))
            undo_redo_store['redo'] = []
            return new_fig, [], stored_transformer_nodes, False, False, False, is_highlight_mode, "Meter nodes reset", lasso_store, undo_redo_store
        
        if button_id == 'reset-transformer-nodes-button':
            new_fig = create_figure(highlight_paths=is_highlight_mode, stored_meter_nodes=stored_meter_nodes, stored_transformer_nodes=[])
            undo_redo_store['undo'].append(('reset_transformer', stored_transformer_nodes))
            undo_redo_store['redo'] = []
            return new_fig, stored_meter_nodes, [], False, False, False, is_highlight_mode, "Transformer nodes reset", lasso_store, undo_redo_store
        
        if button_id == 'find-path-button':
            if start_node is None or end_node is None:
                return dash.no_update, stored_meter_nodes, stored_transformer_nodes, is_placing_meter_mode, is_placing_transformer_mode, is_delete_mode, is_highlight_mode, "Enter start and end nodes and click 'Find Path'", lasso_store, undo_redo_store
            
            try:
                path = nx.shortest_path(G, source=int(start_node), target=int(end_node))
                
                path_x, path_y = [], []
                for node in path:
                    x, y = G.nodes[node]['pos']
                    path_x.append(x)
                    path_y.append(y)
                
                new_fig = create_figure(highlight_paths=is_highlight_mode, stored_meter_nodes=stored_meter_nodes, stored_transformer_nodes=stored_transformer_nodes)
                new_fig.add_trace(go.Scatter(
                    x=path_x, y=path_y,
                    mode='lines',
                    line=dict(color='red', width=2),
                    name='Path'
                ))
                
                start_x, start_y = G.nodes[int(start_node)]['pos']
                end_x, end_y = G.nodes[int(end_node)]['pos']
                
                new_fig.add_trace(go.Scatter(
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
                ))
                return new_fig, stored_meter_nodes, stored_transformer_nodes, is_placing_meter_mode, is_placing_transformer_mode, is_delete_mode, is_highlight_mode, f"Path found: {' -> '.join(map(str, path))}", lasso_store, undo_redo_store
            except nx.NetworkXNoPath:
                return dash.no_update, stored_meter_nodes, stored_transformer_nodes, is_placing_meter_mode, is_placing_transformer_mode, is_delete_mode, is_highlight_mode, "No path found between the specified nodes.", lasso_store, undo_redo_store
            except ValueError:
                return dash.no_update, stored_meter_nodes, stored_transformer_nodes, is_placing_meter_mode, is_placing_transformer_mode, is_delete_mode, is_highlight_mode, "Please enter valid node IDs.", lasso_store, undo_redo_store
        
        if button_id == 'lasso-button':
            return dash.no_update, stored_meter_nodes, stored_transformer_nodes, False, False, False, is_highlight_mode, "Lasso selection enabled", lasso_store, undo_redo_store

        if button_id == 'reset-lasso-button':
            new_fig = create_figure(highlight_paths=is_highlight_mode, stored_meter_nodes=stored_meter_nodes, stored_transformer_nodes=stored_transformer_nodes)
            return new_fig, stored_meter_nodes, stored_transformer_nodes, is_placing_meter_mode, is_placing_transformer_mode, is_delete_mode, is_highlight_mode, "Lasso selection reset", [], undo_redo_store

        if button_id == 'undo-button':
            if undo_redo_store['undo']:
                action = undo_redo_store['undo'].pop()
                undo_redo_store['redo'].append(action)
                if action[0] == 'color_nodes':
                    for node in action[1]:
                        G.nodes[node]['color'] = G.nodes[node].get('original_color', 'green')
                elif action[0] == 'delete_node':
                    G.add_node(action[1], **action[2])
                elif action[0] in ['reset_meter', 'reset_transformer']:
                    if action[0] == 'reset_meter':
                        stored_meter_nodes = action[1]
                    else:
                        stored_transformer_nodes = action[1]
                new_fig = create_figure(highlight_paths=is_highlight_mode, stored_meter_nodes=stored_meter_nodes, stored_transformer_nodes=stored_transformer_nodes)
                return new_fig, stored_meter_nodes, stored_transformer_nodes, is_placing_meter_mode, is_placing_transformer_mode, is_delete_mode, is_highlight_mode, "Undo successful", lasso_store, undo_redo_store
            return dash.no_update, stored_meter_nodes, stored_transformer_nodes, is_placing_meter_mode, is_placing_transformer_mode, is_delete_mode, is_highlight_mode, "Nothing to undo", lasso_store, undo_redo_store

        if button_id == 'redo-button':
            if undo_redo_store['redo']:
                action = undo_redo_store['redo'].pop()
                undo_redo_store['undo'].append(action)
                if action[0] == 'color_nodes':
                    for node in action[1]:
                        G.nodes[node]['color'] = action[2]
                elif action[0] == 'delete_node':
                    G.remove_node(action[1])
                elif action[0] in ['reset_meter', 'reset_transformer']:
                    if action[0] == 'reset_meter':
                        stored_meter_nodes = []
                    else:
                        stored_transformer_nodes = []
                new_fig = create_figure(highlight_paths=is_highlight_mode, stored_meter_nodes=stored_meter_nodes, stored_transformer_nodes=stored_transformer_nodes)
                return new_fig, stored_meter_nodes, stored_transformer_nodes, is_placing_meter_mode, is_placing_transformer_mode, is_delete_mode, is_highlight_mode, "Redo successful", lasso_store, undo_redo_store
            return dash.no_update, stored_meter_nodes, stored_transformer_nodes, is_placing_meter_mode, is_placing_transformer_mode, is_delete_mode, is_highlight_mode, "Nothing to redo", lasso_store, undo_redo_store

        if button_id == 'save-clusters-button':
            transformer_colors = {}
            for node, data in G.nodes(data=True):
                if data.get('is_transformer'):
                    transformer_colors[node] = data['color']
            save_state(stored_meter_nodes, is_highlight_mode, lasso_store, transformer_colors, meter_connections)
            return dash.no_update, stored_meter_nodes, stored_transformer_nodes, is_placing_meter_mode, is_placing_transformer_mode, is_delete_mode, is_highlight_mode, "Clusters, colors, and labels saved", lasso_store, undo_redo_store
        if button_id == 'load-clusters-button':
            loaded_meter_nodes, loaded_highlight_mode, loaded_lasso_store, loaded_transformer_colors, loaded_meter_connections = load_state()

            logging.info(f"Loaded meter nodes: {loaded_meter_nodes}")
            logging.info(f"Loaded highlight mode: {loaded_highlight_mode}")
            logging.info(f"Loaded lasso store: {loaded_lasso_store}")
            logging.info(f"Loaded transformer colors: {loaded_transformer_colors}")
            logging.info(f"Loaded meter connections: {loaded_meter_connections}")
            
            
            
            # Update global variables
            highlight_mode = True
            meter_connections = loaded_meter_connections
            stored_meter_nodes = loaded_meter_nodes
            lasso_store = loaded_lasso_store

 

                    # Update cluster_info
            existing_colors = get_existing_cluster_colors()
            cluster_info = {color: {'transformers': [], 'next_label': 'AA'} for color in existing_colors}

            # Apply loaded transformer colors and assign labels
            for node, color in loaded_transformer_colors.items():
                if int(node) in G.nodes:
                    G.nodes[int(node)]['color'] = color
                    G.nodes[int(node)]['is_transformer'] = True
                    
                    # Check if the node already has a label
                    if 'label' in G.nodes[int(node)]:
                        existing_label = G.nodes[int(node)]['label'].split()[-1]
                        if color not in cluster_info:
                            cluster_info[color] = {'transformers': [], 'next_label': 'AA'}
                        if existing_label >= cluster_info[color]['next_label']:
                            # Update next_label if necessary
                            next_label = chr(ord(existing_label[0]) + 1) + chr(ord(existing_label[1]) + 1)
                            cluster_info[color]['next_label'] = next_label
                    else:
                        # Assign new label
                        label = assign_transformer_label(int(node), color)
                        G.nodes[int(node)]['label'] = label
                    
                    if color not in cluster_info:
                        cluster_info[color] = {'transformers': [], 'next_label': 'AA'}
                    if int(node) not in cluster_info[color]['transformers']:
                        cluster_info[color]['transformers'].append(int(node))

            # Update meter colors based on their transformer colors
            for transformer, meters in loaded_meter_connections.items():
                if int(transformer) in G.nodes:
                    transformer_color = G.nodes[int(transformer)]['color']
                    for meter in meters:
                        if int(meter) in G.nodes:
                            G.nodes[int(meter)]['color'] = transformer_color
                            G.nodes[int(meter)]['is_meter'] = True  # Ensure meter flag is set

            # Set meter nodes
            for node in stored_meter_nodes:
                if int(node['id']) in G.nodes:
                    G.nodes[int(node['id'])]['is_meter'] = True

            # Validate and correct label uniqueness
            for color, info in cluster_info.items():
                used_labels = set()
                for node in info['transformers']:
                    label = G.nodes[node]['label'].split()[-1]
                    if label in used_labels:
                        # Assign a new unique label
                        new_label = assign_transformer_label(node, color)
                        G.nodes[node]['label'] = new_label
                        logging.warning(f"Duplicate label found and corrected for node {node}. New label: {new_label}")
                    else:
                        used_labels.add(label)

            logging.info(f"Number of transformer nodes: {sum(1 for n in G.nodes if G.nodes[n].get('is_transformer'))}")
            logging.info(f"Number of meter nodes: {sum(1 for n in G.nodes if G.nodes[n].get('is_meter'))}")
            logging.info(f"Updated cluster_info: {cluster_info}")

            # Add debug logging for all nodes
            for node, data in G.nodes(data=True):
                if data.get('is_transformer'):
                    logging.debug(f"Transformer node {node}: {data}")
                elif data.get('is_meter'):
                    logging.debug(f"Meter node {node}: {data}")
                else:
                    logging.debug(f"Other node {node}: {data}")

            new_fig = create_figure(
                highlight_paths=is_highlight_mode,  # Force highlighting on
                stored_meter_nodes=stored_meter_nodes,
                stored_transformer_nodes=stored_transformer_nodes,
                lasso_selected=lasso_store[-1] if lasso_store else None,
                transformer_colors={node: data['color'] for node, data in G.nodes(data=True) if data.get('is_transformer')}
            )

            return new_fig, stored_meter_nodes, stored_transformer_nodes, False, False, False, True, "Clusters loaded with highlighting enabled", lasso_store, undo_redo_store
        
        
        if button_id == 'export-data-button':
            export_data = {
                'meter_nodes': stored_meter_nodes,
                'transformer_nodes': stored_transformer_nodes,
                'lasso_selections': lasso_store,
                'transformer_colors': {node: data['color'] for node, data in G.nodes(data=True) if data.get('is_transformer')},
                'meter_connections': meter_connections
            }
            with open('exported_data.json', 'w') as f:
                json.dump(export_data, f)
            return dash.no_update, stored_meter_nodes, stored_transformer_nodes, is_placing_meter_mode, is_placing_transformer_mode, is_delete_mode, is_highlight_mode, "Data exported to exported_data.json", lasso_store, undo_redo_store

        raise PreventUpdate

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return dash.no_update









@app.callback(
    [Output('node-info', 'children'),
     Output('locked-node-info', 'children')],
    [Input('graph', 'hoverData'),
     Input('graph', 'clickData')],
    [State('locked-node-info', 'children')]
)
def display_node_info(hoverData, clickData, locked_info):
    def process_node_data(point):
        logging.debug(f"Processing point data: {point}")

        # Try to extract node information from various possible keys
        node_info = None
        for key in ['text', 'hovertext', 'customdata', 'pointNumber', 'pointIndex', 'curveNumber']:
            if key in point:
                node_info = point[key]
                break

        if node_info is None:
            return f"Unable to extract node information. Available keys: {', '.join(point.keys())}"

        try:
            # If node_info is a list or tuple, take the first element
            if isinstance(node_info, (list, tuple)):
                node_info = node_info[0]
            
            # Try to extract node_id
            node_id = int(str(node_info).split('<br>')[0])
        except (ValueError, IndexError, AttributeError) as e:
            return f"Error parsing node ID: {str(e)}. Raw node info: {node_info}"

        if node_id not in G.nodes:
            return f"Node ID {node_id} not found in the graph."

        node_data = G.nodes[node_id]
        info = f"Node ID: {node_id}\n"
        
        if node_data.get('is_transformer'):
            label = node_data.get('label')
            if not label:
                color = node_data.get('color', 'N/A')
                label = assign_transformer_label(node_id, color)
                G.nodes[node_id]['label'] = label  # Store the new label
            info += f"Type: Transformer\nLabel: {label}\nColor: {node_data.get('color', 'N/A')}\n"
        elif node_data.get('is_meter'):
            info += "Type: Meter\n"
            associated_transformer = next((t for t, meters in meter_connections.items() if node_id in meters), None)
            if associated_transformer:
                transformer_label = G.nodes[associated_transformer].get('label')
                if not transformer_label:
                    transformer_color = G.nodes[associated_transformer].get('color', 'N/A')
                    transformer_label = assign_transformer_label(associated_transformer, transformer_color)
                    G.nodes[associated_transformer]['label'] = transformer_label  # Store the new label
                info += f"Associated Transformer: {transformer_label}\nCluster Color: {G.nodes[associated_transformer].get('color', 'N/A')}\n"
        else:
            info += "Type: Regular Node\n"
        
        return info

    if clickData:
        # Process click data (treat as locking action)
        info = process_node_data(clickData['points'][0])
        return info, f"Locked: {info}"  # Update both hover and locked info
    elif hoverData:
        # Process hover data
        info = process_node_data(hoverData['points'][0])
        return info, locked_info  # Update hover info, keep locked info unchanged
    
    return "Hover over a node to see its information.", locked_info











@app.callback(
    Output('lasso-info', 'children'),
    Input('graph', 'selectedData'),
    State('selection-mode', 'value')
)
def display_lasso_info(selectedData, selection_mode):
    if selectedData and selection_mode == 'lasso':
        selected_points = selectedData['points']
        num_selected = len(selected_points)
        num_transformers = sum(1 for point in selected_points if G.nodes[int(point['text'])].get('is_transformer'))
        num_meters = sum(1 for point in selected_points if G.nodes[int(point['text'])].get('is_meter'))
        return f"Selected: {num_selected} nodes\nTransformers: {num_transformers}\nMeter Nodes: {num_meters}"
    return ""

@app.callback(
    Output('graph', 'config'),
    Input('zoom-pan-lock', 'data')
)
def update_graph_config(lock_data):
    if lock_data['locked']:
        return {'scrollZoom': False, 'modeBarButtonsToAdd': ['drawclosedpath', 'eraseshape']}
    else:
        return {'scrollZoom': True, 'modeBarButtonsToAdd': ['drawclosedpath', 'eraseshape']}

if __name__ == '__main__':
    # Initialize meter connections
    meter_connections = {
        961011153: [961011152, 31258725, 3252904691],
        502815960: [71723092, 11359207136, 9008674764]
    }
    
    app.run_server(debug=True)
    