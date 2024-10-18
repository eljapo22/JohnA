import json
import networkx as nx
import plotly.graph_objects as go
import dash
from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import datetime
import time

# Define the mapping from node labels to node IDs
label_to_id = {
    'DS Substation: AA': 24959509,
    'DS T/F BB': 34404246,
    'DS T/F CC': 34404249,
    'm112': 344474250,  # Meter 112 in Area MM
    'm48': 3030997330   # Meter 48 in Area KK
}

# Define the nodes using their IDs
substation = label_to_id['DS Substation: AA']
transformer_bb = label_to_id['DS T/F BB']
transformer_cc = label_to_id['DS T/F CC']
m112 = label_to_id['m112']
m48 = label_to_id['m48']

class NetworkComponent:
    def __init__(self, id, type, name=None, area=None, feeder=None):
        self.id = id
        self.type = type
        self.name = name or str(id)
        self.area = area
        self.feeder = feeder
        self.connections = []

    def add_connection(self, component):
        if component not in self.connections:
            self.connections.append(component)

    def set_feeder(self, feeder):
        self.feeder = feeder
        for connection in self.connections:
            if connection.feeder is None:
                connection.set_feeder(feeder)

class NetworkTopology:
    def __init__(self):
        self.components = {}

    def add_component(self, component):
        self.components[component.id] = component

    def connect_components(self, component1_id, component2_id):
        self.components[component1_id].add_connection(self.components[component2_id])
        self.components[component2_id].add_connection(self.components[component1_id])

    def get_component(self, id):
        return self.components.get(id)

network_topology = NetworkTopology()

def load_graph_from_cache():
    with open("C:\\Users\\eljapo22\\gephi\\cache\\7644e20afe6c4a94db3dcb733771bb0d4d5a6fc6.json", 'r') as f:
        data = json.load(f)
    G = nx.Graph()
    for element in data['elements']:
        if element['type'] == 'node':
            area = None
            name = str(element['id'])
            if element['id'] == m112:
                area = 'MM'
                name = 'm112'
            elif element['id'] == m48:
                area = 'KK'
                name = 'm48'
            elif element['id'] == transformer_bb:
                name = 'DS T/F BB'
            elif element['id'] == transformer_cc:
                name = 'DS T/F CC'
            elif element['id'] == substation:
                name = 'DS Substation: AA'
            component = NetworkComponent(element['id'], 'node', name=name, area=area)
            G.add_node(element['id'], pos=(element['lon'], element['lat']), component=component)
            network_topology.add_component(component)
        elif element['type'] == 'way':
            for i in range(len(element['nodes']) - 1):
                G.add_edge(element['nodes'][i], element['nodes'][i + 1])
                network_topology.connect_components(element['nodes'][i], element['nodes'][i + 1])
    
    substation_component = network_topology.get_component(substation)
    substation_component.set_feeder('DS Feeder: 11')
    
    return G

G = load_graph_from_cache()
G.graph['crs'] = {'init': 'epsg:4326'}

def get_node_id_from_label(label):
    return label_to_id.get(label)

path_1 = nx.shortest_path(G, source=substation, target=transformer_bb)
path_2 = nx.shortest_path(G, source=transformer_bb, target=transformer_cc)
path_3 = nx.shortest_path(G, source=transformer_bb, target=m112)
path_4 = nx.shortest_path(G, source=transformer_cc, target=m48)

all_paths = [path_1, path_2, path_3, path_4]

def add_path_to_plot(fig, path_color_map, width=2):
    for path, color in path_color_map.items():
        edge_x, edge_y = [], []
        for i in range(len(path) - 1):
            x0, y0 = G.nodes[path[i]]['pos']
            x1, y1 = G.nodes[path[i + 1]]['pos']
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=width, color=color),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
    return fig

def get_component_type(node_id):
    component = network_topology.get_component(node_id)
    if component.type == 'node':
        if node_id == substation:
            return 'substation'
        elif node_id in [transformer_bb, transformer_cc]:
            return 'transformer'
        elif node_id in [m112, m48]:
            return 'meter'
        else:
            return 'line'
    else:
        return component.type

def label_nodes_in_paths(paths):
    for path_index, path in enumerate(paths):
        for i, node_id in enumerate(path):
            sequential_position = i + 1
            prev_node_id = path[i - 1] if i > 0 else None
            next_node_id = path[i + 1] if i < len(path) - 1 else None
            
            label = f"Node {sequential_position} of Path {path_index + 1}"
            if prev_node_id is not None and next_node_id is not None:
                label += f", Between Node {prev_node_id} and Node {next_node_id}"
            elif prev_node_id is not None:
                label += f", After Node {prev_node_id}"
            elif next_node_id is not None:
                label += f", Before Node {next_node_id}"

            G.nodes[node_id]['custom_label'] = label
            
            custom_label = f"ID: {node_id}"
            if G.nodes[node_id].get('custom_label'):           
                custom_label += f" - {G.nodes[node_id]['custom_label']}"
            
            G.nodes[node_id]['display_label'] = custom_label

label_nodes_in_paths([path_1, path_2, path_3, path_4])

highlighted_nodes = set(path_1 + path_2 + path_3 + path_4)

fig = go.Figure()

initial_path_color_map = {tuple(path): 'gray' for path in all_paths}
fig = add_path_to_plot(fig, initial_path_color_map, 3)

for node in highlighted_nodes:
    x, y = G.nodes[node]['pos']
    component = network_topology.get_component(node)
    color = 'blue'
    size = 5
    custom_label = f"ID: {node} - {component.name}"
    
    if node in [m112, m48]:
        color = 'aqua'
        size = 10
        if component.area:
            custom_label += f" (Area: {component.area})"
    elif node == substation:
        color = 'red'
        size = 10
    elif node in [transformer_bb, transformer_cc]:
        color = 'green'
        size = 10
    
    if component.feeder:
        custom_label += f" - {component.feeder}"

    fig.add_trace(go.Scatter(
        x=[x], y=[y],
        mode='markers',
        marker=dict(color=color, size=size),
        customdata=[(node, custom_label, x, y)],
        hovertemplate='%{customdata[1]}<br>Coordinates: (%{customdata[2]}, %{customdata[3]})',
        hoverinfo='text',
        showlegend=False
    ))

edge_x, edge_y = [], []
for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
fig.add_trace(go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(color='rgba(0, 0, 255, 0.1)', width=1),
    mode='lines',
    hoverinfo='none',
    showlegend=False
))

legend_items = [
    {'color': 'red', 'name': 'DS Substation: AA'},
    {'color': 'green', 'name': 'DS T/F BB'},
    {'color': 'green', 'name': 'DS T/F CC'},
    {'color': 'aqua', 'name': 'Meters'},
    {'color': 'gray', 'name': 'Initial Path'},
    {'color': 'red', 'name': 'Disconnected Path'},
    {'color': 'green', 'name': 'Active Path'}
]

for item in legend_items:
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=10, color=item['color']),
        legendgroup=item['name'],
        showlegend=True,
        name=item['name']
    ))

app = Dash(__name__)

app.layout = html.Div([
        html.H1("Interactive Network Map"),
        html.Div([
            html.Button("Simulate Disconnection", id='simulate-disconnection', n_clicks=0),
            html.Button("Simulate Reconnection", id='simulate-reconnection', n_clicks=0, style={'display': 'none'}),
            html.Button("Reset Simulation", id='reset-simulation', n_clicks=0),
        ], style={'margin-bottom': '10px'}),
        dcc.Graph(figure=fig, id='graph', style={'height': '80vh', 'width': '100%'}),
        html.Div(id='path-info', style={'margin-top': '20px', 'font-weight': 'bold', 'font-size': '18px', 'color': 'blue'}),
        html.Div(id='simulation-status', style={'margin-top': '20px', 'font-weight': 'bold', 'font-size': '18px', 'color': 'red'}),
        html.Div(id='simulation-controls', style={'display': 'none'}, children=[
            dcc.Dropdown(id='reconnection-start-point', placeholder="Select reconnection start point"),
            html.Button("Reconnect Next", id='reconnect-next', n_clicks=0),
            html.Button("Auto Reconnect", id='auto-reconnect', n_clicks=0)
        ]),
        dcc.Store(id='network-state', data={'disconnected_paths': [], 'reconnection_sequence': []}),
        html.Div(id='log-output', style={'margin-top': '20px', 'font-weight': 'bold'}),
        html.Button("Toggle ClickData Log", id='toggle-log', n_clicks=0, style={'margin-top': '20px'}),
        html.Div(id='click-data-log', style={'display': 'none', 'border': '1px solid #ccc', 'padding': '10px', 'max-height': '300px', 'overflow-y': 'scroll'}),
        dcc.Store(id='selected-nodes', data={'start_node': None, 'end_node': None}),
        dcc.Store(id='simulation-type', data=None),
        dcc.Store(id='fault-type', data=None),
        dcc.Store(id='simulation-completed', data=False),
        dcc.RadioItems(
            id='fault-type-selector',
            options=[
                {'label': 'Parent Node Fault', 'value': 'Parent Node Fault'},
                {'label': 'Line Fault', 'value': 'Line Fault'}
            ],
            value='',
            style={'display': 'none'},
            labelStyle={'display': 'block'}
        ),
        html.Button("Start Simulation", id='start-simulation', n_clicks=0, style={'display': 'none'})
    ])

@app.callback(
        Output('click-data-log', 'children'),
        Output('click-data-log', 'style'),
        Output('path-info', 'children'),
        Output('simulation-type', 'data'),
        Output('selected-nodes', 'data'),
        Output('start-simulation', 'style'),
        Output('fault-type', 'data'),
        Output('fault-type-selector', 'style'),
        Output('graph', 'figure'),
        Output('simulation-status', 'children'),
        Output('fault-type-selector', 'value'),
        Output('simulation-completed', 'data'),
        Output('simulate-reconnection', 'style'),
        Output('simulation-controls', 'style'),
        Output('network-state', 'data'),
        Output('reconnection-start-point', 'options'),
        Input('graph', 'clickData'),
        Input('toggle-log', 'n_clicks'),
        Input('simulate-disconnection', 'n_clicks'),
        Input('simulate-reconnection', 'n_clicks'),
        Input('reset-simulation', 'n_clicks'),
        Input('start-simulation', 'n_clicks'),
        Input('fault-type-selector', 'value'),
        Input('reconnect-next', 'n_clicks'),
        Input('auto-reconnect', 'n_clicks'),
        Input('reconnection-start-point', 'value'),
        State('click-data-log', 'children'),
        State('simulation-type', 'data'),
        State('selected-nodes', 'data'),
        State('fault-type', 'data'),
        State('simulation-completed', 'data'),
        State('network-state', 'data'),
    )
def handle_all_interactions(clickData, toggle_log_clicks, disconn_clicks, reconn_clicks, reset_clicks, 
                                start_sim_clicks, selected_fault_type, reconnect_next_clicks, auto_reconnect_clicks,
                                reconnection_start_point, log_children, simulation_type, selected_nodes, 
                                stored_fault_type, simulation_completed, network_state):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

        print(f"DEBUG: Triggered ID: {triggered_id}")
        print(f"DEBUG: Current simulation type: {simulation_type}")
        print(f"DEBUG: Selected nodes: {selected_nodes}")
        print(f"DEBUG: Stored fault type: {stored_fault_type}")
        print(f"DEBUG: Network state: {network_state}")

        updated_fig = dash.no_update
        path_info = ""
        simulation_status = ""
        reconnection_options = []

        log_children = log_children or []
        selected_nodes = selected_nodes or {'start_node': None, 'end_node': None}
        simulation_type = simulation_type or None
        stored_fault_type = stored_fault_type or None

        log_style = {'display': 'none'}
        start_sim_style = {'display': 'none'}
        fault_type_selector_style = {'display': 'none'}
        fault_type_selector_value = ''
        simulation_completed = False

        try:
            if triggered_id == 'toggle-log':
                log_style = {'display': 'block' if toggle_log_clicks % 2 == 1 else 'none'}
            elif triggered_id == 'reset-simulation':
                simulation_type, selected_nodes, stored_fault_type = None, {'start_node': None, 'end_node': None}, None
                network_state = {'disconnected_paths': [], 'reconnection_sequence': []}
                path_info = "Simulation reset. Please select a new simulation type."
                updated_fig = go.Figure(fig)
                simulation_status = "Simulation reset."
            elif triggered_id == 'simulate-disconnection':
                simulation_type = 'disconnection'
                selected_nodes = {'start_node': None, 'end_node': None}
                path_info = "Disconnection simulation started. Please select the start node."
                simulation_status = "Disconnection simulation in progress."
            elif triggered_id == 'simulate-reconnection':
                simulation_type = 'reconnection'
                path_info = "Reconnection simulation started. Please select a starting point."
                simulation_status = "Reconnection simulation in progress."
                disconnected_nodes = set(sum(network_state['disconnected_paths'], []))
                available_nodes = [substation] + [node for node in [transformer_bb, transformer_cc] if node not in disconnected_nodes]
                reconnection_options = [
                    {'label': network_topology.get_component(node).name, 'value': node} 
                    for node in available_nodes
                ]
                if not reconnection_options:
                    path_info = "No nodes available for reconnection. All paths may already be connected."
                updated_fig = update_reconnection_visualization(go.Figure(fig), network_state)
            elif triggered_id == 'graph' and simulation_type == 'disconnection':
                if clickData:
                    node_id = clickData['points'][0]['customdata'][0]
                    component = network_topology.get_component(node_id)
                    if selected_nodes['start_node'] is None:
                        selected_nodes['start_node'] = node_id
                        path_info = f"Start node selected: {component.name}. Please select the end node."
                    elif selected_nodes['end_node'] is None:
                        selected_nodes['end_node'] = node_id
                        path_info = f"End node selected: {component.name}. Please select the fault type."
                        fault_type_selector_style = {'display': 'block'}
            elif triggered_id == 'fault-type-selector':
                stored_fault_type = selected_fault_type
                start_sim_style = {'display': 'block'}
            elif triggered_id == 'start-simulation' and simulation_type == 'disconnection':
                if selected_nodes['start_node'] and selected_nodes['end_node'] and stored_fault_type:
                    start_node, end_node = selected_nodes['start_node'], selected_nodes['end_node']
                    paths_to_deactivate = determine_paths_to_deactivate(start_node, end_node, stored_fault_type)
                    network_state['disconnected_paths'] = paths_to_deactivate
                    path_color_map = {tuple(path): 'red' if path in paths_to_deactivate else 'green' for path in all_paths}
                    updated_fig = add_path_to_plot(go.Figure(fig), path_color_map, 3)
                    simulation_status = "Disconnection simulation completed. You can now start a reconnection simulation."
                    simulation_completed = True
                    print(f"Updated network state: {network_state}")
            elif triggered_id == 'auto-reconnect' and simulation_type == 'reconnection':
                if reconnection_start_point:
                    network_state = auto_reconnect(reconnection_start_point, network_state)
                    path_info = "Auto-reconnection completed."
                    updated_fig = update_reconnection_visualization(go.Figure(fig), network_state)
                else:
                    path_info = "Please select a starting point for auto-reconnection."
            elif triggered_id == 'reconnect-next' and simulation_type == 'reconnection':
                if reconnection_start_point:
                    next_component = get_next_reconnection_component(reconnection_start_point, network_state)
                    if next_component:
                        for path in network_state['disconnected_paths']:
                            if next_component in path:
                                path_index = network_state['disconnected_paths'].index(path)
                                reconnected_path = path[:path.index(next_component) + 1]
                                network_state['reconnection_sequence'].extend(reconnected_path)
                                network_state['disconnected_paths'][path_index] = path[path.index(next_component) + 1:]
                        path_info = f"Reconnected up to {network_topology.get_component(next_component).name}."
                        updated_fig = update_reconnection_visualization(go.Figure(fig), network_state)
                        reconnection_start_point = next_component
                    else:
                        path_info = "All possible components reconnected."
                else:
                    path_info = "Please select a starting point for reconnection."

            if not any(network_state['disconnected_paths']):
                simulation_status = "Reconnection simulation completed."
                simulation_completed = True

            log_children.append(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {path_info}")

            return (
                log_children,
                log_style,
                path_info,
                simulation_type,
                selected_nodes,
                start_sim_style,
                stored_fault_type,
                fault_type_selector_style,
                updated_fig,
                simulation_status,
                fault_type_selector_value,
                simulation_completed,
                {'display': 'block' if simulation_completed else 'none'},
                {'display': 'block' if simulation_type == 'reconnection' else 'none'},
                network_state,
                reconnection_options
            )
        except Exception as e:
            print(f"Error in callback: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise



def determine_paths_to_deactivate(start_node, end_node, fault_type):
        paths_to_deactivate = []
        if fault_type == 'Parent Node Fault':
            if start_node == substation:
                paths_to_deactivate = all_paths
            elif start_node == transformer_bb:
                paths_to_deactivate = [path_2, path_3, path_4]
            elif start_node == transformer_cc:
                paths_to_deactivate = [path_4]
        elif fault_type == 'Line Fault':
            if start_node in [transformer_bb, transformer_cc] and end_node in [transformer_bb, transformer_cc]:
                paths_to_deactivate = [path_2, path_4]  # Disconnect both paths after Transformer BB
            elif (start_node in [transformer_bb, transformer_cc] and end_node in [m112, m48]) or \
                (end_node in [transformer_bb, transformer_cc] and start_node in [m112, m48]):
                paths_to_deactivate = [path for path in all_paths if end_node in path or start_node in path]
        return paths_to_deactivate

def get_next_reconnection_component(start_point, network_state):
        for path in network_state['disconnected_paths']:
            if start_point in path:
                return path[path.index(start_point) + 1]
        return None

def auto_reconnect(start_point, network_state):
        while True:
            next_component = get_next_reconnection_component(start_point, network_state)
            if next_component:
                for path in network_state['disconnected_paths']:
                    if next_component in path:
                        path_index = network_state['disconnected_paths'].index(path)
                        reconnected_path = path[:path.index(next_component) + 1]
                        network_state['reconnection_sequence'].extend(reconnected_path)
                        network_state['disconnected_paths'][path_index] = path[path.index(next_component) + 1:]
                start_point = next_component
            else:
                break
        return network_state



def update_reconnection_visualization(fig, network_state):
        if not any(network_state['disconnected_paths']) and not network_state['reconnection_sequence']:
            return fig  # Return the original figure if no changes have been made

        print(f"Updating visualization with network state: {network_state}")
        reconnected = set(network_state['reconnection_sequence'])
        disconnected = set(sum(network_state['disconnected_paths'], []))
        path_color_map = {}
        for path in all_paths:
            if any(node in disconnected for node in path):
                path_color_map[tuple(path)] = 'red'
            elif any(node in reconnected for node in path):
                path_color_map[tuple(path)] = 'green'
            else:
                path_color_map[tuple(path)] = 'gray'
        print(f"Path color map: {path_color_map}")
        return add_path_to_plot(fig, path_color_map, 3)

if __name__ == '__main__':
    app.run_server(debug=True)    