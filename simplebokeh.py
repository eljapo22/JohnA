import dash
from dash import dcc, html, Input, Output, State
import dash_cytoscape as cyto
import json
import base64

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    html.Button("Add Substation", id="add-substation", n_clicks=0),
    html.Button("Add Overhead Transformer", id="add-overhead-transformer", n_clicks=0),
    html.Button("Add Ground Transformer", id="add-ground-transformer", n_clicks=0),
    dcc.Input(id="custom-text-input", type="text", placeholder="Enter custom text"),
    html.Button("Add Custom Text", id="add-custom-text", n_clicks=0),
    html.Button("Save Layout", id="save-layout", n_clicks=0),
    html.Button("Clear Layout", id="clear-layout", n_clicks=0),
    dcc.Upload(
        id='upload-layout',
        children=html.Button('Upload Layout'),
        multiple=False
    ),
    cyto.Cytoscape(
        id='network-graph',
        layout={'name': 'preset'},
        style={'width': '100%', 'height': '800px'},
        elements=[],
        stylesheet=[
            {'selector': 'node', 'style': {'label': 'data(label)', 'width': '20px', 'height': '20px', 'background-color': 'data(color)'}},
            {'selector': 'edge', 'style': {'width': 2, 'line-color': '#ccc'}},
            {'selector': ':selected', 'style': {'background-color': 'yellow', 'line-color': 'yellow', 'target-arrow-color': 'yellow', 'source-arrow-color': 'yellow', 'border-width': 2}}
        ],
        userZoomingEnabled=True,
        userPanningEnabled=True,
    ),
    html.Div(id='selected-node-data', style={'display': 'none'})
])

# Helper function to generate a node
def generate_node(id, label, color, position_offset):
    return {
        'data': {'id': id, 'label': label, 'color': color},
        'position': {'x': 100 + position_offset * 50, 'y': 100},  # Adjusted to use position_offset directly
        'grabbable': True,
        'draggable': True
    }

# Helper function to generate an edge
def generate_edge(source, target):
    return {
        'data': {'source': source, 'target': target}
    }

# Callback to update the graph
@app.callback(
    Output("network-graph", "elements"),
    Output("selected-node-data", "children"),
    Input("add-substation", "n_clicks"),
    Input("add-overhead-transformer", "n_clicks"),
    Input("add-ground-transformer", "n_clicks"),
    Input("add-custom-text", "n_clicks"),
    Input("save-layout", "n_clicks"),
    Input("clear-layout", "n_clicks"),
    Input('upload-layout', 'contents'),
    State("network-graph", "elements"),
    State("custom-text-input", "value"),
    State('network-graph', 'selectedNodeData'),
    prevent_initial_call=True
)
def update_graph(add_substation_clicks, add_overhead_transformer_clicks, add_ground_transformer_clicks, add_custom_text_clicks, save_layout_clicks, clear_layout_clicks, upload_layout_contents, elements, custom_text, selected_node_data):
    if elements is None:
        elements = []
        
    ctx = dash.callback_context
    if not ctx.triggered:
        return elements, selected_node_data
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    position_offset = len(elements)  # Use the current length of elements to determine position offset
    
    if button_id == "add-substation":
        elements.append(generate_node(f"substation-{add_substation_clicks}", f"Substation {add_substation_clicks}", 'red', position_offset))
    elif button_id == "add-overhead-transformer":
        elements.append(generate_node(f"overhead-{add_overhead_transformer_clicks}", f"Overhead {add-overhead_transformer_clicks}", 'grey', position_offset))
    elif button_id == "add-ground-transformer":
        elements.append(generate_node(f"ground-{add-ground_transformer_clicks}", f"Ground {add-ground_transformer_clicks}", 'skyblue', position_offset))
    elif button_id == "add-custom-text" and custom_text:
        elements.append(generate_node(f"custom-{custom_text}", custom_text, 'black', position_offset))
    elif button_id == "save-layout":
        layout_data = json.dumps(elements)
        return elements, dcc.send_data_frame(layout_data, "layout.json")
    elif button_id == "clear-layout":
        elements = []
    elif button_id == 'upload-layout' and upload_layout_contents:
        content_type, content_string = upload_layout_contents.split(',')
        decoded = base64.b64decode(content_string)
        layout_data = json.loads(decoded)
        elements = layout_data
    elif 'network-graph' in button_id and selected_node_data:
        if len(selected_node_data) == 2:
            source = selected_node_data[0]['data']['id']
            target = selected_node_data[1]['data']['id']
            elements.append(generate_edge(source, target))
    
    return elements, selected_node_data

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
