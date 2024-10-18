import dash
from dash import html, dcc, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import base64
import json

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load and encode the image
with open("Assets/image (3).png", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

# Load saved positions
try:
    with open('saved_positions.json', 'r') as f:
        saved_positions = json.load(f)
except FileNotFoundError:
    saved_positions = {}

text_elements = [
    'Substation: AA', 'Feeder: 11', 'T/F: BB', 'T/F: CC',
    'Meter 113', 'Meter 121', 'Meter 48', 'Meter 97', 'Meter 98', 'Meter 108',
    'Active Power (kW)', 'Current (A)', 'Voltage (V)', 'Apparent Power (kVA)',
    'Power Factor', 'Daily', 'Weekly', 'Monthly'
]

app.layout = html.Div([
    # Small image at the top-right corner
    html.Img(src=f'data:image/png;base64,{encoded_image}', id='main-image',
             style={'position': 'absolute', 
                    'top': saved_positions.get('image', {}).get('top', '10px'),
                    'right': saved_positions.get('image', {}).get('right', '300px'),
                    'width': saved_positions.get('image', {}).get('width', '900px'),
                    'z-index': '1000'}),
    
    # Main content area (you can add your charts or other content here)
    html.Div(style={'height': '70vh'}),  # Placeholder for main content
    
    # Text elements and controls at the bottom
    html.Div([
        # Text container
        html.Div([
            html.Div(text, id={'type': 'text-element', 'index': i}, 
                     style={'position': 'absolute', 
                            'left': saved_positions.get('text_elements', {}).get(str(i), {}).get('left', f'{(i%6)*100}px'),
                            'top': saved_positions.get('text_elements', {}).get(str(i), {}).get('top', f'{(i//6)*30}px')})
            for i, text in enumerate(text_elements)
        ], id='text-container', style={'position': 'relative', 'height': '150px', 'border': '1px solid black', 'margin-bottom': '20px'}),
        
        # Controls
        dbc.Row([
            dbc.Col([
                dbc.Select(
                    id='text-selector',
                    options=[{'label': text, 'value': i} for i, text in enumerate(text_elements)],
                    value=0
                )
            ], width=3),
            dbc.Col([
                dbc.Input(id='x-position', type='number', placeholder="X position")
            ], width=2),
            dbc.Col([
                dbc.Input(id='y-position', type='number', placeholder="Y position")
            ], width=2),
            dbc.Col([
                dbc.Button("Update Position", id='update-button', color="primary")
            ], width=2),
            dbc.Col([
                dbc.Button("Capture Positions", id='capture-button', color="success")
            ], width=3)
        ], className="mt-3"),
        
        # Output area for captured positions
        html.Div(id='captured-positions', style={'margin-top': '20px', 'white-space': 'pre-wrap'}),
        
        # Hidden div to store current positions
        dcc.Store(id='current-positions', data=saved_positions)
    ], style={'position': 'fixed', 'bottom': 0, 'left': 0, 'right': 0, 'background': 'white', 'padding': '20px', 'border-top': '1px solid #ddd'})
])

@app.callback(
    Output({'type': 'text-element', 'index': ALL}, 'style'),
    Output('current-positions', 'data'),
    Input('update-button', 'n_clicks'),
    State('text-selector', 'value'),
    State('x-position', 'value'),
    State('y-position', 'value'),
    State({'type': 'text-element', 'index': ALL}, 'style'),
    State('current-positions', 'data')
)
def update_text_position(n_clicks, selected_text, x_pos, y_pos, current_styles, current_positions):
    if n_clicks is None:
        return current_styles, current_positions

    new_styles = [style.copy() if style else {} for style in current_styles]
    
    if selected_text is not None and x_pos is not None and y_pos is not None:
        new_styles[int(selected_text)].update({
            'position': 'absolute',
            'left': f'{x_pos}px',
            'top': f'{y_pos}px'
        })
        current_positions['text_elements'][str(selected_text)] = {'left': f'{x_pos}px', 'top': f'{y_pos}px'}
    
    return new_styles, current_positions

@app.callback(
    Output('captured-positions', 'children'),
    Input('capture-button', 'n_clicks'),
    State('current-positions', 'data'),
    State('main-image', 'style')
)
def capture_positions(n_clicks, current_positions, image_style):
    if n_clicks is None:
        return ""
    
    positions = current_positions.copy()
    positions['image'] = {
        "top": image_style['top'],
        "right": image_style['right'],
        "width": image_style['width']
    }
    
    # Save positions to a file
    with open('saved_positions.json', 'w') as f:
        json.dump(positions, f, indent=2)
    
    return json.dumps(positions, indent=2)

if __name__ == '__main__':
    app.run_server(debug=True)