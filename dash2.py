import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import logging
import base64
import os

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the hierarchical structure
hierarchy = {
    'Substation AA': {
        'Feeder 11': {
            'DT BB': ['Meter 113', 'Meter 121'],
            'DT CC': ['Meter 43', 'Meter 97', 'Meter 58', 'Meter 109'],
        },
        'Feeder 00': {}  # Empty for demonstration purposes
    }
}

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the 3D button style
button_style = {
    'cursor': 'pointer',
    'padding': '15px',
    'margin': '10px 5px',
    'border': '1px solid #ccc',
    'borderRadius': '5px',
    'backgroundColor': '#f0f0f0',
    'boxShadow': '2px 2px 2px rgba(0,0,0,0.1)',
    'transition': 'all 0.3s ease',
    'fontSize': '22px',
    'textAlign': 'center',
}

button_style_selected = {
    **button_style,
    'backgroundColor': '#e6f2ff',  # Subtle blue shade
    'boxShadow': 'inset 2px 2px 2px rgba(0,0,0,0.1)',
    'border': '1px solid #4da6ff',  # Slightly darker blue for border
}

# Read and encode the image
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, 'Assests', 'image (3).png')
with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

# Layout of the app
app.layout = html.Div(children=[
    # Main headers
    html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px', 'marginTop': '20px'}, children=[
        html.H1('Electrical Infrastructure', style={'fontSize': '35px', 'textAlign': 'center', 'width': '33%'}),
        html.H1('Electrical Parameters', style={'fontSize': '35px', 'textAlign': 'center', 'width': '33%'}),
        html.H1('Time Frame', style={'fontSize': '35px', 'textAlign': 'center', 'width': '33%'}),
    ]),
    
    # Substation and other dropdowns
    html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'margin': '10px'}, children=[
        html.Div(id='substation-div', style={'width': '32%', 'border': '1px solid #ccc', 'padding': '10px'}, children=[
            html.Label('Substation', style={'fontSize': '18px'}),
            dcc.Dropdown(
                id='substation-dropdown',
                options=[{'label': k, 'value': k} for k in hierarchy.keys()],
                value='Substation AA'
            ),
            html.Label('Feeder', style={'fontSize': '18px'}),
            dcc.Dropdown(id='feeder-dropdown'),
            html.Label('Transformer', style={'fontSize': '18px'}),
            dcc.Dropdown(id='transformer-dropdown'),
            html.Label('Meter', style={'fontSize': '18px'}),
            dcc.Dropdown(id='meter-dropdown'),
            html.Label('Account Number', style={'fontSize': '18px'}),
            dcc.Dropdown(id='customer-id-dropdown'),
        ]),
        
        html.Div(style={'width': '32%'}, children=[
            html.Div(id='parameter-div', style={'border': '1px solid #ccc', 'padding': '10px'}, children=[
                html.Div(id='active-power-div', style=button_style, children=['Active Power (kW)']),
                html.Div(id='current-div', style=button_style, children=['Current (A)']),
                html.Div(id='voltage-div', style=button_style, children=['Voltage (V)']),
                html.Div(id='apparent-power-div', style=button_style, children=['Apparent Power (kVA)']),
                html.Div(id='power-factor-div', style=button_style, children=['Power Factor']),
            ]),
        ]),
        
        html.Div(style={'width': '32%'}, children=[
            html.Div(id='time-frame-div', style={'border': '1px solid #ccc', 'padding': '10px'}, children=[
                html.Div(id='daily-div', style=button_style, children=['Daily']),
                html.Div(id='weekly-div', style=button_style, children=['Weekly']),
                html.Div(id='monthly-div', style=button_style, children=['Monthly']),
            ]),
        ]),
    ]),
    
    # Add these lines to store the selected parameter and time frame
    dcc.Store(id='selected-parameter', data='Active Power (kW)'),
    dcc.Store(id='selected-time-frame', data='D'),
    
    # Add this line to create space between options and graph
    html.Div(style={'height': '50px'}),  # Adds 50 pixels of vertical space
    
    # Add the electrical infrastructure diagram
    html.Div([
        html.H2("Electrical Infrastructure Diagram", style={'textAlign': 'center'}),
        html.Img(src=f'data:image/png;base64,{encoded_image}', style={'width': '100%', 'maxWidth': '1200px', 'margin': 'auto', 'display': 'block'})
    ], style={'marginTop': '50px', 'marginBottom': '50px'}),
    
    # Graph
    dcc.Graph(id='parameter-chart', style={'margin': '40px 10px 10px 10px'})  # Increased top margin
])

# Helper function to generate active power ranges based on time of day
def generate_active_power(time_index, transformer):
    active_power = np.zeros(len(time_index))
    
    for i, t in enumerate(time_index):
        hour = t.hour

        if transformer == 'DT CC':
            # Transformer CC
            if 17 <= hour < 22:  # 5 PM - 10 PM
                active_power[i] = np.random.uniform(70, 80)  # Max 80 kW
            elif 22 <= hour or hour < 5:  # 10 PM - 5 AM
                active_power[i] = np.random.uniform(10, 20)  # Between 10 and 20 kW
            elif 5 <= hour < 10:  # 5 AM - 10 AM
                active_power[i] = np.random.uniform(50, 60)  # Between 50 and 60 kW
            else:  # 10 AM - 5 PM
                active_power[i] = np.random.uniform(30, 50)  # Default range
        elif transformer == 'DT BB':
            # Transformer BB
            if 17 <= hour < 22:  # 5 PM - 10 PM
                active_power[i] = np.random.uniform(40, 50)  # Max 50 kW
            elif 22 <= hour or hour < 5:  # 10 PM - 5 AM
                active_power[i] = np.random.uniform(10, 15)  # Between 10 and 15 kW
            elif 5 <= hour < 10:  # 5 AM - 10 AM
                active_power[i] = np.random.uniform(30, 40)  # Max 40 kW
            elif 10 <= hour < 17:  # 10 AM - 5 PM
                active_power[i] = np.random.uniform(20, 25)  # Between 20 and 25 kW
        else:
            active_power[i] = np.random.uniform(10, 50)  # Default range for other transformers

    return active_power

# Helper function to generate simulated data based on selection
def generate_data(selected_substation, selected_feeder, selected_transformer, selected_meter, time_frame):
    end_date = pd.Timestamp.now().floor('H')
    
    if time_frame == 'D':
        start_date = end_date - pd.Timedelta(days=1)
        freq = 'H'
    elif time_frame == 'W':
        start_date = end_date - pd.Timedelta(weeks=1)
        freq = 'D'
    else:  # Monthly
        start_date = end_date - pd.Timedelta(days=30)
        freq = 'D'
    
    time_index = pd.date_range(start=start_date, end=end_date, freq=freq)

    if selected_transformer in ['DT BB', 'DT CC']:
        active_power = generate_active_power(time_index, selected_transformer)
    else:
        active_power = np.random.uniform(10, 100, len(time_index))

    voltage = np.random.uniform(220, 240, len(time_index))
    current = active_power * 1000 / voltage  # I = P / V

    # Apparent Power Calculation (kVA) - V * I / 1000
    apparent_power = (voltage * current) / 1000

    # Simulate Power Factor (PF) between 0.7 and 1.0
    power_factor = np.random.uniform(0.7, 1.0, len(time_index))

    data = {
        'Time': time_index,
        'Current R': current * np.random.uniform(0.9, 1.1, len(time_index)),
        'Current Y': current * np.random.uniform(0.9, 1.1, len(time_index)),
        'Current B': current * np.random.uniform(0.9, 1.1, len(time_index)),
        'Voltage R': voltage * np.random.uniform(0.98, 1.02, len(time_index)),
        'Voltage Y': voltage * np.random.uniform(0.98, 1.02, len(time_index)),
        'Voltage B': voltage * np.random.uniform(0.98, 1.02, len(time_index)),
        'Apparent Power (kVA)': apparent_power,
        'Power Factor': power_factor,
        'Active Power (kW)': active_power,
    }
    
    df = pd.DataFrame(data)
    return df

# Update graph based on user selection
@app.callback(
    Output('parameter-chart', 'figure'),
    [Input('substation-dropdown', 'value'),
     Input('feeder-dropdown', 'value'),
     Input('transformer-dropdown', 'value'),
     Input('meter-dropdown', 'value'),
     Input('selected-time-frame', 'data'),
     Input('selected-parameter', 'data')]
)
def update_chart(selected_substation, selected_feeder, selected_transformer, selected_meter, time_frame, selected_parameter):
    # Generate data
    df = generate_data(selected_substation, selected_feeder, selected_transformer, selected_meter, time_frame)

    # Determine which parameter to display
    if selected_parameter == 'Current (A)':
        y_cols = ['Current R', 'Current Y', 'Current B']
    elif selected_parameter == 'Voltage (V)':
        y_cols = ['Voltage R', 'Voltage Y', 'Voltage B']
    elif selected_parameter == 'Apparent Power (kVA)':
        y_cols = ['Apparent Power (kVA)']
    elif selected_parameter == 'Power Factor':
        y_cols = ['Power Factor']
    elif selected_parameter == 'Active Power (kW)':
        y_cols = ['Active Power (kW)']
    else:
        y_cols = ['Active Power (kW)']  # Default to Active Power

    fig = px.line(df, x='Time', y=y_cols, title=f'{selected_parameter} Over Time')

    # Add background shading for peak times
    if time_frame == 'D':
        # Morning Peak (5 AM - 10 AM)
        fig.add_vrect(x0="05:00", x1="10:00",
                      fillcolor="LightBlue", opacity=0.2, line_width=0,
                      annotation_text="Morning Peak", annotation_position="top left",
                      annotation_font_size=14, annotation_font_color="gray")
        # Evening Peak (5 PM - 10 PM)
        fig.add_vrect(x0="17:00", x1="22:00",
                      fillcolor="LightSalmon", opacity=0.2, line_width=0,
                      annotation_text="Evening Peak", annotation_position="top left",
                      annotation_font_size=14, annotation_font_color="gray")
    elif time_frame == 'W':
        # Highlight weekdays (assuming Monday is 0 and Sunday is 6)
        for i in range(5):  # Monday to Friday
            fig.add_vrect(x0=i, x1=i+1, fillcolor="LightGray", opacity=0.2, line_width=0)
        fig.add_annotation(x=2, y=1, text="Weekdays", showarrow=False, yref="paper", font=dict(size=14, color="gray"))
    elif time_frame == 'M':
        # Highlight weekdays for each week
        start_date = df['Time'].min()
        end_date = df['Time'].max()
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Monday to Friday
                fig.add_vrect(x0=current_date, x1=current_date + pd.Timedelta(days=1), 
                              fillcolor="LightGray", opacity=0.2, line_width=0)
            current_date += pd.Timedelta(days=1)
        fig.add_annotation(x=start_date + (end_date - start_date)/2, y=1, text="Weekdays", 
                           showarrow=False, yref="paper", font=dict(size=14, color="gray"))

    # Update x-axis format based on time frame
    if time_frame == 'D':
        fig.update_xaxes(dtick=3600000, tickformat='%H:%M', title_text='Time of Day')
    elif time_frame == 'W':
        fig.update_xaxes(dtick=86400000, tickformat='%a', title_text='Day of Week')
    else:  # Monthly
        fig.update_xaxes(dtick=86400000, tickformat='%d/%m', title_text='Date')

    # Update layout of the figure
    fig.update_layout(
        height=700,
        width=2500,
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode='x unified',
        yaxis_title=selected_parameter,
        title={
            'text': f"{selected_parameter} Over Time",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24, 'color': '#333', 'family': 'Arial, sans-serif'}
        }
    )
    
    return fig

# Callback for updating Feeder dropdown options
@app.callback(
    Output('feeder-dropdown', 'options'),
    [Input('substation-dropdown', 'value')]
)
def update_feeder_options(selected_substation):
    return [{'label': k, 'value': k} for k in hierarchy[selected_substation].keys()]

# Callback for updating Transformer dropdown options
@app.callback(
    Output('transformer-dropdown', 'options'),
    [Input('substation-dropdown', 'value'),
     Input('feeder-dropdown', 'value')]
)
def update_transformer_options(selected_substation, selected_feeder):
    if selected_feeder:
        return [{'label': k, 'value': k} for k in hierarchy[selected_substation][selected_feeder].keys()]
    return []

# Callback for updating Meter dropdown options
@app.callback(
    Output('meter-dropdown', 'options'),
    [Input('substation-dropdown', 'value'),
     Input('feeder-dropdown', 'value'),
     Input('transformer-dropdown', 'value')]
)
def update_meter_options(selected_substation, selected_feeder, selected_transformer):
    if selected_transformer:
        return [{'label': meter, 'value': meter} for meter in hierarchy[selected_substation][selected_feeder][selected_transformer]]
    return []

# Callback for updating Customer ID dropdown options
@app.callback(
    Output('customer-id-dropdown', 'options'),
    [Input('meter-dropdown', 'value')]
)
def update_customer_id_options(selected_meter):
    if selected_meter:
        return [{'label': f'Account for {selected_meter}', 'value': f'AA11BB{selected_meter[-3:]}'}]
    return []

# Callbacks for updating button styles
for param in ['active-power', 'current', 'voltage', 'apparent-power', 'power-factor']:
    @app.callback(
        Output(f'{param}-div', 'style'),
        [Input('selected-parameter', 'data')]
    )
    def update_button_style(selected_parameter, param=param):
        if selected_parameter == param.replace('-', ' ').title():
            return button_style_selected
        return button_style

for time_frame in ['daily', 'weekly', 'monthly']:
    @app.callback(
        Output(f'{time_frame}-div', 'style'),
        [Input('selected-time-frame', 'data')]
    )
    def update_time_frame_style(selected_time_frame, time_frame=time_frame):
        if (selected_time_frame == 'D' and time_frame == 'daily') or \
           (selected_time_frame == 'W' and time_frame == 'weekly') or \
           (selected_time_frame == 'M' and time_frame == 'monthly'):
            return button_style_selected
        return button_style

# Callbacks for updating selected parameter and time frame
@app.callback(
    Output('selected-parameter', 'data'),
    [Input(f'{param}-div', 'n_clicks') for param in ['active-power', 'current', 'voltage', 'apparent-power', 'power-factor']],
    [State('selected-parameter', 'data')]
)
def update_selected_parameter(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    return button_id.replace('-div', '').replace('-', ' ').title()

@app.callback(
    Output('selected-time-frame', 'data'),
    [Input('daily-div', 'n_clicks'),
     Input('weekly-div', 'n_clicks'),
     Input('monthly-div', 'n_clicks')],
    [State('selected-time-frame', 'data')]
)
def update_selected_time_frame(daily_clicks, weekly_clicks, monthly_clicks, current_time_frame):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'daily-div':
        return 'D'
    elif button_id == 'weekly-div':
        return 'W'
    elif button_id == 'monthly-div':
        return 'M'
    return current_time_frame

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)