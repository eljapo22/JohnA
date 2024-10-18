import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
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
    dcc.Store(id='app-state', data={
        'selected_parameter': None,
        'selected_time_frame': 'D',
        'substation': 'Substation AA',
        'feeder': None,
        'transformer': None,
        'meter': None
    }),
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
    
    # Add the electrical infrastructure diagram
    html.Div([
        html.H2("Electrical Infrastructure Diagram", style={'textAlign': 'center'}),
        html.Img(src=f'data:image/png;base64,{encoded_image}', style={'width': '100%', 'maxWidth': '1200px', 'margin': 'auto', 'display': 'block'})
    ], style={'marginTop': '50px', 'marginBottom': '50px'}),
    
    # Graph
    dcc.Graph(id='parameter-chart', style={'margin': '40px 10px 10px 10px'}),
    
    # Interaction log
    html.Div(id='interaction-log', style={'marginTop': '20px', 'padding': '10px', 'border': '1px solid #ddd'})
])

# Helper function to generate simulated data based on selection
def generate_data(selected_substation, selected_feeder, selected_transformer, selected_meter, time_frame):
    end_date = pd.Timestamp.now().floor('H')
    
    if time_frame == 'D':
        start_date = end_date - pd.Timedelta(days=1)
        time_index = pd.date_range(start=start_date, end=end_date, freq='H')
    elif time_frame == 'W':
        start_date = end_date - pd.Timedelta(weeks=1)
        time_index = pd.date_range(start=start_date, end=end_date, freq='H')
    else:  # Monthly
        start_date = end_date - pd.Timedelta(days=30)
        time_index = pd.date_range(start=start_date, end=end_date, freq='H')

    # Generate base active power
    active_power = np.random.uniform(10, 50, len(time_index))
    
    # Generate more distinct voltage and current data
    voltage_base = 230 + np.random.normal(0, 2, len(time_index))  # Base voltage around 230V
    current_base = active_power * 1000 / (voltage_base * np.sqrt(3))  # Base current calculation

    voltage_r = voltage_base + np.random.normal(0, 1, len(time_index))
    voltage_y = voltage_base + np.random.normal(0, 1, len(time_index))
    voltage_b = voltage_base + np.random.normal(0, 1, len(time_index))

    current_r = current_base * (1 + np.random.normal(0, 0.05, len(time_index)))
    current_y = current_base * (1 + np.random.normal(0, 0.05, len(time_index)))
    current_b = current_base * (1 + np.random.normal(0, 0.05, len(time_index)))

    apparent_power = np.sqrt(3) * voltage_base * current_base / 1000  # in kVA
    power_factor = np.random.uniform(0.8, 0.95, len(time_index))

    data = {
        'Time': time_index,
        'Current R': current_r,
        'Current Y': current_y,
        'Current B': current_b,
        'Voltage R': voltage_r,
        'Voltage Y': voltage_y,
        'Voltage B': voltage_b,
        'Apparent Power (kVA)': apparent_power,
        'Power Factor': power_factor,
        'Active Power (kW)': active_power,
    }
    
    df = pd.DataFrame(data)
    return df

# Main callback to update the app state
@app.callback(
    Output('app-state', 'data'),
    Output('interaction-log', 'children'),
    [Input('substation-dropdown', 'value'),
     Input('feeder-dropdown', 'value'),
     Input('transformer-dropdown', 'value'),
     Input('meter-dropdown', 'value'),
     Input('active-power-div', 'n_clicks'),
     Input('current-div', 'n_clicks'),
     Input('voltage-div', 'n_clicks'),
     Input('apparent-power-div', 'n_clicks'),
     Input('power-factor-div', 'n_clicks'),
     Input('daily-div', 'n_clicks'),
     Input('weekly-div', 'n_clicks'),
     Input('monthly-div', 'n_clicks')],
    [State('app-state', 'data'),
     State('interaction-log', 'children')]
)
def update_state(substation, feeder, transformer, meter, 
                 active_power_clicks, current_clicks, voltage_clicks, 
                 apparent_power_clicks, power_factor_clicks,
                 daily_clicks, weekly_clicks, monthly_clicks, 
                 current_state, current_log):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_state, current_log

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    new_state = current_state.copy()
    log_entry = f"User action: {trigger_id} selected. "

    if trigger_id == 'substation-dropdown':
        new_state['substation'] = substation
        new_state['feeder'] = None
        new_state['transformer'] = None
        new_state['meter'] = None
        log_entry += f"Substation set to {substation}."
    elif trigger_id == 'feeder-dropdown':
        new_state['feeder'] = feeder
        new_state['transformer'] = None
        new_state['meter'] = None
        log_entry += f"Feeder set to {feeder}."
    elif trigger_id == 'transformer-dropdown':
        new_state['transformer'] = transformer
        new_state['meter'] = None
        log_entry += f"Transformer set to {transformer}."
    elif trigger_id == 'meter-dropdown':
        new_state['meter'] = meter
        log_entry += f"Meter set to {meter}."
    elif trigger_id in ['active-power-div', 'current-div', 'voltage-div', 'apparent-power-div', 'power-factor-div']:
        parameter_mapping = {
            'active-power-div': 'Active Power (kW)',
            'current-div': 'Current (A)',
            'voltage-div': 'Voltage (V)',
            'apparent-power-div': 'Apparent Power (kVA)',
            'power-factor-div': 'Power Factor'
        }
        new_state['selected_parameter'] = parameter_mapping[trigger_id]
        log_entry += f"Parameter set to {new_state['selected_parameter']}."
    elif trigger_id in ['daily-div', 'weekly-div', 'monthly-div']:
        new_state['selected_time_frame'] = trigger_id[0].upper()
        log_entry += f"Time frame set to {new_state['selected_time_frame']}."

    new_log = html.Div([
        html.Div(log_entry),
        html.Br(),
        html.Div(current_log) if current_log else None
    ])

    return new_state, new_log

# Callback to update the chart based on the app state
@app.callback(
    Output('parameter-chart', 'figure'),
    Input('app-state', 'data')
)
def update_chart(state):
    if state['selected_parameter'] is None:
        return px.scatter(title="Please select a parameter")

    df = generate_data(state['substation'], state['feeder'], state['transformer'], state['meter'], state['selected_time_frame'])

    if state['selected_parameter'] == 'Current (A)':
        fig = px.line(df, x='Time', y=['Current R', 'Current Y', 'Current B'], title='Current Over Time (A)')
        y_axis_title = 'Current (A)'
    elif state['selected_parameter'] == 'Voltage (V)':
        fig = px.line(df, x='Time', y=['Voltage R', 'Voltage Y', 'Voltage B'], title='Voltage Over Time (V)')
        y_axis_title = 'Voltage (V)'
    elif state['selected_parameter'] == 'Apparent Power (kVA)':
        fig = px.line(df, x='Time', y='Apparent Power (kVA)', title='Apparent Power Over Time (kVA)')
        y_axis_title = 'Apparent Power (kVA)'
    elif state['selected_parameter'] == 'Power Factor':
        fig = px.line(df, x='Time', y='Power Factor', title='Power Factor Over Time')
        y_axis_title = 'Power Factor'
    else:  # Active Power (kW)
        fig = px.line(df, x='Time', y='Active Power (kW)', title='Active Power Over Time (kW)')
        y_axis_title = 'Active Power (kW)'

    # Update layout of the figure
    fig.update_layout(
        height=700,
        width=2500,
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode='x unified',
        xaxis_title='Time',
        yaxis_title=y_axis_title,
        title={
            'text': f"{state['selected_parameter']} Over Time",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24, 'color': '#333', 'family': 'Arial, sans-serif'}
        }
    )
    
    # Add shading for peak times (assuming 5 PM to 10 PM is peak)
    if state['selected_time_frame'] == 'D':
        fig.add_vrect(x0="17:00", x1="22:00", fillcolor="LightSalmon", opacity=0.2, line_width=0)

    # Add shading for weekdays if weekly or monthly view
    if state['selected_time_frame'] in ['W', 'M']:
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
    if state['selected_time_frame'] == 'D':
        fig.update_xaxes(dtick=3600000, tickformat='%H:%M', title_text='Time of Day')
    elif state['selected_time_frame'] == 'W':
        fig.update_xaxes(dtick=86400000, tickformat='%a', title_text='Day of Week')
    else:  # Monthly
        fig.update_xaxes(dtick=86400000, tickformat='%d/%m', title_text='Date')
    
    return fig

# Callback to update button styles
@app.callback(
    [Output(f'{param}-div', 'style') for param in ['active-power', 'current', 'voltage', 'apparent-power', 'power-factor']] +
    [Output(f'{time_frame}-div', 'style') for time_frame in ['daily', 'weekly', 'monthly']],
    Input('app-state', 'data')
)
def update_styles(state):
    parameter_mapping = {
        'Active Power (kW)': 'active-power',
        'Current (A)': 'current',
        'Voltage (V)': 'voltage',
        'Apparent Power (kVA)': 'apparent-power',
        'Power Factor': 'power-factor'
    }
    parameter_styles = [
        button_style_selected if state['selected_parameter'] == param else button_style
        for param in ['Active Power (kW)', 'Current (A)', 'Voltage (V)', 'Apparent Power (kVA)', 'Power Factor']
    ]
    time_frame_styles = [
        button_style_selected if state['selected_time_frame'] == tf[0].upper() else button_style
        for tf in ['daily', 'weekly', 'monthly']
    ]
    return parameter_styles + time_frame_styles

# Callbacks for updating dropdown options
@app.callback(
    Output('feeder-dropdown', 'options'),
    [Input('substation-dropdown', 'value')]
)
def update_feeder_dropdown(selected_substation):
    return [{'label': k, 'value': k} for k in hierarchy[selected_substation].keys()]

@app.callback(
    Output('transformer-dropdown', 'options'),
    [Input('feeder-dropdown', 'value'),
     Input('substation-dropdown', 'value')]
)
def update_transformer_dropdown(selected_feeder, selected_substation):
    if selected_feeder:
        return [{'label': k, 'value': k} for k in hierarchy[selected_substation][selected_feeder].keys()]
    return []

@app.callback(
    Output('meter-dropdown', 'options'),
    [Input('transformer-dropdown', 'value'),
     Input('feeder-dropdown', 'value'),
     Input('substation-dropdown', 'value')]
)
def update_meter_dropdown(selected_transformer, selected_feeder, selected_substation):
    if selected_transformer:
        return [{'label': meter, 'value': meter} for meter in hierarchy[selected_substation][selected_feeder][selected_transformer]]
    return []

@app.callback(
    Output('customer-id-dropdown', 'options'),
    [Input('meter-dropdown', 'value')]
)
def update_customer_id_dropdown(selected_meter):
    if selected_meter:
        return [{'label': f'Account for {selected_meter}', 'value': f'AA11BB{selected_meter[-3:]}'},]
    return []

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)