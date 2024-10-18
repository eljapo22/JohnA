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

# Define the button style
button_style = {
    'cursor': 'pointer',
    'padding': '5px 10px',
    'margin': '5px 2px',
    'border': '1px solid #ccc',
    'borderRadius': '5px',
    'backgroundColor': '#f0f0f0',
    'boxShadow': '2px 2px 2px rgba(0,0,0,0.1)',
    'transition': 'all 0.3s ease',
    'fontSize': '16px',
    'textAlign': 'center',
    'width': 'fit-content',
}

button_style_selected = {
    **button_style,
    'backgroundColor': '#e6f2ff',
    'boxShadow': 'inset 2px 2px 2px rgba(0,0,0,0.1)',
    'border': '1px solid #4da6ff',
}

# Read and encode the image
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, 'Assets', 'image (3).png')
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
    # Main container
    html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'flex-start', 'marginBottom': '20px'}, children=[
        # Substation and other dropdowns
        html.Div(id='substation-div', style={'width': '25%', 'padding': '10px'}, children=[
            html.H1('Electrical Infrastructure', style={'fontSize': '24px', 'textAlign': 'center', 'marginBottom': '20px'}),
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
        
        # Electrical infrastructure diagram
        html.Div(style={'width': '50%', 'textAlign': 'center'}, children=[
            html.H2("Electrical Infrastructure Diagram", style={'fontSize': '24px', 'marginBottom': '10px'}),
            html.Img(src=f'data:image/png;base64,{encoded_image}', style={
                'width': '100%',
                'maxWidth': '800px',
                'margin': 'auto'
            })
        ]),
        
        # Electrical Parameters and Time Frame
        html.Div(style={'width': '25%', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'flex-end'}, children=[
            html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'width': '100%'}, children=[
                # Time Frame
                html.Div(children=[
                    html.H1('Time Frame', style={'fontSize': '24px', 'textAlign': 'center', 'marginBottom': '10px'}),
                    html.Div(id='time-frame-div', children=[
                        html.Div(id='daily-div', style=button_style, children=['Daily']),
                        html.Div(id='weekly-div', style=button_style, children=['Weekly']),
                        html.Div(id='monthly-div', style=button_style, children=['Monthly']),
                    ]),
                ]),
                # Electrical Parameters
                html.Div(children=[
                    html.H1('Electrical Parameters', style={'fontSize': '24px', 'textAlign': 'center', 'marginBottom': '10px'}),
                    html.Div(id='parameter-div', children=[
                        html.Div(id='active-power-div', style=button_style, children=['Active Power (kW)']),
                        html.Div(id='current-div', style=button_style, children=['Current (A)']),
                        html.Div(id='voltage-div', style=button_style, children=['Voltage (V)']),
                        html.Div(id='apparent-power-div', style=button_style, children=['Apparent Power (kVA)']),
                        html.Div(id='power-factor-div', style=button_style, children=['Power Factor']),
                    ]),
                ]),
            ]),
        ]),
    ]),

    # Graph
    html.Div(
        dcc.Graph(id='parameter-chart'),
        style={'width': '100%', 'height': '700px', 'margin': '20px 0'}
    ),
    
    # Interaction log
    html.Div(id='interaction-log', style={'marginTop': '20px', 'padding': '10px', 'border': '1px solid #ddd'})
])

# Helper function to generate simulated data based on selection
def generate_data(selected_substation, selected_feeder, selected_transformer, selected_meter, time_frame):
    np.random.seed(42)  # Set a fixed seed for reproducibility
    end_date = pd.Timestamp.now().floor('H')
    
    if time_frame == 'D':
        start_date = end_date - pd.Timedelta(days=2)  # 48 hours for daily view
        time_index = pd.date_range(start=start_date, end=end_date, freq='H')
    elif time_frame == 'W':
        start_date = end_date - pd.Timedelta(weeks=1)
        time_index = pd.date_range(start=start_date, end=end_date, freq='H')
    else:  # Monthly
        start_date = end_date - pd.Timedelta(days=30)
        time_index = pd.date_range(start=start_date, end=end_date, freq='H')

    def base_load(time):
        return 30 + 20 * np.sin(time.hour * 2 * np.pi / 24)

    def generate_transformer_bb_data():
        max_capacity = 60
        data = np.array([base_load(t) for t in time_index])
        morning_peak = (time_index.hour >= 6) & (time_index.hour < 9)
        evening_peak = (time_index.hour >= 17) & (time_index.hour < 20)
        data[morning_peak] *= 1.8  # 90% of max during morning peak
        data[evening_peak] *= 1.9  # 95% of max during evening peak
        weekend_mask = (time_index.dayofweek >= 5)
        data[weekend_mask] *= 0.9  # 90% load on weekends
        return np.clip(data, 0, max_capacity)

    def generate_transformer_cc_data():
        max_capacity = 60
        data = np.array([base_load(t) for t in time_index])
        morning_peak = (time_index.hour >= 7) & (time_index.hour < 9)
        evening_peak = (time_index.hour >= 18) & (time_index.hour < 20)
        data[morning_peak] *= 2.0  # Can exceed capacity
        data[evening_peak] *= 2.2  # Can exceed capacity more
        weekend_mask = (time_index.dayofweek >= 5)
        data[weekend_mask] *= 0.8  # 80% load on weekends
        # Ensure 20% of peak hours exceed capacity
        peak_mask = morning_peak | evening_peak
        exceed_indices = np.random.choice(np.where(peak_mask)[0], int(0.2 * np.sum(peak_mask)), replace=False)
        data[exceed_indices] = np.random.uniform(60, 70, size=len(exceed_indices))
        return data

    def generate_meter_data(transformer_data):
        # Generate meter data as a fraction of transformer data
        fraction = np.random.uniform(0.1, 0.3)
        return transformer_data * fraction * (1 + 0.1 * np.sin(np.arange(len(time_index)) * 2 * np.pi / 24))

    # Generate active power based on selection
    if selected_transformer == 'DT BB':
        active_power = generate_transformer_bb_data()
    elif selected_transformer == 'DT CC':
        active_power = generate_transformer_cc_data()
    else:
        active_power = np.array([base_load(t) for t in time_index])

    if selected_meter:
        active_power = generate_meter_data(active_power)

    # Generate voltage and current data
    voltage_base = 230 + 2 * np.sin(np.arange(len(time_index)) * 2 * np.pi / 24)
    current_base = active_power * 1000 / (voltage_base * np.sqrt(3))

    voltage_r = voltage_base + np.sin(np.arange(len(time_index)) * 2 * np.pi / 12)
    voltage_y = voltage_base + np.sin(np.arange(len(time_index)) * 2 * np.pi / 12 + 2*np.pi/3)
    voltage_b = voltage_base + np.sin(np.arange(len(time_index)) * 2 * np.pi / 12 + 4*np.pi/3)

    current_r = current_base * (1 + 0.05 * np.sin(np.arange(len(time_index)) * 2 * np.pi / 8))
    current_y = current_base * (1 + 0.05 * np.sin(np.arange(len(time_index)) * 2 * np.pi / 8 + 2*np.pi/3))
    current_b = current_base * (1 + 0.05 * np.sin(np.arange(len(time_index)) * 2 * np.pi / 8 + 4*np.pi/3))

    apparent_power = np.sqrt(3) * voltage_base * current_base / 1000  # in kVA
    power_factor = 0.85 + 0.1 * np.sin(np.arange(len(time_index)) * 2 * np.pi / 48)

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

# Function to update the app state
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
        time_frame_mapping = {
            'daily-div': 'D',
            'weekly-div': 'W',
            'monthly-div': 'M'
        }
        new_state['selected_time_frame'] = time_frame_mapping[trigger_id]
        log_entry += f"Time frame set to {new_state['selected_time_frame']}."

    new_log = html.Div([
        html.Div(log_entry),
        html.Br(),
        html.Div(current_log) if current_log else None
    ])

    return new_state, new_log

# Callback to update app state
@app.callback(
    [Output('app-state', 'data'),
     Output('interaction-log', 'children')],
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
def update_app_state(substation, feeder, transformer, meter,
                     active_power_clicks, current_clicks, voltage_clicks,
                     apparent_power_clicks, power_factor_clicks,
                     daily_clicks, weekly_clicks, monthly_clicks,
                     current_state, current_log):
    return update_state(substation, feeder, transformer, meter,
                        active_power_clicks, current_clicks, voltage_clicks,
                        apparent_power_clicks, power_factor_clicks,
                        daily_clicks, weekly_clicks, monthly_clicks,
                        current_state, current_log)










@app.callback(
    Output('parameter-chart', 'figure'),
    Input('app-state', 'data')
)
def update_chart(state):
    if state['selected_parameter'] is None:
        return px.scatter(title="Please select a parameter")

    df = generate_data(state['substation'], state['feeder'], state['transformer'], state['meter'], state['selected_time_frame'])
    parameter = state['selected_parameter']
    
    if parameter == 'Current (A)':
        y_data = df[['Current R', 'Current Y', 'Current B']]
    elif parameter == 'Voltage (V)':
        y_data = df[['Voltage R', 'Voltage Y', 'Voltage B']]
    else:
        y_data = df[parameter]

    fig = go.Figure()

    if isinstance(y_data, pd.DataFrame):
        for column in y_data.columns:
            fig.add_trace(go.Scatter(x=df['Time'], y=y_data[column], mode='lines', name=column))
    else:
        fig.add_trace(go.Scatter(x=df['Time'], y=y_data, mode='lines', name=parameter))

    y_axis_title = parameter
    if parameter == 'Power Factor':
        y_axis_title = 'Power Factor (dimensionless)'

    max_value = y_data.max().max() if isinstance(y_data, pd.DataFrame) else y_data.max()
    min_value = y_data.min().min() if isinstance(y_data, pd.DataFrame) else y_data.min()
    avg_value = y_data.mean().mean() if isinstance(y_data, pd.DataFrame) else y_data.mean()

    fig.add_hline(y=max_value, line_dash="dash", line_color="green", annotation_text="Max", annotation_position="top right")
    fig.add_hline(y=min_value, line_dash="dash", line_color="red", annotation_text="Min", annotation_position="bottom right")
    fig.add_hline(y=avg_value, line_dash="dash", line_color="blue", annotation_text="Avg", annotation_position="top left")

    fig.update_layout(
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
    
    # Add time-of-day highlighting for daily and weekly views
    if state['selected_time_frame'] in ['D', 'W']:
        start_date = df['Time'].min().floor('D')
        end_date = df['Time'].max().ceil('D')
        current_date = start_date
        day_count = 0
        while current_date <= end_date:
            # Morning peak (5 AM to 10 AM)
            morning_start = current_date.replace(hour=5, minute=0)
            morning_end = current_date.replace(hour=10, minute=0)
            fig.add_vrect(
                x0=morning_start, 
                x1=morning_end,
                fillcolor="rgba(255,255,0,0.1)", 
                opacity=0.5, 
                line_width=0
            )
            # Add morning peak annotation
            fig.add_annotation(
                x=morning_start + (morning_end - morning_start) / 2,
                y=1.05 if day_count % 2 == 0 else -0.05,
                text="Morning Peak",
                showarrow=False,
                yref="paper",
                font=dict(size=10)
            )

            # Evening peak (5 PM to 10 PM)
            evening_start = current_date.replace(hour=17, minute=0)
            evening_end = current_date.replace(hour=22, minute=0)
            fig.add_vrect(
                x0=evening_start, 
                x1=evening_end,
                fillcolor="rgba(173,216,230,0.1)", 
                opacity=0.5, 
                line_width=0
            )
            # Add evening peak annotation
            fig.add_annotation(
                x=evening_start + (evening_end - evening_start) / 2,
                y=1.05 if day_count % 2 == 1 else -0.05,
                text="Evening Peak",
                showarrow=False,
                yref="paper",
                font=dict(size=10)
            )

            current_date += pd.Timedelta(days=1)
            day_count += 1

        # Add legend entries
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(size=10, color='rgba(255,255,0,0.1)'),
                                 legendgroup="time", showlegend=True, name='Morning Peak (5 AM - 10 AM)'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                 marker=dict(size=10, color='rgba(173,216,230,0.1)'),
                                 legendgroup="time", showlegend=True, name='Evening Peak (5 PM - 10 PM)'))

    # Add shading for weekdays if weekly or monthly view
    if state['selected_time_frame'] in ['W', 'M']:
        start_date = df['Time'].min().floor('D')
        end_date = df['Time'].max().ceil('D')
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
        fig.update_xaxes(
            dtick=3600000, 
            tickformat='%H:%M', 
            title_text='Time of Day',
            range=[df['Time'].min(), df['Time'].max()]  # Ensure full range is displayed
        )
        # Add day labels
        for i in range(2):
            day_start = df['Time'].min() + pd.Timedelta(days=i)
            fig.add_annotation(
                x=day_start, 
                y=1.05, 
                text=f"Day {i+1}", 
                showarrow=False, 
                xref="x", 
                yref="paper"
            )
    elif state['selected_time_frame'] == 'W':
        fig.update_xaxes(dtick=86400000, tickformat='%a', title_text='Day of Week')
    else:  # Monthly
        fig.update_xaxes(dtick=86400000, tickformat='%d/%m', title_text='Date')
    
    # Adjust y-axis range to accommodate max and min lines
    y_range = max_value - min_value
    fig.update_yaxes(range=[min_value - 0.1*y_range, max_value + 0.1*y_range])

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