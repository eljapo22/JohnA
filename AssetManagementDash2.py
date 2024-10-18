import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy import stats
import dash_extensions as de

# Initialize Dash Application
app = dash.Dash(__name__)

# Load and Preprocess Data
print("Loading data...")
data = pd.read_csv('C:\\Users\\eljapo22\\gephi\\Node2Edge2JSON\\excel\\AugmentatedData(Asha).csv')

# Convert DateTime column to pandas datetime
data['CreatedDate'] = pd.to_datetime(data['CreatedDate'])

# Ensure necessary columns are present
required_columns = ['CreatedDate', 'MeterSerialNo', 'KVAH', 'KWH', 'RPhaseVoltage', 'YPhaseVoltage', 'BPhaseVoltage']
for col in required_columns:
    if col not in data.columns:
        print(f"Warning: Missing required column: {col}")
        data[col] = None  # Add a column with None values if missing

# Set DateTime as index for easier time series manipulation
data.set_index('CreatedDate', inplace=True)

# Sort the index to ensure it is monotonic
data.sort_index(inplace=True)

print("Data loaded and preprocessed.")

# Create dictionaries to store the relationships between entities
meter_to_consumer = data.groupby('MeterSerialNo')['ConsumerID'].first().to_dict() if 'MeterSerialNo' in data.columns and 'ConsumerID' in data.columns else {}

# Function to convert columns to numeric, dropping any non-numeric values
def to_numeric_columns(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=columns)

# Function for adaptive resampling
def adaptive_resample(data, time_granularity):
    if not isinstance(time_granularity, str):
        time_granularity = str(time_granularity)
    
    if time_granularity == 'adaptive':
        time_diff = data.index.to_series().diff().mean()
        
        if time_diff <= pd.Timedelta(hours=1):
            resample_freq = 'H'
        elif time_diff <= pd.Timedelta(days=1):
            resample_freq = 'D'
        elif time_diff <= pd.Timedelta(weeks=1):
            resample_freq = 'W'
        else:
            resample_freq = 'M'
    else:
        resample_freq = time_granularity

    try:
        resampled_data = data.resample(resample_freq).mean()
        return resampled_data, resample_freq
    except ValueError as e:
        print(f"Error in resampling: {e}")
        return data, 'original'

# Layout Design
app.layout = html.Div(children=[
    html.H1(children='AMI Data Dashboard'),

    # Progress Bar
    dcc.Loading(
        id="loading",
        children=[html.Div(id="loading-output")],
        type="default",
    ),

    # Date Picker Range with Preset Buttons
    html.Div([
        dcc.DatePickerRange(
            id='date-picker',
            start_date=data.index.max() - pd.DateOffset(weeks=1),
            end_date=data.index.max(),
            display_format='Y-MM-DD',
        ),
        html.Button('Last Week', id='last-week-button', n_clicks=0),
        html.Button('Last Month', id='last-month-button', n_clicks=0),
        html.Button('Last Year', id='last-year-button', n_clicks=0),
    ], style={'margin-bottom': '20px'}),

    # Dropdown for Meter
    dcc.Dropdown(
        id='meter-dropdown',
        options=[{'label': str(i), 'value': str(i)} for i in data['MeterSerialNo'].unique()[:len(data['MeterSerialNo'].unique())//2] if pd.notna(i)],
        multi=True,
        placeholder="Select Meter"
    ),

    # Time Granularity Selector
    dcc.Dropdown(
        id='time-granularity',
        options=[
            {'label': 'Adaptive', 'value': 'adaptive'},
            {'label': 'Hourly', 'value': 'H'},
            {'label': 'Daily', 'value': 'D'},
            {'label': 'Weekly', 'value': 'W'},
            {'label': 'Monthly', 'value': 'M'}
        ],
        value='adaptive',
        placeholder="Select Time Granularity"
    ),

    # Interpolation Method Selector
    dcc.Dropdown(
        id='interpolation-method',
        options=[
            {'label': 'Linear', 'value': 'linear'},
            {'label': 'Spline', 'value': 'spline'},
            {'label': 'Step', 'value': 'hv'}
        ],
        value='linear',
        placeholder="Select Interpolation Method"
    ),

    # Graphs
    dcc.Graph(id='combined-voltage-line-chart'),
    dcc.Graph(id='voltage-extremes-chart'),
    dcc.Graph(id='meter-voltage-chart'),
    dcc.Graph(id='data-density-heatmap'),
    dcc.Graph(id='r-phase-gauge'),
    dcc.Graph(id='b-phase-gauge'),
    dcc.Graph(id='y-phase-gauge')
])

# Callbacks for updating graphs
@app.callback(
    [Output('combined-voltage-line-chart', 'figure'),
     Output('voltage-extremes-chart', 'figure'),
     Output('meter-voltage-chart', 'figure'),
     Output('data-density-heatmap', 'figure'),
     Output('r-phase-gauge', 'figure'),
     Output('b-phase-gauge', 'figure'),
     Output('y-phase-gauge', 'figure')],
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('meter-dropdown', 'value'),
     Input('time-granularity', 'value'),
     Input('interpolation-method', 'value')]
)
def update_graphs(start_date, end_date, selected_meters, time_granularity, interpolation_method):
    # Filter data based on selections
    mask = (data.index >= start_date) & (data.index <= end_date)
    filtered_data = data.loc[mask]

    if selected_meters:
        filtered_data = filtered_data[filtered_data['MeterSerialNo'].isin(selected_meters)]

    # Create figures
    combined_voltage_line_chart = create_combined_voltage_line_chart(filtered_data, interpolation_method)
    voltage_extremes_chart = create_voltage_extremes_chart(filtered_data)
    meter_voltage_chart = create_meter_voltage_chart(filtered_data, selected_meters, time_granularity, interpolation_method)
    data_density_heatmap = create_data_density_heatmap(filtered_data, time_granularity)
    r_gauge, b_gauge, y_gauge = create_gauge_meters(filtered_data)

    return combined_voltage_line_chart, voltage_extremes_chart, meter_voltage_chart, data_density_heatmap, r_gauge, b_gauge, y_gauge

def create_combined_voltage_line_chart(filtered_data, interpolation_method):
    fig = go.Figure()
    for phase in ['RPhaseVoltage', 'YPhaseVoltage', 'BPhaseVoltage']:
        fig.add_trace(go.Scatter(
            x=filtered_data.index,
            y=filtered_data[phase],
            mode='lines+markers',
            name=phase,
            line=dict(shape=interpolation_method)
        ))
    fig.update_layout(
        title='Combined Voltage Line Chart',
        xaxis_title='DateTime',
        yaxis_title='Voltage (V)',
        hovermode='x unified'
    )
    return fig

def create_voltage_extremes_chart(filtered_data):
    fig = go.Figure()
    for phase in ['RPhaseVoltage', 'YPhaseVoltage', 'BPhaseVoltage']:
        fig.add_trace(go.Box(
            y=filtered_data[phase],
            name=phase,
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))
    fig.update_layout(
        title='Voltage Extremes',
        yaxis_title='Voltage (V)',
        boxmode='group'
    )
    return fig

def create_meter_voltage_chart(filtered_data, selected_meters, time_granularity, interpolation_method):
    fig = go.Figure()

    if selected_meters:
        for meter in selected_meters:
            meter_data = filtered_data[filtered_data['MeterSerialNo'] == meter]
            numeric_columns = ['RPhaseVoltage', 'YPhaseVoltage', 'BPhaseVoltage']
            meter_data = to_numeric_columns(meter_data, numeric_columns)
            resampled_data, _ = adaptive_resample(meter_data[numeric_columns], time_granularity)
            for phase in numeric_columns:
                fig.add_trace(go.Scatter(
                    x=resampled_data.index,
                    y=resampled_data[phase],
                    mode='lines+markers',
                    name=f'Meter {meter} - {phase}',
                    line=dict(shape=interpolation_method)
                ))

        fig.update_layout(
            title='Voltage Over Time by Meter',
            xaxis_title='DateTime',
            yaxis_title='Voltage (V)',
            legend_title='Meter and Phase',
            hovermode='x unified'
        )
    else:
        fig.update_layout(
            title='Select specific meters to view detailed voltage data',
            xaxis_title='DateTime',
            yaxis_title='Voltage (V)'
        )

    return fig

def create_data_density_heatmap(filtered_data, time_granularity):
    if filtered_data.empty:
        return go.Figure().update_layout(title='No data available for heatmap')

    # Handle adaptive resampling
    if time_granularity == 'adaptive':
        time_diff = filtered_data.index.to_series().diff().mean()
        if time_diff <= pd.Timedelta(hours=1):
            resample_freq = 'H'
        elif time_diff <= pd.Timedelta(days=1):
            resample_freq = 'D'
        elif time_diff <= pd.Timedelta(weeks=1):
            resample_freq = 'W'
        else:
            resample_freq = 'M'
    else:
        resample_freq = time_granularity

    try:
        # Resample data to get count of readings
        data_density = filtered_data['KVAH'].resample(resample_freq).count()
        
        # Apply logarithmic scaling
        log_density = np.log1p(data_density.values)
        
        fig = px.imshow(log_density.reshape(1, -1),
                        labels=dict(x="Time", y="", color="Log(Reading Count + 1)"),
                        x=data_density.index,
                        aspect="auto",
                        color_continuous_scale="Viridis")
        
        fig.update_layout(
            title='Data Density Heatmap (Log Scale)',
            xaxis_title='DateTime',
            yaxis_title='',
            coloraxis_colorbar=dict(title='Log(Reading Count + 1)')
        )
        
        return fig
    except Exception as e:
        print(f"Error in creating heatmap: {e}")
        return go.Figure().update_layout(title='Error in creating heatmap')

def create_gauge_meters(filtered_data):
    if filtered_data.empty:
        return go.Figure(), go.Figure(), go.Figure()

    avg_r_phase_voltage = filtered_data['RPhaseVoltage'].mean()
    avg_y_phase_voltage = filtered_data['YPhaseVoltage'].mean()
    avg_b_phase_voltage = filtered_data['BPhaseVoltage'].mean()

    r_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_r_phase_voltage,
        title={'text': "R Phase Voltage"},
        gauge={'axis': {'range': [0, 300]}}
    ))
    b_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_b_phase_voltage,
        title={'text': "B Phase Voltage"},
        gauge={'axis': {'range': [0, 300]}}
    ))
    y_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_y_phase_voltage,
        title={'text': "Y Phase Voltage"},
        gauge={'axis': {'range': [0, 300]}}
    ))
    return r_gauge, b_gauge, y_gauge

# Run the Server
if __name__ == '__main__':
    app.run_server(debug=True)