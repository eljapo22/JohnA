import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy import stats

# Initialize Dash Application
app = dash.Dash(__name__)

# Load and Preprocess Data
print("Loading data...")
data = pd.read_excel('C:\\Users\\eljapo22\\Desktop\\AssetManagementV1.ods')

# Convert DateTime column to pandas datetime
data['DateTime'] = pd.to_datetime(data['DateTime'])

# Ensure necessary columns are present
required_columns = ['DateTime', 'Area', 'Substation', 'Feeder', 'T/F', 'MeterSerialNo', 'RPhaseVoltage', 'BPhaseVoltage', 'YPhaseVoltage']
for col in required_columns:
    if col not in data.columns:
        print(f"Warning: Missing required column: {col}")
        data[col] = None  # Add a column with None values if missing

# Set DateTime as index for easier time series manipulation
data.set_index('DateTime', inplace=True)

# Sort the index to ensure it is monotonic
data.sort_index(inplace=True)

print("Data loaded and preprocessed.")

# Create dictionaries to store the relationships between entities
area_to_transformer = data.groupby('Area')['T/F'].unique().to_dict()
transformer_to_meter = data.groupby('T/F')['MeterSerialNo'].unique().to_dict()
area_to_meter = data.groupby('Area')['MeterSerialNo'].unique().to_dict()
meter_to_area = data.groupby('MeterSerialNo')['Area'].first().to_dict()
meter_to_transformer = data.groupby('MeterSerialNo')['T/F'].first().to_dict()

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
        # Calculate the average time difference between consecutive readings
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
            start_date=data.index.min(),
            end_date=data.index.max(),
            display_format='Y-MM-DD',
        ),
        html.Button('Last Week', id='last-week-button', n_clicks=0),
        html.Button('Last Month', id='last-month-button', n_clicks=0),
        html.Button('Last Year', id='last-year-button', n_clicks=0),
    ], style={'margin-bottom': '20px'}),

    # Dropdowns for Area, Transformer, Meter
    dcc.Dropdown(
        id='area-dropdown',
        options=[{'label': i, 'value': i} for i in data['Area'].unique() if i is not None],
        multi=True,
        placeholder="Select Area"
    ),
    dcc.Dropdown(
        id='transformer-dropdown',
        options=[{'label': i, 'value': i} for i in data['T/F'].unique() if i is not None],
        multi=True,
        placeholder="Select Transformer"
    ),
    dcc.Dropdown(
        id='meter-dropdown',
        options=[{'label': i, 'value': i} for i in data['MeterSerialNo'].unique() if i is not None],
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
            {'label': 'Step', 'value': 'hv'},
            {'label': 'Linear', 'value': 'linear'},
            {'label': 'Spline', 'value': 'spline'}
        ],
        value='hv',
        placeholder="Select Interpolation Method"
    ),

    # Combined line chart for phase voltages
    dcc.Graph(id='combined-voltage-line-chart'),

    # Gauges for latest phase voltages
    html.Div([
        dcc.Graph(id='r-gauge'),
        dcc.Graph(id='b-gauge'),
        dcc.Graph(id='y-gauge')
    ], style={'display': 'flex', 'justify-content': 'space-around'}),

    # Bar chart for voltage extremes
    dcc.Graph(id='voltage-extremes-chart'),

    # Meter-specific voltage chart
    dcc.Graph(id='meter-voltage-chart'),

    # Heatmap for data density
    dcc.Graph(id='data-density-heatmap'),

    # Data Quality Indicators
    html.Div(id='data-quality-indicators'),

    # Peak Demand Analysis
    html.Div(id='peak-demand-analysis'),

    # Store component to share data between callbacks
    dcc.Store(id='intermediate-value')
])

# Callback for date range preset buttons
@app.callback(
    [Output('date-picker', 'start_date'),
     Output('date-picker', 'end_date')],
    [Input('last-week-button', 'n_clicks'),
     Input('last-month-button', 'n_clicks'),
     Input('last-year-button', 'n_clicks')]
)
def update_date_range(last_week_clicks, last_month_clicks, last_year_clicks):
    ctx = callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    end_date = pd.Timestamp.now()
    
    if button_id == 'last-week-button':
        start_date = end_date - pd.Timedelta(weeks=1)
    elif button_id == 'last-month-button':
        start_date = end_date - pd.Timedelta(days=30)
    elif button_id == 'last-year-button':
        start_date = end_date - pd.Timedelta(days=365)
    else:
        return dash.no_update, dash.no_update

    return start_date.date(), end_date.date()

# Combined callback for updating dropdown options
@app.callback(
    [Output('area-dropdown', 'options'),
     Output('transformer-dropdown', 'options'),
     Output('meter-dropdown', 'options')],
    [Input('area-dropdown', 'value'),
     Input('transformer-dropdown', 'value'),
     Input('meter-dropdown', 'value')]
)
def update_dropdown_options(selected_areas, selected_transformers, selected_meters):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    if triggered_id == 'area-dropdown':
        if not selected_areas:
            transformer_options = [{'label': i, 'value': i} for i in data['T/F'].unique() if i is not None]
            meter_options = [{'label': i, 'value': i} for i in data['MeterSerialNo'].unique() if i is not None]
        else:
            transformers = set()
            meters = set()
            for area in selected_areas:
                transformers.update(area_to_transformer.get(area, []))
                meters.update(area_to_meter.get(area, []))
            transformer_options = [{'label': i, 'value': i} for i in transformers if i is not None]
            meter_options = [{'label': i, 'value': i} for i in meters if i is not None]
        area_options = [{'label': i, 'value': i} for i in data['Area'].unique() if i is not None]
    
    elif triggered_id == 'transformer-dropdown':
        if not selected_transformers:
            area_options = [{'label': i, 'value': i} for i in data['Area'].unique() if i is not None]
            meter_options = [{'label': i, 'value': i} for i in data['MeterSerialNo'].unique() if i is not None]
        else:
            areas = set()
            meters = set()
            for transformer in selected_transformers:
                areas.update(data[data['T/F'] == transformer]['Area'].unique())
                meters.update(transformer_to_meter.get(transformer, []))
            area_options = [{'label': i, 'value': i} for i in areas if i is not None]
            meter_options = [{'label': i, 'value': i} for i in meters if i is not None]
        transformer_options = [{'label': i, 'value': i} for i in data['T/F'].unique() if i is not None]
    
    elif triggered_id == 'meter-dropdown':
        if not selected_meters:
            area_options = [{'label': i, 'value': i} for i in data['Area'].unique() if i is not None]
            transformer_options = [{'label': i, 'value': i} for i in data['T/F'].unique() if i is not None]
        else:
            areas = set()
            transformers = set()
            for meter in selected_meters:
                areas.add(meter_to_area.get(meter))
                transformers.add(meter_to_transformer.get(meter))
            area_options = [{'label': i, 'value': i} for i in areas if i is not None]
            transformer_options = [{'label': i, 'value': i} for i in transformers if i is not None]
        meter_options = [{'label': i, 'value': i} for i in data['MeterSerialNo'].unique() if i is not None]
    
    else:
        area_options = [{'label': i, 'value': i} for i in data['Area'].unique() if i is not None]
        transformer_options = [{'label': i, 'value': i} for i in data['T/F'].unique() if i is not None]
        meter_options = [{'label': i, 'value': i} for i in data['MeterSerialNo'].unique() if i is not None]

    return area_options, transformer_options, meter_options

# Main callback for updating charts
@app.callback(
    [Output('combined-voltage-line-chart', 'figure'),
     Output('r-gauge', 'figure'),
     Output('b-gauge', 'figure'),
     Output('y-gauge', 'figure'),
     Output('voltage-extremes-chart', 'figure'),
     Output('meter-voltage-chart', 'figure'),
     Output('data-density-heatmap', 'figure'),
     Output('data-quality-indicators', 'children'),
     Output('peak-demand-analysis', 'children'),
     Output('intermediate-value', 'data'),
     Output('loading-output', 'children')],
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('area-dropdown', 'value'),
     Input('transformer-dropdown', 'value'),
     Input('meter-dropdown', 'value'),
     Input('time-granularity', 'value'),
     Input('interpolation-method', 'value')]
)
def update_charts(start_date, end_date, selected_areas, selected_transformers, selected_meters, time_granularity, interpolation_method):
    # Filter data based on date range and selections
    filtered_data = data[(data.index >= start_date) & (data.index <= end_date)]
    
    if selected_areas:
        filtered_data = filtered_data[filtered_data['Area'].isin(selected_areas)]
    if selected_transformers:
        filtered_data = filtered_data[filtered_data['T/F'].isin(selected_transformers)]
    if selected_meters:
        filtered_data = filtered_data[filtered_data['MeterSerialNo'].isin(selected_meters)]
    
    numeric_columns = ['RPhaseVoltage', 'BPhaseVoltage', 'YPhaseVoltage']
    filtered_data = to_numeric_columns(filtered_data, numeric_columns)
    
    # Resample data based on selected time granularity
    grouped_data, actual_granularity = adaptive_resample(filtered_data[numeric_columns], time_granularity)

    # Get latest readings for the gauge meters
    latest_data = filtered_data.iloc[-1]

    # Create figures
    combined_voltage_line_chart = create_combined_voltage_line_chart(grouped_data, interpolation_method)
    r_gauge, b_gauge, y_gauge = create_gauge_meters(latest_data)
    voltage_extremes_chart = create_voltage_extremes_chart(filtered_data)
    meter_voltage_chart = create_meter_voltage_chart(filtered_data, selected_meters, time_granularity, interpolation_method)
    data_density_heatmap = create_data_density_heatmap(filtered_data, time_granularity)

    # Data Quality Indicators
    missing_values = filtered_data.isnull().sum().sum()
    total_values = filtered_data.size
    missing_percentage = (missing_values / total_values) * 100
    outliers = (filtered_data[numeric_columns] > filtered_data[numeric_columns].quantile(0.99)).sum().sum()
    outlier_percentage = (outliers / total_values) * 100
    data_quality_indicators = html.Div([
        html.P(f"Missing Values: {missing_values} ({missing_percentage:.2f}%)"),
        html.P(f"Outliers: {outliers} ({outlier_percentage:.2f}%)")
    ])

    # Peak Demand Analysis
    daily_power = filtered_data.resample('D')['ActivePower'].max()
    peak_demand = daily_power.max()
    peak_date = daily_power.idxmax()
    peak_demand_analysis = html.Div([
        html.P(f"Peak Demand: {peak_demand:.2f} kW on {peak_date.date()}")
    ])

    return (combined_voltage_line_chart, r_gauge, b_gauge, y_gauge, voltage_extremes_chart, 
            meter_voltage_chart, data_density_heatmap, data_quality_indicators, peak_demand_analysis, 
            filtered_data.to_dict('records'), "Data loaded successfully")

# Visualization Functions
def create_combined_voltage_line_chart(grouped_data, interpolation_method):
    fig = go.Figure()
    for col in grouped_data.columns:
        fig.add_trace(go.Scatter(x=grouped_data.index, y=grouped_data[col], mode='lines', name=col))
    fig.update_layout(title='Combined Voltage Line Chart', xaxis_title='DateTime', yaxis_title='Voltage')
    return fig

def create_gauge_meters(latest_data):
    r_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest_data['RPhaseVoltage'],
        title={'text': "R Phase Voltage"},
        gauge={'axis': {'range': [0, 300]}}
    ))
    b_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest_data['BPhaseVoltage'],
        title={'text': "B Phase Voltage"},
        gauge={'axis': {'range': [0, 300]}}
    ))
    y_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest_data['YPhaseVoltage'],
        title={'text': "Y Phase Voltage"},
        gauge={'axis': {'range': [0, 300]}}
    ))
    return r_gauge, b_gauge, y_gauge

def create_voltage_extremes_chart(filtered_data):
    fig = go.Figure()
    for phase in ['RPhaseVoltage', 'BPhaseVoltage', 'YPhaseVoltage']:
        fig.add_trace(go.Bar(
            x=[f'Min {phase}', f'Max {phase}'],
            y=[filtered_data[phase].min(), filtered_data[phase].max()],
            name=phase,
            marker_color={'RPhaseVoltage': 'red', 'BPhaseVoltage': 'black', 'YPhaseVoltage': 'blue'}[phase]
        ))
    fig.update_layout(
        title='Voltage Extremes Across Phases',
        xaxis_title='Phase',
        yaxis_title='Voltage (V)',
        barmode='group'
    )
    return fig

def create_meter_voltage_chart(filtered_data, selected_meters, time_granularity, interpolation_method):
    fig = go.Figure()
    if selected_meters:
        for meter in selected_meters:
            meter_data = filtered_data[filtered_data['MeterSerialNo'] == meter]
            numeric_columns = ['RPhaseVoltage', 'BPhaseVoltage', 'YPhaseVoltage']
            meter_data = to_numeric_columns(meter_data, numeric_columns)
            resampled_data, _ = adaptive_resample(meter_data[numeric_columns], time_granularity)
            for phase in numeric_columns:
                fig.add_trace(go.Scatter(
                    x=resampled_data.index,
                    y=resampled_data[phase],
                    mode='lines+markers',
                    name=f'Meter {meter} - {phase}',
                    line=dict(shape=interpolation_method, color={'RPhaseVoltage': 'red', 'BPhaseVoltage': 'black', 'YPhaseVoltage': 'blue'}[phase])
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
        data_density = filtered_data['RPhaseVoltage'].resample(resample_freq).count()
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

# Run the Server
if __name__ == '__main__':
    app.run_server(debug=True)