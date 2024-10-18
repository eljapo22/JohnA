import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import seaborn as sns
import math

# Custom date parser to match the format seen in the CSV (MM/DD/YYYY HH:MM)
def custom_date_parser(date_str):
    return pd.to_datetime(date_str, format='%m/%d/%Y %H:%M')

# Define dtype for each column
dtype_dict = {
    'MeterSerialNo': 'float32',
    'ConsumerID': 'float32',
    'KVAH': 'float32',
    'KWH': 'float32',
    'RPhaseVoltage': 'float32',
    'YPhaseVoltage': 'float32',
    'BPhaseVoltage': 'float32',
    'RPhaseCurrent': 'float32',
    'YPhaseCurrent': 'float32',
    'BPhaseCurrent': 'float32',
    'ActivePower': 'float32',
    'ReactivePower': 'float32',
    'ApparentPower': 'float32',
    'TotalPowerFactor': 'float32',
    'Frequency': 'float32',
    'MaximumDemandKW': 'float32',
    'MaximumDemandKVA': 'float32'
}

# Column names based on the sample data you provided
column_names = [
    'RowIndex', 'MeterSerialNo', 'ConsumerID', 'CreatedDate', 'KVAH', 'KWH', 
    'RPhaseVoltage', 'YPhaseVoltage', 'BPhaseVoltage', 'RPhaseCurrent', 
    'YPhaseCurrent', 'BPhaseCurrent', 'ActivePower', 'ReactivePower', 
    'ApparentPower', 'TotalPowerFactor', 'Frequency', 'MaximumDemandKW', 
    'MaximumDemandKVA'
]

# File path
data_file_path = r"C:\Users\eljapo22\gephi\Node2Edge2JSON\excel\AugmentatedData(Asha).csv"

def load_data():
    print("Loading data...")
    try:
        data = pd.read_csv(
            data_file_path, 
            names=column_names,
            dtype=dtype_dict,
            skiprows=1
        )
        data['CreatedDate'] = pd.to_datetime(data['CreatedDate'], format='%m/%d/%Y %H:%M')
        data.set_index('CreatedDate', inplace=True)
        print("Data loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def check_missing_data(df):
    print("\nChecking for missing data...")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("Missing Data per Column:")
        print(missing[missing > 0])
    else:
        print("No missing data found.")

class GraphNavigator:
    def __init__(self, figures):
        self.figures = figures
        self.current_index = 0
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.2)
        self.show_current_figure()

        ax_prev = plt.axes([0.4, 0.05, 0.1, 0.075])
        ax_next = plt.axes([0.5, 0.05, 0.1, 0.075])
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_prev.on_clicked(self.prev_figure)
        self.btn_next.on_clicked(self.next_figure)

    def show_current_figure(self):
        self.ax.clear()
        current_fig = self.figures[self.current_index]
        for ax in current_fig.axes:
            new_ax = self.fig.add_subplot(len(current_fig.axes), 1, ax.get_subplotspec().num1 + 1)
            new_ax.set_title(ax.get_title())
            new_ax.set_xlabel(ax.get_xlabel())
            new_ax.set_ylabel(ax.get_ylabel())
            for line in ax.lines:
                new_ax.plot(line.get_xdata(), line.get_ydata(), label=line.get_label())
            if ax.collections:
                sns.heatmap(ax.collections[0].get_array().reshape(ax.get_ylim()[0], ax.get_xlim()[0]),
                            cmap=ax.collections[0].get_cmap(), ax=new_ax)
            new_ax.legend()
        self.fig.suptitle(f"Graph {self.current_index + 1} of {len(self.figures)}")
        plt.draw()

    def prev_figure(self, event):
        self.current_index = (self.current_index - 1) % len(self.figures)
        self.show_current_figure()

    def next_figure(self, event):
        self.current_index = (self.current_index + 1) % len(self.figures)
        self.show_current_figure()

def analyze_zero_values(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    zero_value_meters = df.groupby('MeterSerialNo').apply(lambda x: (x[['RPhaseCurrent', 'YPhaseCurrent', 'BPhaseCurrent', 'ActivePower', 'ReactivePower', 'ApparentPower']] == 0).all(axis=1).sum())
    sns.histplot(zero_value_meters, bins=30, kde=True, ax=ax)
    ax.set_title("Distribution of Zero Value Occurrences Across Meters")
    ax.set_xlabel("Number of readings with all zero values")
    ax.set_ylabel("Number of meters")
    return fig

def plot_sample_meter(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    sample_meter = df['MeterSerialNo'].unique()[0]
    sample_data = df[df['MeterSerialNo'] == sample_meter]
    ax.plot(sample_data.index, sample_data['ActivePower'], label='Active Power')
    ax.plot(sample_data.index, sample_data['RPhaseCurrent'], label='R Phase Current')
    ax.set_title(f"Power and Current Readings for Meter {sample_meter}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    return fig

def plot_correlations(df):
    fig, ax = plt.subplots(figsize=(12, 10))
    corr_columns = ['RPhaseVoltage', 'YPhaseVoltage', 'BPhaseVoltage', 'RPhaseCurrent', 'YPhaseCurrent', 'BPhaseCurrent', 'ActivePower', 'ReactivePower', 'ApparentPower', 'TotalPowerFactor']
    correlation_matrix = df[corr_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax)
    ax.set_title("Correlation Between Measurements")
    return fig

def plot_histograms(df):
    metrics = ['RPhaseVoltage', 'YPhaseVoltage', 'BPhaseVoltage', 'RPhaseCurrent', 'YPhaseCurrent', 'BPhaseCurrent', 'ActivePower', 'ReactivePower', 'ApparentPower', 'TotalPowerFactor']
    fig, axes = plt.subplots(5, 2, figsize=(15, 25))
    df[metrics].hist(bins=30, ax=axes.flatten())
    fig.suptitle("Histograms of Key Metrics")
    return fig

def plot_box_plots(df):
    fig, ax = plt.subplots(figsize=(12, 8))
    metrics = ['RPhaseVoltage', 'YPhaseVoltage', 'BPhaseVoltage', 'RPhaseCurrent', 'YPhaseCurrent', 'BPhaseCurrent', 'ActivePower', 'ReactivePower', 'ApparentPower', 'TotalPowerFactor']
    df[metrics].boxplot(ax=ax)
    ax.set_title("Box Plots of Key Metrics")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    return fig

def check_power_factor(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    df['CalculatedPowerFactor'] = df['ActivePower'] / df['ApparentPower']
    ax.plot(df.index, df['TotalPowerFactor'], label='Reported Power Factor', color='blue')
    ax.plot(df.index, df['CalculatedPowerFactor'], label='Calculated Power Factor', color='green', linestyle='--')
    ax.legend()
    ax.set_title('Reported vs Calculated Power Factor')
    ax.set_ylabel('Power Factor')
    ax.set_xlabel('Time')
    return fig

def compare_aggregation(df, column, freq):
    fig, ax = plt.subplots(figsize=(12, 6))
    agg_mean = df[column].resample(freq).mean()
    agg_sum = df[column].resample(freq).sum()
    ax.plot(agg_mean.index, agg_mean, label=f'{column} (Mean)', color='blue')
    ax.plot(agg_sum.index, agg_sum, label=f'{column} (Sum)', color='red', linestyle='--')
    ax.legend()
    ax.set_title(f'{column} Aggregation Comparison - {freq} Frequency')
    ax.set_ylabel(column)
    ax.set_xlabel('Time')
    return fig

def main():
    data = load_data()
    if data is None:
        return

    data.dropna(inplace=True)
    
    check_missing_data(data)

    print("Generating plots...")
    figures = [
        analyze_zero_values(data),
        plot_sample_meter(data),
        plot_correlations(data),
        plot_histograms(data),
        plot_box_plots(data),
        check_power_factor(data)
    ]
    
    # Add aggregation plots for different metrics and frequencies
    for column in ['ActivePower', 'ReactivePower', 'ApparentPower']:
        for freq in ['H', 'D', 'W', 'M']:
            figures.append(compare_aggregation(data, column, freq))

    print(f"Generated {len(figures)} plots.")
    print("Displaying interactive graph navigator...")
    navigator = GraphNavigator(figures)
    plt.show()

if __name__ == "__main__":
    main()