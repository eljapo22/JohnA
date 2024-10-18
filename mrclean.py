import pandas as pd
import numpy as np
from datetime import datetime
import logging
from collections import defaultdict
from bisect import bisect_left
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuration for default values
DEFAULT_VALUES = {
    'RPhaseVoltage': 230.0,
    'YPhaseVoltage': 230.0,
    'BPhaseVoltage': 230.0,
    'ActivePower': 1.0,
    'KWH': 0.1,
    'KVAH': 0.1
}

def clean_and_correct_data(csv_file_path, output_file_path, start_date, end_date):
    try:
        # Convert the start and end dates to datetime objects
        start_time = datetime.strptime(start_date, '%Y-%m-%d')
        end_time = datetime.strptime(end_date, '%Y-%m-%d')

        # Load data from the CSV file
        df = pd.read_csv(csv_file_path, parse_dates=['CreatedDate'])
        
        # Filter data within the date range
        mask = (df['CreatedDate'] >= start_time) & (df['CreatedDate'] <= end_time)
        df = df.loc[mask]

        if df.empty:
            logger.error("No data loaded from the CSV file within the specified date range.")
            return

        logger.info(f"Loaded {len(df)} records from {start_date} to {end_date}")

        # Log initial zero value counts
        initial_zero_counts = (df[DEFAULT_VALUES.keys()] == 0).sum()
        logger.info(f"Initial zero value counts:\n{initial_zero_counts}")

        # Group data by meter
        grouped = df.groupby('MeterSerialNo')

        # Calculate global hourly averages
        global_hourly_averages = calculate_global_hourly_averages(df)

        # Process each meter's records
        corrected_dfs = []
        for meter, meter_df in grouped:
            logger.info(f"Processing meter: {meter} with {len(meter_df)} records.")
            
            # Log zero value counts before correction for this meter
            before_zero_counts = (meter_df[DEFAULT_VALUES.keys()] == 0).sum()
            logger.info(f"Zero value counts before correction for meter {meter}:\n{before_zero_counts}")
            
            # Sort records by timestamp
            meter_df = meter_df.sort_values('CreatedDate')
            
            # Build indices for interpolation optimization
            metric_indices = build_metric_indices(meter_df)
            
            # Correct data for the meter
            corrected_meter_df = correct_data(meter_df, global_hourly_averages, metric_indices)
            
            # Log statistics after correction for this meter
            after_zero_counts = (corrected_meter_df[DEFAULT_VALUES.keys()] == 0).sum()
            logger.info(f"Zero value counts after correction for meter {meter}:\n{after_zero_counts}")
            
            for column in DEFAULT_VALUES.keys():
                if before_zero_counts[column] > 0:
                    corrected_values = corrected_meter_df[corrected_meter_df[column] != 0][column]
                    logger.info(f"Corrected {column} for meter {meter}:")
                    logger.info(f"  Min: {corrected_values.min():.2f}")
                    logger.info(f"  Max: {corrected_values.max():.2f}")
                    logger.info(f"  Mean: {corrected_values.mean():.2f}")
            
            corrected_dfs.append(corrected_meter_df)

        # Combine all corrected data
        corrected_df = pd.concat(corrected_dfs, ignore_index=True)

        # Log final zero value counts
        final_zero_counts = (corrected_df[DEFAULT_VALUES.keys()] == 0).sum()
        logger.info(f"Final zero value counts after all corrections:\n{final_zero_counts}")

        # Save the cleaned data
        corrected_df.to_csv(output_file_path, index=False)
        logger.info(f"Cleaned data saved to {output_file_path}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())

def calculate_global_hourly_averages(df):
    """Calculate global hourly averages for each metric across all meters, excluding zero values."""
    # Replace zero values with NaN to exclude them from the mean calculation
    df_non_zero = df.replace(0, np.nan)
    
    # Group by the hour and calculate the mean, ignoring NaN values
    hourly_averages = df_non_zero.groupby(df['CreatedDate'].dt.hour).mean()
    hourly_averages.index.name = 'Hour'
    
    return hourly_averages.reset_index()

def build_metric_indices(df):
    """Build indices of non-zero records for each metric to optimize interpolation."""
    indices = {}
    for metric in DEFAULT_VALUES.keys():
        indices[metric] = df[df[metric] > 0]['CreatedDate'].tolist()
    return indices

def correct_data(df, global_hourly_averages, metric_indices):
    """Correct zero values, missing data, and outliers for a single meter's records."""
    # Set 'CreatedDate' as the index
    df = df.set_index('CreatedDate')
    
    hourly_averages = df.groupby(df.index.hour).mean()

    columns_to_correct = list(DEFAULT_VALUES.keys()) + ['RPhaseCurrent', 'YPhaseCurrent', 'BPhaseCurrent', 'ReactivePower', 'ApparentPower']

    for metric in columns_to_correct:
        zero_mask = df[metric] == 0
        if zero_mask.any():
            # Interpolate short sequences
            df.loc[zero_mask, metric] = df[metric].interpolate(method='time', limit=2)
            
            # Use time-of-day averages for longer sequences
            still_zero = df[metric] == 0
            if still_zero.any():
                df.loc[still_zero, metric] = df.loc[still_zero].index.hour.map(
                    lambda h: global_hourly_averages[global_hourly_averages['Hour'] == h][metric].iloc[0]
                ).fillna(global_hourly_averages[metric].mean())

    logger.info(f"Corrected data for meter {df['MeterSerialNo'].iloc[0]}")
    
    # Reset the index to make 'CreatedDate' a column again
    return df.reset_index()

def interpolate_value(timestamp, metric, df, metric_index):
    """Interpolate value using pre-built index."""
    if not metric_index:
        return DEFAULT_VALUES.get(metric, 0.1)
    
    pos = bisect_left(metric_index, timestamp)
    before = metric_index[pos - 1] if pos > 0 else None
    after = metric_index[pos] if pos < len(metric_index) else None

    if before and after:
        before_value = df[df['CreatedDate'] == before][metric].iloc[0]
        after_value = df[df['CreatedDate'] == after][metric].iloc[0]
        time_diff = (after - before).total_seconds()
        if time_diff == 0:
            return DEFAULT_VALUES.get(metric, 0.1)
        value_diff = after_value - before_value
        proportion = (timestamp - before).total_seconds() / time_diff
        return before_value + proportion * value_diff

    return DEFAULT_VALUES.get(metric, 0.1)

if __name__ == "__main__":
    csv_file_path = r"C:\Users\eljapo22\gephi\Node2Edge2JSON\excel\CondensedData2.csv"
    output_file_path = r"C:\Users\eljapo22\gephi\Node2Edge2JSON\excel\CleanedData3.csv"
    start_date = "2024-01-01"
    end_date = "2024-12-31"

    clean_and_correct_data(csv_file_path, output_file_path, start_date, end_date)
