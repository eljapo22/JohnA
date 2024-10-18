import os
import pandas as pd
import math

# Splitting meters into two transformers roughly evenly
def split_meters(meter_list):
    half = math.ceil(len(meter_list) / 2)
    transformer_bb_meters = meter_list[:half]   # Meters for D/S T/F BB (Area MM)
    transformer_cc_meters = meter_list[half:]   # Meters for D/S T/F CC (Area KK)
    return transformer_bb_meters, transformer_cc_meters

# Load the CleanedData3.csv file to get the unique meter numbers
csv_file_path = r'C:\Users\eljapo22\gephi\Node2Edge2JSON\excel\CleanedData3.csv'
df = pd.read_csv(csv_file_path)
unique_meters = df['MeterSerialNo'].unique()

# Split the meters between Transformer BB (Area MM) and Transformer CC (Area KK)
transformer_bb_meters, transformer_cc_meters = split_meters(unique_meters)

# Base folder where the data will be saved
BASE_FOLDER = 'Data'

# Helper function to create the folder structure
def create_folder_structure(substation, feeder, area, transformer, meter_id):
    """Create folder structure based on the hierarchy."""
    path = os.path.join(BASE_FOLDER, f"Substation_AA", f"Feeder_11", f"Area_{area}", f"Transformer_{transformer}")
    os.makedirs(path, exist_ok=True)
    return path

# Function to aggregate data at different granularities and save to CSV files
def aggregate_and_save_data(df, meter_id, path):
    """Aggregate data at different levels and save as CSV."""
    # Ensure the CreatedDate column is in datetime format
    df.loc[:, 'CreatedDate'] = pd.to_datetime(df['CreatedDate'])

    # Minute-level data (no aggregation)
    minute_folder = os.path.join(path, 'Minute-Level')
    os.makedirs(minute_folder, exist_ok=True)
    df.to_csv(os.path.join(minute_folder, f'meter_{meter_id}.csv'), index=False)

    # Hourly aggregation
    hourly_folder = os.path.join(path, 'Hourly-Granularity')
    os.makedirs(hourly_folder, exist_ok=True)
    df_hourly = df.resample('H', on='CreatedDate').agg({
        'KWH': 'sum',
        'KVAH': 'sum',
        'RPhaseVoltage': 'mean',
        'YPhaseVoltage': 'mean',
        'BPhaseVoltage': 'mean',
        'RPhaseCurrent': 'mean',
        'YPhaseCurrent': 'mean',
        'BPhaseCurrent': 'mean',
        'ActivePower': 'mean',
        'ReactivePower': 'mean',
        'ApparentPower': 'mean',
        'TotalPowerFactor': 'mean',
        'Frequency': 'mean',
        'MaximumDemandKW': 'max',
        'MaximumDemandKVA': 'max'
    }).reset_index()
    df_hourly.to_csv(os.path.join(hourly_folder, f'hourly_meter_{meter_id}.csv'), index=False)

    # Daily aggregation
    daily_folder = os.path.join(path, 'Daily-Granularity')
    os.makedirs(daily_folder, exist_ok=True)
    df_daily = df.resample('D', on='CreatedDate').agg({
        'KWH': 'sum',
        'KVAH': 'sum',
        'RPhaseVoltage': 'mean',
        'YPhaseVoltage': 'mean',
        'BPhaseVoltage': 'mean',
        'RPhaseCurrent': 'mean',
        'YPhaseCurrent': 'mean',
        'BPhaseCurrent': 'mean',
        'ActivePower': 'mean',
        'ReactivePower': 'mean',
        'ApparentPower': 'mean',
        'TotalPowerFactor': 'mean',
        'Frequency': 'mean',
        'MaximumDemandKW': 'max',
        'MaximumDemandKVA': 'max'
    }).reset_index()
    df_daily.to_csv(os.path.join(daily_folder, f'daily_meter_{meter_id}.csv'), index=False)

    # Weekly aggregation
    weekly_folder = os.path.join(path, 'Weekly-Granularity')
    os.makedirs(weekly_folder, exist_ok=True)
    df_weekly = df.resample('W', on='CreatedDate').agg({
        'KWH': 'sum',
        'KVAH': 'sum',
        'RPhaseVoltage': 'mean',
        'YPhaseVoltage': 'mean',
        'BPhaseVoltage': 'mean',
        'RPhaseCurrent': 'mean',
        'YPhaseCurrent': 'mean',
        'BPhaseCurrent': 'mean',
        'ActivePower': 'mean',
        'ReactivePower': 'mean',
        'ApparentPower': 'mean',
        'TotalPowerFactor': 'mean',
        'Frequency': 'mean',
        'MaximumDemandKW': 'max',
        'MaximumDemandKVA': 'max'
    }).reset_index()
    df_weekly.to_csv(os.path.join(weekly_folder, f'weekly_meter_{meter_id}.csv'), index=False)

    # Monthly aggregation
    monthly_folder = os.path.join(path, 'Monthly-Granularity')
    os.makedirs(monthly_folder, exist_ok=True)
    df_monthly = df.resample('MS', on='CreatedDate').agg({
        'KWH': 'sum',
        'KVAH': 'sum',
        'RPhaseVoltage': 'mean',
        'YPhaseVoltage': 'mean',
        'BPhaseVoltage': 'mean',
        'RPhaseCurrent': 'mean',
        'YPhaseCurrent': 'mean',
        'BPhaseCurrent': 'mean',
        'ActivePower': 'mean',
        'ReactivePower': 'mean',
        'ApparentPower': 'mean',
        'TotalPowerFactor': 'mean',
        'Frequency': 'mean',
        'MaximumDemandKW': 'max',
        'MaximumDemandKVA': 'max'
    }).reset_index()
    df_monthly.to_csv(os.path.join(monthly_folder, f'monthly_meter_{meter_id}.csv'), index=False)

# Process the data for each meter and save it in the correct structure
def process_data(df, meter_list, transformer, area):
    for meter_id in meter_list:
        # Filter the data for the specific meter
        df_meter = df[df['MeterSerialNo'] == meter_id]

        # Create the folder structure
        path = create_folder_structure("AA", "11", area, transformer, meter_id)

        # Aggregate and save the data
        aggregate_and_save_data(df_meter, meter_id, path)

# Main function to run the process
if __name__ == "__main__":
    # Process data for Transformer BB (Area MM)
    process_data(df, transformer_bb_meters, "BB", "MM")

    # Process data for Transformer CC (Area KK)
    process_data(df, transformer_cc_meters, "CC", "KK")

    print("Data processing completed!")
