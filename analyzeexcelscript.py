import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_excel(file_path):
    # Load the data
    print("Loading data...")
    df = pd.read_csv(file_path)  # Changed to read_csv for CSV files
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")

    # Analyze column types
    print("\nColumn Types:")
    for col in df.columns:
        print(f"{col}: {df[col].dtype}")

    # Identify hierarchical columns
    hierarchy = ['Area', 'Substation', 'Feeder', 'T/F', 'MeterSerialNo']
    
    # Analyze hierarchical structure
    print("\nHierarchical Structure Analysis:")
    for i, level in enumerate(hierarchy):
        if level in df.columns:
            unique_count = df[level].nunique()
            print(f"{level}: {unique_count} unique values")
            if i > 0:
                parent = hierarchy[i-1]
                avg_children = df.groupby(parent)[level].nunique().mean()
                print(f"  Average {level}s per {parent}: {avg_children:.2f}")
        else:
            print(f"{level}: Not found in dataset")

    # Analyze relationships and consistency
    print("\nRelationship Consistency:")
    for i in range(1, len(hierarchy)):
        if hierarchy[i] in df.columns and hierarchy[i-1] in df.columns:
            consistency = df.groupby(hierarchy[i])[hierarchy[i-1]].nunique().max()
            print(f"Max {hierarchy[i-1]}s per {hierarchy[i]}: {consistency}")

    # New section: Detailed relationship analysis
    print("\nDetailed Relationship Analysis:")
    
    # Area to Transformer relationship
    area_transformer = df.groupby('Area')['T/F'].nunique()
    print("\nArea to Transformer relationship:")
    for area, count in area_transformer.items():
        print(f"  Area '{area}' has {count} transformer(s)")
    
    # Transformer to Meter relationship
    transformer_meter = df.groupby('T/F')['MeterSerialNo'].nunique()
    print("\nTransformer to Meter relationship:")
    for transformer, count in transformer_meter.items():
        print(f"  Transformer '{transformer}' serves {count} meter(s)")
    
    # Area to Meter relationship
    area_meter = df.groupby('Area')['MeterSerialNo'].nunique()
    print("\nArea to Meter relationship:")
    for area, count in area_meter.items():
        print(f"  Area '{area}' has {count} meter(s)")
    
    # Detailed meter information
    print("\nDetailed Meter Information:")
    meter_info = df.groupby(['Area', 'T/F', 'MeterSerialNo']).size().reset_index(name='count')
    for _, row in meter_info.iterrows():
        print(f"  Area '{row['Area']}', Transformer '{row['T/F']}', Meter {row['MeterSerialNo']}")

    # Basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print("\nNumeric Column Statistics:")
    print(df[numeric_cols].describe())

    # Correlation analysis for voltage columns
    voltage_cols = [col for col in numeric_cols if 'Voltage' in col]
    if voltage_cols:
        corr = df[voltage_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title("Voltage Correlations")
        plt.tight_layout()
        plt.savefig('voltage_correlations.png')
        print("\nVoltage correlation heatmap saved as 'voltage_correlations.png'")

    # Time series analysis
    if 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        print("\nTime Series Information:")
        print(f"Date Range: {df['DateTime'].min()} to {df['DateTime'].max()}")
        time_diff = df['DateTime'].diff().mode().iloc[0]
        print(f"Most common time difference between readings: {time_diff}")

    # Data quality check
    print("\nData Quality:")
    for col in df.columns:
        missing = df[col].isnull().sum()
        print(f"{col}: {missing} missing values ({missing/len(df)*100:.2f}%)")

    # Voltage analysis
    if voltage_cols:
        print("\nVoltage Analysis:")
        for col in voltage_cols:
            print(f"{col}:")
            print(f"  Range: {df[col].min():.2f} to {df[col].max():.2f}")
            print(f"  Mean: {df[col].mean():.2f}")
            print(f"  Std Dev: {df[col].std():.2f}")

    # Distribution of meters per transformer
    if 'T/F' in df.columns and 'MeterSerialNo' in df.columns:
        meters_per_transformer = df.groupby('T/F')['MeterSerialNo'].nunique()
        print("\nMeters per Transformer:")
        print(f"  Min: {meters_per_transformer.min()}")
        print(f"  Max: {meters_per_transformer.max()}")
        print(f"  Average: {meters_per_transformer.mean():.2f}")

    print("\nAnalysis complete.")

# Usage
file_path = r'C:\Users\eljapo22\gephi\Node2Edge2JSON\excel\CleanedData3.csv'  # Use raw string to handle backslashes
analyze_excel(file_path)