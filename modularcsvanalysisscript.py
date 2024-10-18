import os
import pandas as pd

# Function to check the structure of a CSV file
def analyze_file(file_path, granularity):
    try:
        df = pd.read_csv(file_path)
        
        # Display basic info about the file
        print(f"\nAnalyzing file: {file_path}")
        print(f"Time Granularity: {granularity}")
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        
        # Display column names and data types
        print("\nColumn Names and Data Types:")
        print(df.dtypes)
        
        # Show the first few rows as an example of the data format
        print("\nSample Data (first 5 rows):")
        print(df.head())
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Function to collect and list meter files for each granularity
def collect_meter_files(base_folder):
    print("\nCollecting meter files by time granularity...\n")
    
    granularities = ["Minute-Level", "Hourly-Granularity", "Daily-Granularity", "Weekly-Granularity", "Monthly-Granularity"]
    
    meter_files = {g: [] for g in granularities}  # Dictionary to store files by granularity
    
    for root, dirs, files in os.walk(base_folder):
        for g in granularities:
            if g in root:
                for file in files:
                    if file.endswith(".csv"):
                        meter_files[g].append(os.path.join(root, file))
    
    # Print out the collected files for each granularity
    for granularity, files in meter_files.items():
        print(f"\nMeter files for {granularity}:")
        if files:
            for file in files:
                print(f"  {file}")
        else:
            print("  No meter files found for this granularity.")

# Function to recursively walk through the folder structure and analyze CSV files
def analyze_folder_structure(base_folder):
    print(f"\nAnalyzing folder structure under: {base_folder}")
    
    granularities = ["Minute-Level", "Hourly-Granularity", "Daily-Granularity", "Weekly-Granularity", "Monthly-Granularity"]
    
    for root, dirs, files in os.walk(base_folder):
        granularity = None
        for g in granularities:
            if g in root:
                granularity = g
                break
        
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                analyze_file(file_path, granularity if granularity else "Unknown Granularity")

# Main function to start the analysis and collect meter files
if __name__ == "__main__":
    base_folder = 'Data'  # Adjust this to the actual root folder path
    
    # Analyze the folder structure and provide insights on each dataset
    analyze_folder_structure(base_folder)
    
    # Collect and display the meter files grouped by time granularity
    collect_meter_files(base_folder)

    print("\nFolder structure analysis complete.")
