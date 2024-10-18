import pandas as pd

# Load the CSV file
file_path = r"C:\Users\eljapo22\gephi\Node2Edge2JSON\excel\AugmentatedData(Asha).csv"
df = pd.read_csv(file_path)

# Group by MeterSerialNo and condense meters by halving the amount of rows for each MeterSerialNo
grouped_df = df.groupby('MeterSerialNo')
condensed_df = grouped_df.apply(lambda x: x.sample(frac=0.5, random_state=42)).reset_index(drop=True)

# Save the condensed dataframe to a new CSV file
output_path = r"C:\Users\eljapo22\gephi\Node2Edge2JSON\excel\CondensedData.csv"
condensed_df.to_csv(output_path, index=False)

print(f"Condensed data saved to {output_path}")
