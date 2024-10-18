import pandas as pd

# Load the CSV files
df1 = pd.read_csv('C:/Users/eljapo22/gephi/Node2Edge2JSON/excel/CondensedData.csv')
df2 = pd.read_csv('C:/Users/eljapo22/gephi/Node2Edge2JSON/excel/CondensedData2.csv')

# Ensure the 'CreatedDate' column is parsed as datetime
df1['CreatedDate'] = pd.to_datetime(df1['CreatedDate'])
df2['CreatedDate'] = pd.to_datetime(df2['CreatedDate'])

# Compare the two DataFrames
def compare_dataframes(df1, df2):
    # Find rows in df1 that are not in df2
    diff1 = pd.concat([df1, df2, df2]).drop_duplicates(keep=False)
    # Find rows in df2 that are not in df1
    diff2 = pd.concat([df2, df1, df1]).drop_duplicates(keep=False)
    
    return diff1, diff2

# Get the differences
diff1, diff2 = compare_dataframes(df1, df2)

# Print the differences
print("Rows in CondensedData.csv but not in CondensedData2.csv:")
print(diff1)

print("\nRows in CondensedData2.csv but not in CondensedData.csv:")
print(diff2)

# Check for differences in specific columns
def compare_columns(df1, df2, column):
    merged = df1.merge(df2, on='Unnamed: 0', suffixes=('_1', '_2'))
    diff = merged[merged[f'{column}_1'] != merged[f'{column}_2']]
    return diff[[f'Unnamed: 0', f'{column}_1', f'{column}_2']]

# Example: Compare 'KVAH' column
column_diffs = compare_columns(df1, df2, 'KVAH')
print("\nDifferences in 'KVAH' column:")
print(column_diffs)