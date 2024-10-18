import csv
from datetime import datetime

def is_valid_date(date_string):
    try:
        datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
        return True
    except ValueError:
        return False

filename = r"C:\Users\eljapo22\gephi\Node2Edge2JSON\excel\CleanedData.csv"  # Note the 'r' before the string

try:
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        
        # Read and print header
        header = next(csv_reader)
        print("CSV Header:", ','.join(header))
        print("Expected field count:", len(header))
        
        print("\nFirst 5 lines of data:")
        for i in range(5):
            try:
                line = next(csv_reader)
                print(f"Line {i+1}: {','.join(line)}")
                if line:
                    print(f"  Date field: {line[0]}")
                    print(f"  Is valid date? {'Yes' if is_valid_date(line[0]) else 'No'}")
                    print(f"  Field count: {len(line)}")
                    if len(line) != len(header):
                        print("  WARNING: Unexpected field count!")
            except StopIteration:
                break
        
        # Reset file pointer
        file.seek(0)
        next(csv_reader)  # Skip header
        
        line_count = 0
        error_count = 0
        for line in csv_reader:
            line_count += 1
            if len(line) != len(header):
                print(f"Line {line_count}: Unexpected field count. Expected: {len(header)}, Got: {len(line)}")
                error_count += 1
            if not is_valid_date(line[0]):
                print(f"Line {line_count}: Invalid date format: {line[0]}")
                error_count += 1
            
            if line_count >= 1000:
                break
        
        print("\nAnalysis complete.")
        print(f"Lines checked: {line_count}")
        print(f"Errors found: {error_count}")

except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
except Exception as e:
    print(f"An error occurred: {str(e)}")