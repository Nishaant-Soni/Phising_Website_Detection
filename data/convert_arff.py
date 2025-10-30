"""
Script to convert ARFF file to CSV and XLSX formats
"""
import pandas as pd
import re

def parse_arff_to_dataframe(arff_file_path):
    """
    Parse an ARFF file and convert it to a pandas DataFrame
    
    Args:
        arff_file_path: Path to the ARFF file
        
    Returns:
        pandas DataFrame with the data
    """
    with open(arff_file_path, 'r') as f:
        lines = f.readlines()
    
    # Extract attribute names
    attributes = []
    data_section = False
    data_start_idx = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Check if we've reached the data section
        if line.lower() == '@data':
            data_section = True
            data_start_idx = i + 1
            break
            
        # Extract attribute names
        if line.lower().startswith('@attribute'):
            # Parse attribute line: @attribute name type
            parts = line.split()
            if len(parts) >= 2:
                attr_name = parts[1]
                attributes.append(attr_name)
    
    print(f"Found {len(attributes)} attributes:")
    for i, attr in enumerate(attributes, 1):
        print(f"  {i}. {attr}")
    
    # Extract data rows
    data_rows = []
    for line in lines[data_start_idx:]:
        line = line.strip()
        if line and not line.startswith('%'):  # Skip empty lines and comments
            # Split by comma and convert to appropriate types
            values = [val.strip() for val in line.split(',')]
            if len(values) == len(attributes):
                # Convert to integers where possible
                converted_values = []
                for val in values:
                    try:
                        converted_values.append(int(val))
                    except ValueError:
                        converted_values.append(val)
                data_rows.append(converted_values)
    
    print(f"\nExtracted {len(data_rows)} data rows")
    
    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=attributes)
    
    return df

def main():
    # File paths
    arff_file = 'Training Dataset.arff'
    csv_file = 'Training_Dataset.csv'
    xlsx_file = 'Training_Dataset.xlsx'
    
    print("Converting ARFF file to CSV and XLSX formats...")
    print("=" * 60)
    
    # Parse ARFF file
    df = parse_arff_to_dataframe(arff_file)
    
    # Display DataFrame info
    print("\nDataFrame Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nLast few rows:")
    print(df.tail())
    print("\nData types:")
    print(df.dtypes)
    print("\nBasic statistics:")
    print(df.describe())
    
    # Save to CSV
    print(f"\nSaving to CSV: {csv_file}")
    df.to_csv(csv_file, index=False)
    print("✓ CSV file created successfully")
    
    # Save to XLSX
    print(f"\nSaving to XLSX: {xlsx_file}")
    df.to_excel(xlsx_file, index=False, engine='openpyxl')
    print("✓ XLSX file created successfully")
    
    print("\n" + "=" * 60)
    print("Conversion completed successfully!")
    print(f"  - CSV file: {csv_file}")
    print(f"  - XLSX file: {xlsx_file}")
    print(f"  - Total rows: {len(df)}")
    print(f"  - Total columns: {len(df.columns)}")

if __name__ == "__main__":
    main()
