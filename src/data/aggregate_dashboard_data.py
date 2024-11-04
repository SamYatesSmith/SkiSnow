import pandas as pd
import os
import glob
import numpy as np

def aggregate_dashboard_data(processed_cds_dir, output_file):
    """
    Aggregates all cleaned CSV files into a single dashboard_data.csv,
    calculates 'temperature_avg', extracts the 'month' from the 'date' column,
    and adds a 'country' column based on the 'alps' column.
    
    Parameters:
    - processed_cds_dir (str): Path to the 'data/processed/cds/' directory.
    - output_file (str): Path where 'dashboard_data.csv' will be saved.
    """
    # Define a mapping from 'alps' to 'country'
    alps_to_country = {
        'Austrian Alps': 'Austria',
        'Italian Alps': 'Italy',
        'Slovenian Alps': 'Slovenia',
        'Swiss Alps': 'Switzerland'
    }

    # Initialize an empty list to hold individual DataFrames
    df_list = []
    
    # Traverse through all subdirectories and CSV files
    for alps_dir in os.listdir(processed_cds_dir):
        alps_path = os.path.join(processed_cds_dir, alps_dir)
        if os.path.isdir(alps_path):
            for resort_dir in os.listdir(alps_path):
                resort_path = os.path.join(alps_path, resort_dir)
                if os.path.isdir(resort_path):
                    # Find all cleaned CSV files in the resort directory
                    cleaned_files = glob.glob(os.path.join(resort_path, '*_cleaned_*.csv'))
                    for file in cleaned_files:
                        try:
                            df = pd.read_csv(file)
                            
                            # Add 'alps' column if not present
                            if 'alps' not in df.columns:
                                df['alps'] = alps_dir.replace('_', ' ').title()
                            
                            # Add 'resort' column if not present
                            if 'resort' not in df.columns:
                                df['resort'] = resort_dir.replace('_', ' ').title()
                            
                            # Calculate 'temperature_avg' if 'temperature_min' and 'temperature_max' exist
                            if 'temperature_min' in df.columns and 'temperature_max' in df.columns:
                                df['temperature_avg'] = df[['temperature_min', 'temperature_max']].mean(axis=1)
                            else:
                                print(f"Missing temperature columns in {file}. Skipping 'temperature_avg' calculation.")
                                df['temperature_avg'] = pd.NA  # Assign NA if columns are missing
                            
                            # Map 'alps' to 'country'
                            df['country'] = df['alps'].map(alps_to_country)
                            
                            # Handle cases where mapping might fail
                            missing_countries = df['country'].isna().sum()
                            if missing_countries > 0:
                                print(f"Missing country mapping for 'alps' in {file}.")
                                df['country'].fillna('Unknown', inplace=True)
                            
                            # Append the DataFrame to the list
                            df_list.append(df)
                        except Exception as e:
                            print(f"Error reading {file}: {e}")
    
    if df_list:
        # Concatenate all DataFrames
        dashboard_df = pd.concat(df_list, ignore_index=True)
        
        # Convert 'date' column to datetime
        dashboard_df['date'] = pd.to_datetime(dashboard_df['date'], errors='coerce')
        
        # Extract 'month' from 'date' and create a new 'month' column
        dashboard_df['month'] = dashboard_df['date'].dt.month_name()
        
        # Handle missing 'temperature_avg' values
        if dashboard_df['temperature_avg'].isnull().sum() > 0:
            print(f"Found {dashboard_df['temperature_avg'].isnull().sum()} missing 'temperature_avg' values.")
            mean_temp_avg = dashboard_df['temperature_avg'].mean()
            dashboard_df['temperature_avg'].fillna(mean_temp_avg, inplace=True)
            print(f"Filled missing 'temperature_avg' with mean value: {mean_temp_avg:.2f}")
        
        initial_row_count = len(dashboard_df)
        dashboard_df.dropna(subset=['month'], inplace=True)
        final_row_count = len(dashboard_df)
        dropped_rows = initial_row_count - final_row_count
        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} rows due to invalid or missing 'date' values.")
        
        # Save the updated DataFrame to CSV
        dashboard_df.to_csv(output_file, index=False)
        print(f"Dashboard data successfully saved to {output_file}")
    else:
        print("No cleaned CSV files found to aggregate.")

if __name__ == "__main__":
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_cds_dir = os.path.join(base_dir, '..', '..', 'data', 'processed', 'cds')
    output_file = os.path.join(processed_cds_dir, 'dashboard_data.csv')
    
    # Aggregate data
    aggregate_dashboard_data(processed_cds_dir, output_file)
