import os
import unicodedata
import re
import pandas as pd
import numpy as np

def normalize_name(name):
    """
    Normalize names by converting to lowercase, removing accents, and replacing non-alphanumeric characters with underscores.
    """
    # Convert to lowercase
    name = name.lower()
    # Remove accents and diacritics
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')
    # Replace non-alphanumeric characters with underscores
    name = re.sub(r'[^a-z0-9]+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    return name

def standardize_columns(df):
    """
    Standardize column names based on dataset type.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to standardize.
    
    Returns:
    - pd.DataFrame: Standardized DataFrame.
    """
    # Rename 'time' to 'date'
    df = df.rename(columns={'time': 'date'})
    
    # Ensure 'precipitation_sum' and 'snow_depth' columns exist
    if 'precipitation' in df.columns:
        df = df.rename(columns={'precipitation': 'precipitation_sum'})
    if 'snowfall' in df.columns:
        df = df.rename(columns={'snowfall': 'snow_depth'})
    
    # If 'snow_depth' is missing, add it with NaN values
    if 'snow_depth' not in df.columns:
        df['snow_depth'] = np.nan

    return df

def process_meteostat_data(file_path):
    """
    Process the downloaded Meteostat CSV data.
    
    Parameters:
    - file_path (str): Path to the downloaded CSV file.
    
    Returns:
    - pd.DataFrame: Processed data.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert 'time' to datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Keep only necessary columns
    columns_to_keep = ['time', 'temperature_min', 'temperature_max', 'precipitation_sum', 'snow_depth']
    df = df[columns_to_keep]
    
    # Rename 'time' to 'date'
    df = df.rename(columns={'time': 'date'})
    
    # Handle missing values
    df['temperature_min'] = df['temperature_min'].replace(-9999, np.nan)
    df['temperature_max'] = df['temperature_max'].replace(-9999, np.nan)
    df['precipitation_sum'] = df['precipitation_sum'].replace(-9999, np.nan)
    df['snow_depth'] = df['snow_depth'].replace(-9999, np.nan)
    
    return df

def compile_meteostat_data(resort_name, data_df, compiled_csv_path):
    """
    Compile the downloaded Meteostat CSV data for the resort.
    
    Parameters:
    - resort_name (str): Name of the resort.
    - data_df (pd.DataFrame): The downloaded weather data DataFrame.
    - compiled_csv_path (str): Path to save the compiled CSV file.
    
    Returns:
    - None
    """
    if data_df.empty:
        print(f"No data to compile for {resort_name}.")
        return
    
    print(f"Processing data for {resort_name}...")
    
    # Standardize columns
    processed_df = standardize_columns(data_df)
    
    # Save the processed DataFrame to CSV
    os.makedirs(os.path.dirname(compiled_csv_path), exist_ok=True)
    processed_df.to_csv(compiled_csv_path, index=False)
    print(f"Compiled data saved to {compiled_csv_path}.")