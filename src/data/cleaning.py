import pandas as pd
import os
import numpy as np
from datetime import datetime
from src.data.processing import normalize_name, standardize_columns

def get_all_csv_files_with_metadata(root_dir):
    """
    Traverse the directory structure and collect all CSV files with metadata.
    
    Parameters:
    - root_dir (str): Root directory containing raw data.
    
    Returns:
    - list of dict: List containing metadata for each CSV file.
    """
    exclude_resorts = [
        'french_alps/chamonix',
        'french_alps/val_d_isere_tignes',
        'french_alps/les_trois_vallees',
        'swiss_alps/verbier',
        'swiss_alps/zermatt'
    ]

    csv_files = []
    for country in os.listdir(root_dir):
        country_path = os.path.join(root_dir, country)
        if os.path.isdir(country_path):
            normalized_country = normalize_name(country)
            for resort in os.listdir(country_path):
                resort_path = os.path.join(country_path, resort)
                if os.path.isdir(resort_path):
                    normalized_resort = normalize_name(resort)
                    key = f"{normalized_country}/{normalized_resort}"
                    if key in exclude_resorts:
                        print(f"Excluding resort due to insufficient data: {key}")
                        continue  # Skip this resort
                    for file in os.listdir(resort_path):
                        if file.endswith('.csv'):
                            file_path = os.path.join(resort_path, file)
                            dataset_type = 'unknown'
                            try:
                                df_sample = pd.read_csv(file_path, nrows=1)
                                columns = df_sample.columns.tolist()
                                
                                # Check for 'new' dataset columns
                                new_columns = {'temperature_min', 'temperature_max', 'precipitation_sum', 'snow_depth'}
                                
                                if new_columns.issubset(columns):
                                    dataset_type = 'new'
                                else:
                                    print(f"Skipping 'old' dataset: {file_path}")
                                    continue  # Skip this file
                            except Exception as e:
                                print(f"Error reading {file_path}: {e}")
                                continue
                                
                            csv_files.append({
                                'type': dataset_type,
                                'country': normalized_country,
                                'resort': normalized_resort,
                                'file_path': file_path
                            })
    return csv_files

def clean_and_filter_data(file_info, optional_cutoff_date=None):
    """
    Cleans and filters data based on dataset type.
    
    Parameters:
    - file_info (dict): Information about the file.
    - optional_cutoff_date (str, optional): Date to filter data from.
    
    Returns:
    - key (str): Unique key for the resort.
    - df (pd.DataFrame): Cleaned DataFrame.
    """
    country = file_info['country']
    resort = file_info['resort']
    file_path = file_info['file_path']

    key = f"{country}/{resort}"
    
    try:
        df = pd.read_csv(file_path)
        df = standardize_columns(df)
        
        # Convert 'date' column to datetime format
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])  # Drop rows where date conversion failed
        
        # Apply optional cutoff date filter if specified
        if optional_cutoff_date:
            cutoff_date = pd.to_datetime(optional_cutoff_date)
            df = df[df['date'] >= cutoff_date]

        # Convert snow_depth units conditionally
        if 'snow_depth' in df.columns:
            if key == 'slovenian_alps/kranjska_gora':
                df['snow_depth'] = df['snow_depth'] / 10  # Convert millimeters to centimeters
                print(f"{key}: Converted 'snow_depth' from millimeters to centimeters.")
            else:
                print(f"{key}: 'snow_depth' is assumed to be in centimeters. No conversion applied.")
        
        df = df.reset_index(drop=True)
        return key, df
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

def save_cleaned_data(data_frames, processed_data_root):
    """
    Save cleaned DataFrames to the processed data directory with normalized folder structure.
    
    Parameters:
    - data_frames (dict): Dictionary containing cleaned DataFrames.
    - processed_data_root (str): Root directory to save processed data.
    """
    for key, df in data_frames.items():
        try:
            # Split the key back into country and resort
            country, resort = key.split('/')
            
            # Build the processed data path
            processed_dir = os.path.join(processed_data_root, country, resort)
            os.makedirs(processed_dir, exist_ok=True)
            
            # Generate current timestamp in 'YYYY-MM-DD_HH-MM-SS' format
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            # Define the new file name with timestamp to prevent overwriting
            processed_file_path = os.path.join(processed_dir, f"{resort}_cleaned_{timestamp}.csv")
            
            # Save the cleaned DataFrame to the new CSV file
            df.to_csv(processed_file_path, index=False)
            
            # Informative message indicating successful save
            print(f"Saved cleaned data to {processed_file_path}.")
            
        except Exception as e:
            # Handle potential errors, such as key not having exactly two parts
            print(f"Error saving data for {key}: {e}")