import os
import pandas as pd
from meteostat import Stations, Daily
from datetime import datetime
import time
import numpy as np
import sys
import tarfile

# Import existing processing functions
from src.data.processing import normalize_name, standardize_columns

# Path to the raw data directory and compressed file
raw_data_dir = 'data/raw'
compressed_data_file = 'data/raw.tar.gz'

# Check if the raw data directory exists
if not os.path.exists(raw_data_dir) and os.path.exists(compressed_data_file):
    print("Extracting raw data...")
    with tarfile.open(compressed_data_file, 'r:gz') as tar:
        tar.extractall(path='data')
    print("Extraction complete.")

def get_nearest_station(latitude, longitude, start_year, end_year):
    """
    Find the nearest weather station to the given coordinates with available data.
    
    Parameters:
    - latitude (float): Latitude of the location.
    - longitude (float): Longitude of the location.
    - start_year (int): Starting year for data retrieval.
    - end_year (int): Ending year for data retrieval.
    
    Returns:
    - str or None: Station ID if found, else None.
    """
    # Define the time period
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    
    # Search for stations nearby
    stations = Stations()
    stations = stations.nearby(latitude, longitude)
    
    # Filter stations with daily data availability
    stations = stations.inventory('daily', (start, end))
    
    # Fetch the nearest station
    station = stations.fetch(1)
    
    if not station.empty:
        return station.index[0]
    else:
        return None

def download_meteostat_data(resort_name, latitude, longitude, start_year, end_year, output_dir):
    """
    Download daily weather data for the specified resort using Meteostat.
    
    Parameters:
    - resort_name (str): Name of the resort.
    - latitude (float): Latitude of the resort.
    - longitude (float): Longitude of the resort.
    - start_year (int): Starting year for data retrieval.
    - end_year (int): Ending year for data retrieval.
    - output_dir (str): Directory to save the downloaded CSV file.
    
    Returns:
    - None
    """
    # Find the nearest station
    station_id = get_nearest_station(latitude, longitude, start_year, end_year)
    
    if not station_id:
        print(f"No station found for {resort_name} at ({latitude}, {longitude}).")
        return
    
    print(f"Nearest station for {resort_name}: {station_id}")
    
    # Define the time period
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    
    # Fetch daily data
    data = Daily(station_id, start, end)
    data = data.fetch()
    
    if data.empty:
        print(f"No data available for {resort_name} from {start_year} to {end_year}.")
        return
    
    # Select required variables and rename them
    data = data[['tmin', 'tmax', 'prcp', 'snow']]
    data = data.rename(columns={
        'tmin': 'temperature_min',
        'tmax': 'temperature_max',
        'prcp': 'precipitation_sum',
        'snow': 'snow_depth'
    })
    
    # Reset index to have 'time' as a column
    data = data.reset_index()
    
    # Instead of saving to CSV, return the DataFrame
    print(f"Data for {resort_name} fetched successfully.")
    return data

