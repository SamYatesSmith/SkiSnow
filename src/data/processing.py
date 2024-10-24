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