import pandas as pd
import numpy as np

def detect_snow_depth_anomalies(df, threshold=20):
    """
    Identifies snow_depth values that exceed a specified threshold during the off-season.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing 'snow_depth' and 'is_operating_season'.
    - threshold (float): The maximum plausible snow_depth value during the off-season.
    
    Returns:
    - pd.DataFrame: DataFrame with anomalies flagged in 'snow_depth_anomaly' column.
    """
    df = df.copy()
    
    # Ensure 'is_operating_season' column exists
    if 'is_operating_season' not in df.columns:
        raise ValueError("'is_operating_season' column is missing in the DataFrame.")
    
    # Identify anomalies: snow_depth > threshold during off-season
    off_season_mask = df['is_operating_season'] == False
    anomaly_mask = (df['snow_depth'] > threshold) & off_season_mask
    
    df['snow_depth_anomaly'] = anomaly_mask
    
    return df

def handle_snow_depth_anomalies(df):
    """
    Handles detected snow_depth anomalies by setting them to NaN.
    
    Parameters:
    - df (pd.DataFrame): DataFrame with 'snow_depth_anomaly' column.
    
    Returns:
    - pd.DataFrame: DataFrame with anomalies handled.
    """
    df = df.copy()
    if 'snow_depth_anomaly' in df.columns:
        # Set anomalous snow_depth values to NaN
        df.loc[df['snow_depth_anomaly'], 'snow_depth'] = np.nan
        # Drop the 'snow_depth_anomaly' column if not needed
        df = df.drop(columns=['snow_depth_anomaly'])
    else:
        print("No 'snow_depth_anomaly' column found.")
    return df