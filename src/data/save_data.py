import pandas as pd
import os
from datetime import datetime

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