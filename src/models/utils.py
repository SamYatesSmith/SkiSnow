import os
import joblib
import pandas as pd
import logging
import sys
from pathlib import Path

def setup_logging():
    """
    Sets up logging with both console and file handlers.
    Ensures that the 'logs' directory exists relative to the project root.
    """
        # Determine the project root directory relative to this file
    project_root = Path(__file__).resolve().parents[2]  # Adjust the number if your structure is different

    # Define the logs directory path
    logs_dir = project_root / 'logs' / 'modeling'
    
    # Create the logs directory if it doesn't exist
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the full path to the log file
    log_file = logs_dir / 'modeling.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file))
        ]
    )
    logger = logging.getLogger(__name__)
    return logger

logger = setup_logging()

def save_split_datasets(X_train, y_train, X_val, y_val, X_test, y_test, save_dir):
    """
    Save the split datasets to the specified directory.
    
    Parameters:
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training target.
    - X_val (pd.DataFrame): Validation features.
    - y_val (pd.Series): Validation target.
    - X_test (pd.DataFrame): Testing features.
    - y_test (pd.Series): Testing target.
    - save_dir (str): Directory to save the datasets.
    
    Returns:
    - None
    """
    try:
        # Optional: Check if the targets are log-transformed
        if y_train.lt(0).any() or y_val.lt(0).any() or y_test.lt(0).any():
            logging.warning("Negative values detected in target variables. Ensure targets are not log-transformed.")
        
        os.makedirs(save_dir, exist_ok=True)
        X_train.to_csv(os.path.join(save_dir, 'X_train.csv'), index=False)
        y_train.to_csv(os.path.join(save_dir, 'y_train.csv'), index=False)
        
        X_val.to_csv(os.path.join(save_dir, 'X_val.csv'), index=False)
        y_val.to_csv(os.path.join(save_dir, 'y_val.csv'), index=False)
        
        X_test.to_csv(os.path.join(save_dir, 'X_test.csv'), index=False)
        y_test.to_csv(os.path.join(save_dir, 'y_test.csv'), index=False)
        
        logging.info(f"Split datasets successfully saved to {save_dir}")
    except Exception as e:
        logging.error(f"Failed to save split datasets to {save_dir}: {e}")

def save_model(model, save_path):
    """
    Save the trained model to the specified path.
    
    Parameters:
    - model: Trained machine learning model.
    - save_path (str): Path to save the model.
    
    Returns:
    - None
    """
    try:
        joblib.dump(model, save_path)
        logging.info(f"Model successfully saved to {save_path}")
    except Exception as e:
        logging.error(f"Failed to save model to {save_path}: {e}")

def save_split_datasets(X_train, y_train, X_val, y_val, X_test, y_test, save_dir):
    """
    Save the split datasets to the specified directory.
    
    Parameters:
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training target.
    - X_val (pd.DataFrame): Validation features.
    - y_val (pd.Series): Validation target.
    - X_test (pd.DataFrame): Testing features.
    - y_test (pd.Series): Testing target.
    - save_dir (str): Directory to save the datasets.
    
    Returns:
    - None
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        X_train.to_csv(os.path.join(save_dir, 'X_train.csv'), index=False)
        y_train.to_csv(os.path.join(save_dir, 'y_train.csv'), index=False)
        
        X_val.to_csv(os.path.join(save_dir, 'X_val.csv'), index=False)
        y_val.to_csv(os.path.join(save_dir, 'y_val.csv'), index=False)
        
        X_test.to_csv(os.path.join(save_dir, 'X_test.csv'), index=False)
        y_test.to_csv(os.path.join(save_dir, 'y_test.csv'), index=False)
        
        logging.info(f"Split datasets successfully saved to {save_dir}")
    except Exception as e:
        logging.error(f"Failed to save split datasets to {save_dir}: {e}")