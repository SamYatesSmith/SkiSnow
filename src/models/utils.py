import os
import joblib
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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