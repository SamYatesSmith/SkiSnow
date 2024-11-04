# src/models/utils.py

import os
import sys
import joblib
import pandas as pd
import logging
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

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

    # Configure logging
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

# Initialize logger
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
        if (y_train < 0).any() or (y_val < 0).any() or (y_test < 0).any():
            logger.warning("Negative values detected in target variables. Ensure targets are not log-transformed.")
        
        # Create the save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Save datasets to CSV
        X_train.to_csv(os.path.join(save_dir, 'X_train.csv'), index=False)
        y_train.to_csv(os.path.join(save_dir, 'y_train.csv'), index=False)
        
        X_val.to_csv(os.path.join(save_dir, 'X_val.csv'), index=False)
        y_val.to_csv(os.path.join(save_dir, 'y_val.csv'), index=False)
        
        X_test.to_csv(os.path.join(save_dir, 'X_test.csv'), index=False)
        y_test.to_csv(os.path.join(save_dir, 'y_test.csv'), index=False)
        
        logger.info(f"Split datasets successfully saved to {save_dir}")
    except Exception as e:
        logger.error(f"Failed to save split datasets to {save_dir}: {e}")
        raise e

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
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the model using joblib
        joblib.dump(model, save_path)
        logger.info(f"Model successfully saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save model to {save_path}: {e}")
        raise e

def load_model(load_path):
    """
    Load a trained model from the specified path.
    
    Parameters:
    - load_path (str): Path to the saved model.
    
    Returns:
    - model: Loaded machine learning model.
    """
    try:
        model = joblib.load(load_path)
        logger.info(f"Model successfully loaded from {load_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {load_path}: {e}")
        raise e

def calculate_vif(X):
    """
    Calculate Variance Inflation Factor (VIF) for each feature in the DataFrame.
    
    Parameters:
    - X (pd.DataFrame): Feature matrix.
    
    Returns:
    - pd.DataFrame: DataFrame containing features and their VIF values.
    """
    vif = pd.DataFrame()
    vif['Feature'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif

def remove_high_vif_features(X, vif_threshold=10):
    """
    Iteratively remove features with VIF greater than the threshold.
    
    Parameters:
    - X (pd.DataFrame): Feature matrix.
    - vif_threshold (float): Threshold for VIF to determine feature removal.
    
    Returns:
    - pd.DataFrame: Cleaned feature matrix with VIF below the threshold.
    - pd.DataFrame: DataFrame containing final VIF values.
    """
    while True:
        vif_data = calculate_vif(X)
        high_vif = vif_data[vif_data['VIF'] > vif_threshold]
        if high_vif.empty:
            break
        # Remove the feature with the highest VIF
        feature_to_remove = high_vif.sort_values('VIF', ascending=False)['Feature'].iloc[0]
        logger.info(f"Removing feature '{feature_to_remove}' with VIF={high_vif['VIF'].max():.2f} > {vif_threshold}")
        X = X.drop(columns=[feature_to_remove])
    final_vif = calculate_vif(X)
    return X, final_vif

def handle_non_numeric_features(X):
    """
    Handle non-numeric features by one-hot encoding and converting boolean columns.
    
    Parameters:
    - X (pd.DataFrame): Feature matrix.
    
    Returns:
    - pd.DataFrame: Processed feature matrix with numeric values.
    """
    # Identify non-numeric categorical columns
    non_numeric_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    logger.info(f"Non-numeric columns in X: {non_numeric_cols}")
    
    # One-Hot Encode non-numeric categorical columns
    if non_numeric_cols:
        X = pd.get_dummies(X, columns=non_numeric_cols, drop_first=True)
        logger.info(f"Applied one-hot encoding to columns: {non_numeric_cols}")
    else:
        logger.info("No non-numeric categorical columns to encode.")
    
    # Identify and convert boolean columns to integers
    bool_cols = X.select_dtypes(include=['bool']).columns.tolist()
    if bool_cols:
        X[bool_cols] = X[bool_cols].astype(int)
        logger.info(f"Converted boolean columns to integers: {bool_cols}")
    else:
        logger.info("No boolean columns to convert.")
    
    return X

def handle_missing_values(X):
    """
    Replace infinite values with NaN and fill missing values with column means.
    
    Parameters:
    - X (pd.DataFrame): Feature matrix.
    
    Returns:
    - pd.DataFrame: Processed feature matrix with no missing or infinite values.
    """
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    if X.isnull().sum().sum() > 0:
        X.fillna(X.mean(), inplace=True)
        logger.info("Handled missing and infinite values in X.")
    else:
        logger.info("No missing or infinite values in X.")
    return X

def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.
    
    Parameters:
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training target.
    
    Returns:
    - LinearRegression: Trained Linear Regression model.
    """
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    logger.info("Linear Regression model trained.")
    return lr

def evaluate_model(model, X, y):
    """
    Evaluate the model and return metrics.
    
    Parameters:
    - model: Trained machine learning model.
    - X (pd.DataFrame): Features for prediction.
    - y (pd.Series): True target values.
    
    Returns:
    - float: Mean Squared Error (MSE).
    - float: R-squared (R2).
    - np.ndarray: Predicted target values.
    """
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    logger.info(f"Model evaluation completed: MSE={mse:.2f}, R2={r2:.2f}")
    return mse, r2, y_pred
