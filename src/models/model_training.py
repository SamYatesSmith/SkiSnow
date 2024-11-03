import pandas as pd
import logging
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from src.models.utils import save_model
from sklearn.ensemble import RandomForestRegressor 


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(model, X_train, y_train, param_dist, n_iter, cv_splitter, scoring='neg_mean_squared_error', random_state=42):
    """
    Train a machine learning model with hyperparameter tuning using RandomizedSearchCV.
    
    Parameters:
    - model: The machine learning model to train (e.g., RandomForestRegressor).
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training target.
    - param_dist (dict): Dictionary with parameter names (`str`) as keys and distributions or lists of parameters to try.
    - n_iter (int): Number of parameter settings that are sampled.
    - cv_splitter: Cross-validation splitter (e.g., TimeSeriesSplit).
    - scoring (str): A string or callable to evaluate the predictions on the test set.
    - random_state (int): Controls the randomness of the estimator.
    
    Returns:
    - best_estimator_: Best estimator found by the search.
    - best_params_: Parameter setting that gave the best results on the hold out data.
    - best_score_: Best score achieved.
    """
    
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv_splitter,
        scoring=scoring,
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )
    
    logging.info("Starting RandomizedSearchCV for Random Forest Regressor...")
    random_search.fit(X_train, y_train)
    logging.info("RandomizedSearchCV completed.")
    
    logging.info(f"Best Parameters: {random_search.best_params_}")
    logging.info(f"Best Score: {random_search.best_score_}")
    
    return random_search.best_estimator_, random_search.best_params_, random_search.best_score_

def evaluate_model(model, X, y, dataset_name="Validation", offset=0):
    """
    Evaluate the model's performance on a given dataset.
    
    Parameters:
    - model: Trained machine learning model.
    - X (pd.DataFrame): Features of the dataset.
    - y (pd.Series): True target values.
    - dataset_name (str): Name of the dataset (e.g., "Validation", "Test").
    
    Returns:
    - dict: Dictionary containing MSE and R-squared scores.
    """
    # Make predictions (predictions are on log scale)
    predictions_log = model.predict(X)

    # Inverse transform predictions and true values back to original scale
    predictions = np.exp(predictions_log) - offset
    y_true = np.exp(y) - offset

    # Calculate evaluation metrics on original scale
    mse = mean_squared_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)
    
    logging.info(f"\n{dataset_name} Set Evaluation:")
    logging.info(f"Mean Squared Error: {mse:.2f}")
    logging.info(f"R-squared: {r2:.2f}")
    
    return {"MSE": mse, "R2": r2}