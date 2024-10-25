import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from src.models.utils import save_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_random_forest(X_train, y_train, param_dist, n_iter=10, cv=3, scoring='neg_mean_squared_error', random_state=42):
    """
    Train a Random Forest Regressor with hyperparameter tuning using RandomizedSearchCV.
    
    Parameters:
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training target.
    - param_dist (dict): Dictionary with parameters names (`str`) as keys and distributions or lists of parameters to try.
    - n_iter (int): Number of parameter settings that are sampled.
    - cv (int): Number of folds in cross-validation.
    - scoring (str): A string or callable to evaluate the predictions on the test set.
    - random_state (int): Controls the randomness of the estimator.
    
    Returns:
    - best_estimator_: Best estimator found by the search.
    - best_params_: Parameter setting that gave the best results on the hold out data.
    """
    rf = RandomForestRegressor(random_state=random_state)
    
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
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
    
    return random_search.best_estimator_, random_search.best_params_

def evaluate_model(model, X, y, dataset_name="Validation"):
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
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    logging.info(f"\n{dataset_name} Set Evaluation:")
    logging.info(f"Mean Squared Error: {mse:.2f}")
    logging.info(f"R-squared: {r2:.2f}")
    
    return {"MSE": mse, "R2": r2}