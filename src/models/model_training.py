# src/models/model_training.py

import pandas as pd
import os
from .utils import (
    calculate_vif,
    remove_high_vif_features,
    handle_non_numeric_features,
    handle_missing_values,
    train_linear_regression,
    evaluate_model,
    save_model,
    save_split_datasets
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def train_and_save_model():
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', '..', 'data', 'processed')
    modeling_data_dir = os.path.join(data_dir, 'modeling_data')
    processed_data_path = os.path.join(data_dir, 'processed_data_for_modeling.csv')
    
    # Load the combined processed data
    df = pd.read_csv(processed_data_path)
    
    # Ensure 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter data for operating season
    data_operating = df[df['is_operating_season'] == True].reset_index(drop=True)
    print(f"Data shape after filtering for operating season: {data_operating.shape}")
    
    # Define the target variable
    y = data_operating['snow_depth']
    
    # Exclude 'date' and 'snow_depth' from features
    feature_columns = [col for col in data_operating.columns if col not in ['date', 'snow_depth']]
    X = data_operating[feature_columns].copy()
    
    print("Excluded 'date' and 'snow_depth' from features.")
    print(f"Final feature columns: {X.columns.tolist()}")
    print(f"Initial shape of X: {X.shape}")
    print(f"Initial shape of y: {y.shape}")
    
    # Feature Engineering: Adding Polynomial Feature
    X['temperature_avg_squared'] = X['temperature_avg'] ** 2
    print("Added 'temperature_avg_squared' to features.")
    
    # Creating Lagged Snow Depth Features
    lags = [1, 7]  # Adjust lag periods as needed
    lag_features = ['snow_depth']
    
    for feature in lag_features:
        for lag in lags:
            lag_col = f"{feature}_lag{lag}"
            if feature in X.columns:
                X[lag_col] = X[feature].shift(lag)
                print(f"Created lag feature '{lag_col}'.")
            else:
                print(f"Feature '{feature}' not found in X. Skipping lag creation for this feature.")
    
    # Drop rows with NaN values resulting from lagging
    X = X.dropna().reset_index(drop=True)
    y = y.iloc[X.index].reset_index(drop=True)
    
    print(f"Shape of X after adding lag features: {X.shape}")
    print(f"Shape of y after adding lag features: {y.shape}")
    
    # Handle non-numeric features
    X = handle_non_numeric_features(X)
    
    # Handle missing and infinite values
    X = handle_missing_values(X)
    
    # Handling Multicollinearity
    X_cleaned, final_vif = remove_high_vif_features(X, vif_threshold=10)
    
    print("Final VIF after removing multicollinear features:")
    print(final_vif.sort_values('VIF', ascending=False))
    
    # Feature Selection Using Feature Importance
    rf_temp = RandomForestRegressor(random_state=42)
    rf_temp.fit(X_cleaned, y)
    
    # Retrieve feature importances
    importances = rf_temp.feature_importances_
    feature_names = X_cleaned.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    
    # Sort features by importance and select the top N features
    N = 20  # Adjust as needed
    top_features = importance_df.sort_values(by='Importance', ascending=False).head(N)['Feature'].tolist()
    
    print(f"Top {N} features selected based on feature importance:")
    print(top_features)
    
    # Filter datasets to include only the top features
    X_train_final = X_cleaned[top_features]
    
    # Splitting the Data into Training, Validation, and Testing Sets
    total_length = len(X_cleaned)
    train_size = int(0.7 * total_length)
    val_size = int(0.15 * total_length)
    test_size = total_length - train_size - val_size
    
    print(f"Total samples: {total_length}")
    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")
    print(f"Test set size: {test_size}")
    
    # Split the data based on indices to maintain temporal order
    X_train = X_train_final.iloc[:train_size].reset_index(drop=True)
    y_train = y.iloc[:train_size].reset_index(drop=True)
    
    X_val = X_train_final.iloc[train_size:train_size + val_size].reset_index(drop=True)
    y_val = y.iloc[train_size:train_size + val_size].reset_index(drop=True)
    
    X_test = X_train_final.iloc[train_size + val_size:].reset_index(drop=True)
    y_test = y.iloc[train_size + val_size:].reset_index(drop=True)
    
    print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Validation set shape: X_val={X_val.shape}, y_val={y_val.shape}")
    print(f"Test set shape: X_test={X_test.shape}, y_test={y_test.shape}")
    
    # Verify alignment
    assert X_train.shape[0] == y_train.shape[0], "Mismatch in training set."
    assert X_val.shape[0] == y_val.shape[0], "Mismatch in validation set."
    assert X_test.shape[0] == y_test.shape[0], "Mismatch in test set."
    
    print("All splits are aligned correctly.")
    
    # Proceed with Linear Regression Modeling
    lr = train_linear_regression(X_train, y_train)
    
    # Predict on validation and test sets
    y_val_pred_lr = lr.predict(X_val)
    y_test_pred_lr = lr.predict(X_test)
    
    # Calculate metrics
    mse_val_lr, r2_val_lr, _ = evaluate_model(lr, X_val, y_val)
    mse_test_lr, r2_test_lr, _ = evaluate_model(lr, X_test, y_test)
    
    print("Linear Regression Evaluation on Validation Set:")
    print(f"Mean Squared Error: {mse_val_lr:.2f}")
    print(f"R-squared: {r2_val_lr:.2f}")
    
    print("\nLinear Regression Evaluation on Test Set:")
    print(f"Mean Squared Error: {mse_test_lr:.2f}")
    print(f"R-squared: {r2_test_lr:.2f}")
    
    # Save the trained Linear Regression model
    model_save_path_lr = os.path.join(model_save_dir, 'linear_regression_model.joblib')
    save_model(lr, model_save_path_lr)
    
    # Save the split datasets using utils.py
    save_split_datasets(X_train_final, y_train, X_val, y_val, X_test, y_test, modeling_data_dir)

if __name__ == "__main__":
    train_and_save_model()
