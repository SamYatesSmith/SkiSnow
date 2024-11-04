import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from .utils import logger
import logging

logger = logging.getLogger(__name__)

def plot_residuals(y_true, y_pred, residuals, dataset_name, save_dir):
    """
    Plots residuals vs predicted values and histogram of residuals.

    Parameters:
    - y_true (pd.Series or np.array): Actual target values.
    - y_pred (pd.Series or np.array): Predicted target values.
    - dataset_name (str): Name of the dataset (e.g., 'Validation', 'Test').
    - save_dir (str): Directory to save the plots.
    """

    # Residuals vs Predicted
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'Residuals vs Predicted Values ({dataset_name} Set)')
    plt.xlabel('Predicted Snow Depth')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.tight_layout()

    # Save the plot
    residuals_plot_path = os.path.join(save_dir, f'residuals_vs_predicted_{dataset_name.lower()}.png')
    plt.savefig(residuals_plot_path)
    plt.close()
    logger.info(f"Saved Residuals vs Predicted plot for {dataset_name} Set at {residuals_plot_path}")

    # Residuals Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30)
    plt.title(f'Distribution of Residuals ({dataset_name} Set)')
    plt.xlabel('Residuals (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.tight_layout()

    # Save the plot
    residuals_dist_path = os.path.join(save_dir, f'residuals_distribution_{dataset_name.lower()}.png')
    plt.savefig(residuals_dist_path)
    plt.close()
    logger.info(f"Saved Residuals Distribution plot for {dataset_name} Set at {residuals_dist_path}")

def perform_residual_analysis(model, X, y, dataset_name, save_dir):
    """
    Generates residual plots for a given dataset.

    Parameters:
    - model: Trained machine learning model.
    - X (pd.DataFrame): Feature set.
    - y (pd.Series or np.array): True target values.
    - dataset_name (str): Name of the dataset (e.g., 'Validation', 'Test').
    - save_dir (str): Directory to save the plots.
    """
    # Predict on original scale
    y_pred = model.predict(X)

    # Calculate residuals
    residuals = y - y_pred

    # Handle potential NaNs/Infs in residuals
    valid_mask = (~np.isnan(residuals)) & (~np.isinf(residuals))
    if not valid_mask.all():
        num_invalid = (~valid_mask).sum()
        logger.warning(f"Found {num_invalid} invalid residual(s) in {dataset_name} Set. These will be excluded from plots.")
        y_true = y[valid_mask]
        y_pred = y_pred[valid_mask]
        residuals = residuals[valid_mask]
    else:
        y_true = y
        y_pred = y_pred

    # Ensure save_dir exists
    os.makedirs(save_dir, exist_ok=True)

    # Plot residuals using the corrected function signature
    plot_residuals(y_true, y_pred, residuals, dataset_name, save_dir)