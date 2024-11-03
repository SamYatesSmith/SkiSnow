import matplotlib.pyplot as plt
import seaborn as sns
import os
from .utils import logger
import logging

logger = logging.getLogger(__name__)

def plot_residuals(y_true, y_pred, dataset_name, save_dir):
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
    # Predict on log scale
    y_pred_log = model.predict(X)

    # Inverse transform predictions and true values
    y_true = np.expm1(y_log)
    y_pred = np.expm1(y_pred_log)

    # Calculate residuals
    residuals = y_true - y_pred

    # Proceed with plotting residuals using y_true, y_pred, and residuals
    plot_residuals(y_true, y_pred, residuals, dataset_name, save_dir)