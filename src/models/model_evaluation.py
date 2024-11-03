import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_actual_vs_predicted(y_true_log, y_pred_log, dataset_name="Validation", save_dir=None, offset=0):
    """
    Plot actual vs. predicted values.
    
    Parameters:
    - y_true (pd.Series or np.ndarray): True target values.
    - y_pred (np.ndarray): Predicted target values.
    - dataset_name (str): Name of the dataset (e.g., "Validation", "Test").
    - save_dir (str): Directory to save the plot. If None, the plot is displayed.
    - offset (float): Offset added during log transformation.
    
    Returns:
    - None
    """

    # Inverse transform the log-transformed values
    y_true = np.exp(y_true_log) - offset
    y_pred = np.exp(y_pred_log) - offset

    # Existing plotting code using y_true and y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual Snow Depth')
    plt.ylabel('Predicted Snow Depth')
    plt.title(f'Actual vs Predicted Snow Depth ({dataset_name} Set)')
    plt.tight_layout()

    if save_dir is not None:
        plot_path = os.path.join(save_dir, f'actual_vs_predicted_{dataset_name.lower()}.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Saved Actual vs Predicted plot for {dataset_name} Set at {plot_path}")
    else:
        plt.show()


def plot_feature_importances(model, feature_names, save_dir=None):
    """
    Plot the feature importances from the model.
    
    Parameters:
    - model: Trained machine learning model with feature_importances_ attribute.
    - feature_names (list): List of feature names.
    - save_dir (str): Directory to save the plot. If None, the plot is displayed.
    
    Returns:
    - None
    """

    # Convert log-transformed predictions and true values back to original scale
    y_true = np.exp(y_true_log) - offset
    y_pred = np.exp(y_pred_log) - offset

    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=sorted_importances, y=sorted_features, palette='viridis')
    plt.title('Feature Importances')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'feature_importances.png')
        plt.savefig(save_path)
        logging.info(f"Feature importances plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()