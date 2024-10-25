import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_actual_vs_predicted(y_true, y_pred, dataset_name="Validation", save_dir=None):
    """
    Plot actual vs. predicted values.
    
    Parameters:
    - y_true (pd.Series or np.ndarray): True target values.
    - y_pred (np.ndarray): Predicted target values.
    - dataset_name (str): Name of the dataset (e.g., "Validation", "Test").
    - save_dir (str): Directory to save the plot. If None, the plot is displayed.
    
    Returns:
    - None
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    sns.lineplot(x=y_true, y=y_true, color='red')  # Diagonal line
    plt.title(f'Actual vs. Predicted Snow Depth ({dataset_name} Set)')
    plt.xlabel('Actual Snow Depth (cm)')
    plt.ylabel('Predicted Snow Depth (cm)')
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'actual_vs_predicted_{dataset_name.lower()}.png')
        plt.savefig(save_path)
        logging.info(f"Actual vs. Predicted plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

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