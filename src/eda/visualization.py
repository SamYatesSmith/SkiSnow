import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf

sns.set(style="whitegrid")  # Set a clean theme for seaborn

def plot_boxplot(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str, rotation: int = 45):
    """
    Create and display a box plot.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - x (str): Column name for the x-axis.
    - y (str): Column name for the y-axis.
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - rotation (int): Rotation angle for x-axis labels.

    Returns:
    - None
    """
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=x, y=y, data=df)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()

def plot_violinplot(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str, rotation: int = 45):
    """
    Create and display a violin plot.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - x (str): Column name for the x-axis.
    - y (str): Column name for the y-axis.
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - rotation (int): Rotation angle for x-axis labels.

    Returns:
    - None
    """
    plt.figure(figsize=(12, 6))
    sns.violinplot(x=x, y=y, data=df, inner='quartile')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()

def plot_scatter(df: pd.DataFrame, x: str, y: str, hue: str, title: str, xlabel: str, ylabel: str, alpha: float = 0.5):
    """
    Create and display a scatter plot.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - x (str): Column name for the x-axis.
    - y (str): Column name for the y-axis.
    - hue (str): Column name for color encoding.
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - alpha (float): Transparency level of the points.

    Returns:
    - None
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x, y=y, hue=hue, data=df, alpha=alpha)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_line(df: pd.DataFrame, x: str, y: str, hue: str = None, title: str = "", xlabel: str = "", ylabel: str = ""):
    """
    Create and display a line plot.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - x (str): Column name for the x-axis.
    - y (str): Column name for the y-axis.
    - hue (str, optional): Column name for color encoding.
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.

    Returns:
    - None
    """
    plt.figure(figsize=(12, 8))
    sns.lineplot(x=x, y=y, hue=hue, data=df, marker='o', estimator='mean')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if hue:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df: pd.DataFrame, title: str = ""):
    """
    Create and display a correlation heatmap.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - title (str): Title of the heatmap.

    Returns:
    - None
    """
    corr_matrix = df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_pairplot(df: pd.DataFrame, columns: list, title: str = ""):
    """
    Create and display a pair plot.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - columns (list): List of column names to include in the pair plot.
    - title (str): Title of the pair plot.

    Returns:
    - None
    """
    sns.pairplot(df[columns].dropna(), diag_kind='kde')
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()

def decompose_time_series(series: pd.Series, model: str = 'additive'):
    """
    Perform seasonal decomposition on a time series and plot the results.

    Parameters:
    - series (pd.Series): The time series data.
    - model (str): Type of seasonal component ('additive' or 'multiplicative').

    Returns:
    - decomposition: The result of the decomposition.
    """
    decomposition = seasonal_decompose(series, model=model)
    decomposition.plot()
    plt.tight_layout()
    plt.show()
    return decomposition

def plot_autocorrelation(series: pd.Series, lags: int = 30, title: str = ""):
    """
    Create and display an autocorrelation plot.

    Parameters:
    - series (pd.Series): The time series data.
    - lags (int): Number of lags to display.
    - title (str): Title of the plot.

    Returns:
    - None
    """
    plt.figure(figsize=(10, 6))
    plot_acf(series, lags=lags)
    plt.title(title)
    plt.tight_layout()
    plt.show()
