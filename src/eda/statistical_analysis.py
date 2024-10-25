import pandas as pd
from sklearn.linear_model import LinearRegression

def calculate_correlation(df: pd.DataFrame, column1: str, column2: str) -> float:
    """
    Calculate the Pearson correlation coefficient between two columns.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - column1 (str): The first column name.
    - column2 (str): The second column name.

    Returns:
    - float: The Pearson correlation coefficient.
    """
    return df[column1].corr(df[column2])

def perform_linear_regression(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    """
    Fit a linear regression model.

    Parameters:
    - X (pd.DataFrame): The independent variables.
    - y (pd.Series): The dependent variable.

    Returns:
    - LinearRegression: The fitted regression model.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model

def get_regression_coefficients(model: LinearRegression):
    """
    Retrieve the intercept and coefficients from a linear regression model.

    Parameters:
    - model (LinearRegression): The fitted regression model.

    Returns:
    - dict: Dictionary containing 'intercept' and 'slope'.
    """
    return {
        'intercept': model.intercept_,
        'slope': model.coef_[0] if len(model.coef_) > 0 else None
    }
