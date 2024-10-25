import unittest
from src.eda.statistical_analysis import (
    calculate_correlation,
    perform_linear_regression,
    get_regression_coefficients
)
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class TestStatisticalAnalysisFunctions(unittest.TestCase):
    def test_calculate_correlation(self):
        df = pd.DataFrame({
            'A': [1,2,3,4],
            'B': [2,4,6,8],
            'C': [1,3,5,7]
        })
        corr = calculate_correlation(df, 'A', 'B')
        self.assertEqual(corr, 1.0)
        
        corr = calculate_correlation(df, 'A', 'C')
        self.assertAlmostEqual(corr, 1.0)

    def test_perform_linear_regression(self):
        df = pd.DataFrame({
            'temperature_min': [-5, -3, -2, -4],
            'snow_depth': [10, 15, 20, 25]
        })
        X = df[['temperature_min']]
        y = df['snow_depth']
        model = perform_linear_regression(X, y)
        self.assertIsInstance(model, LinearRegression)
        self.assertEqual(model.coef_.tolist(), [2.0])
        self.assertEqual(model.intercept_, 24.5)

    def test_get_regression_coefficients(self):
        model = LinearRegression()
        model.intercept_ = 35.0
        model.coef_ = [5.0]
        coeffs = get_regression_coefficients(model)
        expected = {'intercept': 35.0, 'slope': 5.0}
        self.assertEqual(coeffs, expected)

if __name__ == '__main__':
    unittest.main()
