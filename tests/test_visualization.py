import unittest
from unittest.mock import patch, MagicMock
from src.eda.visualization import (
    plot_boxplot,
    plot_violinplot,
    plot_scatter,
    plot_line,
    plot_correlation_heatmap,
    plot_pairplot,
    decompose_time_series,
    plot_autocorrelation
)
import pandas as pd
import numpy as np

class TestVisualizationFunctions(unittest.TestCase):
    @patch('matplotlib.pyplot.show')
    def test_plot_boxplot(self, mock_show):
        df = pd.DataFrame({
            'resort': ['A', 'A', 'B', 'B'],
            'snow_depth': [10, 15, 20, 25]
        })
        plot_boxplot(df, 'resort', 'snow_depth', 'Test Boxplot', 'Resort', 'Snow Depth')
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_plot_violinplot(self, mock_show):
        df = pd.DataFrame({
            'resort': ['A', 'A', 'B', 'B'],
            'snow_depth': [10, 15, 20, 25]
        })
        plot_violinplot(df, 'resort', 'snow_depth', 'Test Violinplot', 'Resort', 'Snow Depth')
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_plot_scatter(self, mock_show):
        df = pd.DataFrame({
            'temperature_min': [ -5, -3, -2, -4],
            'snow_depth': [10, 15, 20, 25],
            'resort': ['A', 'A', 'B', 'B']
        })
        plot_scatter(df, 'temperature_min', 'snow_depth', 'resort', 'Test Scatterplot', 'Temperature Min', 'Snow Depth')
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_plot_line(self, mock_show):
        df = pd.DataFrame({
            'month': [1,2,3,4],
            'snow_depth': [10,15,20,25],
            'resort': ['A', 'A', 'B', 'B']
        })
        plot_line(df, 'month', 'snow_depth', 'resort', 'Test Line Plot', 'Month', 'Snow Depth')
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_plot_correlation_heatmap(self, mock_show):
        df = pd.DataFrame({
            'temperature_min': [-5, -3, -2, -4],
            'temperature_max': [5, 7, 6, 8],
            'precipitation_sum': [10, 20, 15, 25],
            'snow_depth': [10, 15, 20, 25]
        })
        plot_correlation_heatmap(df, 'Test Correlation Heatmap')
        mock_show.assert_called_once()

    @patch('seaborn.pairplot')
    @patch('matplotlib.pyplot.show')
    def test_plot_pairplot(self, mock_show, mock_pairplot):
        df = pd.DataFrame({
            'temperature_min': [-5, -3, -2, -4],
            'temperature_max': [5, 7, 6, 8],
            'precipitation_sum': [10, 20, 15, 25],
            'snow_depth': [10, 15, 20, 25]
        })
        # Call the function
        plot_pairplot(
            df=df,
            columns=['temperature_min', 'temperature_max', 'precipitation_sum', 'snow_depth'],
            title='Test Pair Plot'
        )

        # Assert that pairplot was called once
        mock_pairplot.assert_called_once()

        # Retrieve the call arguments
        args, kwargs = mock_pairplot.call_args

        # Check that the DataFrame has the correct columns
        passed_df = args[0]
        expected_columns = ['temperature_min', 'temperature_max', 'precipitation_sum', 'snow_depth']
        self.assertListEqual(list(passed_df.columns), expected_columns)

        # Check that diag_kind is set to 'kde'
        self.assertEqual(kwargs.get('diag_kind'), 'kde')

        # Assert that plt.show() was called
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_decompose_time_series(self, mock_show):
        rng = pd.date_range('2020-01-01', periods=100, freq='D')
        ts = pd.Series(np.random.randn(len(rng)), index=rng)
        decompose_time_series(ts, model='additive')
        mock_show.assert_called()

    @patch('matplotlib.pyplot.show')
    def test_plot_autocorrelation(self, mock_show):
        rng = pd.date_range('2020-01-01', periods=100, freq='D')
        ts = pd.Series(np.random.randn(len(rng)), index=rng)
        plot_autocorrelation(ts, lags=30, title='Test Autocorrelation')
        mock_show.assert_called_once()

if __name__ == '__main__':
    unittest.main()
