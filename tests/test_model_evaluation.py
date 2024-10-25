import unittest
from unittest.mock import patch, MagicMock, call
from src.models.model_evaluation import plot_actual_vs_predicted, plot_feature_importances
import pandas as pd
import numpy as np
import os
import tempfile

class TestModelEvaluationFunctions(unittest.TestCase):
    @patch('src.models.model_evaluation.os.makedirs')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('seaborn.scatterplot')
    @patch('seaborn.lineplot')
    def test_plot_actual_vs_predicted(self, mock_lineplot, mock_scatterplot, mock_close, mock_savefig, mock_makedirs):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Sample data
            y_true = pd.Series([10, 20, 30, 40])
            y_pred = np.array([12, 18, 33, 37])
            dataset_name = "Validation"
            save_dir = tmpdirname
            
            # Reset mock_close to ignore any prior calls
            mock_close.reset_mock()
            
            # Call the function
            plot_actual_vs_predicted(
                y_true=y_true,
                y_pred=y_pred,
                dataset_name=dataset_name,
                save_dir=save_dir
            )
            
            # Assertions
            mock_makedirs.assert_called_once_with(save_dir, exist_ok=True)
            mock_scatterplot.assert_called_once_with(x=y_true, y=y_pred, alpha=0.5)
            mock_lineplot.assert_called_once_with(x=y_true, y=y_true, color='red')
            mock_savefig.assert_called_once_with(os.path.join(save_dir, 'actual_vs_predicted_validation.png'))
            
            # Check plt.close() was called once with no arguments
            self.assertIn(call(), mock_close.call_args_list)
    
    @patch('src.models.model_evaluation.os.makedirs')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('seaborn.barplot')
    def test_plot_feature_importances(self, mock_barplot, mock_close, mock_savefig, mock_makedirs):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Sample data
            model = MagicMock()
            model.feature_importances_ = np.array([0.1, 0.3, 0.6])
            feature_names = ['feature1', 'feature2', 'feature3']
            save_dir = tmpdirname
            
            # Reset mock_close to ignore any prior calls
            mock_close.reset_mock()
            
            # Call the function
            plot_feature_importances(
                model=model,
                feature_names=feature_names,
                save_dir=save_dir
            )
            
            # Assertions
            mock_makedirs.assert_called_once_with(save_dir, exist_ok=True)
            mock_barplot.assert_called_once()
            mock_savefig.assert_called_once_with(os.path.join(save_dir, 'feature_importances.png'))
            
            self.assertIn(call(), mock_close.call_args_list)

if __name__ == '__main__':
    unittest.main()