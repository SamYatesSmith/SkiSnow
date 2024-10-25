import unittest
from unittest.mock import patch, MagicMock
from src.models.utils import save_model, save_split_datasets
import pandas as pd
import os

class TestUtilsFunctions(unittest.TestCase):
    @patch('src.models.utils.joblib.dump')
    @patch('src.models.utils.logging.info')
    @patch('src.models.utils.logging.error')
    def test_save_model_success(self, mock_error, mock_info, mock_joblib_dump):
        # Sample model
        mock_model = MagicMock()
        save_path = "/dummy/model.pkl"
        
        # Call the function
        save_model(mock_model, save_path)
        
        # Assertions
        mock_joblib_dump.assert_called_once_with(mock_model, save_path)
        mock_info.assert_called_once_with(f"Model successfully saved to {save_path}")
        mock_error.assert_not_called()
    
    @patch('src.models.utils.joblib.dump', side_effect=Exception("Save failed"))
    @patch('src.models.utils.logging.info')
    @patch('src.models.utils.logging.error')
    def test_save_model_failure(self, mock_error, mock_info, mock_joblib_dump):
        # Sample model
        mock_model = MagicMock()
        save_path = "/dummy/model.pkl"
        
        # Call the function
        save_model(mock_model, save_path)
        
        # Assertions
        mock_joblib_dump.assert_called_once_with(mock_model, save_path)
        mock_info.assert_not_called()
        mock_error.assert_called_once_with(f"Failed to save model to {save_path}: Save failed")
    
    @patch('src.models.utils.pd.DataFrame.to_csv')
    @patch('src.models.utils.pd.Series.to_csv')
    @patch('src.models.utils.os.makedirs', side_effect=Exception("Make directory failed"))
    @patch('src.models.utils.logging.info')
    @patch('src.models.utils.logging.error')
    def test_save_split_datasets_failure(self, mock_error, mock_info, mock_makedirs, mock_series_to_csv, mock_df_to_csv):
        # Sample datasets
        X_train = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        y_train = pd.Series([10, 20])
        X_val = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
        y_val = pd.Series([30, 40])
        X_test = pd.DataFrame({'A': [9, 10], 'B': [11, 12]})
        y_test = pd.Series([50, 60])
        save_dir = "/dummy/save_dir"
        
        # Call the function
        save_split_datasets(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            save_dir=save_dir
        )
        
        # Assertions
        mock_makedirs.assert_called_once_with(save_dir, exist_ok=True)
        mock_df_to_csv.assert_not_called()
        mock_series_to_csv.assert_not_called()
        mock_error.assert_called_once_with(f"Failed to save split datasets to {save_dir}: Make directory failed")
        mock_info.assert_not_called()

if __name__ == '__main__':
    unittest.main()