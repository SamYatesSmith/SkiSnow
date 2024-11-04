# tests/test_model_training.py

import unittest
from unittest.mock import patch, MagicMock
from src.models.model_training import train_random_forest, evaluate_model
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class TestModelTrainingFunctions(unittest.TestCase):
    @patch('src.models.model_training.RandomizedSearchCV')
    def test_train_model(self, mock_random_search):
        # Mocking the RandomizedSearchCV
        mock_estimator = MagicMock(spec=RandomForestRegressor)
        mock_random_search.return_value.best_estimator_ = mock_estimator
        mock_random_search.return_value.best_params_ = {'n_estimators': 200, 'max_depth': 20}
        
        # Sample data
        X_train = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [2, 3, 4, 5]
        })
        y_train = pd.Series([10, 20, 30, 40])
        
        # Parameter distribution
        param_dist = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        
        # Call the function
        best_estimator, best_params = train_model(
            X_train=X_train,
            y_train=y_train,
            param_dist=param_dist,
            n_iter=2,
            cv=2,
            scoring='neg_mean_squared_error',
            random_state=42
        )
        
        # Assertions
        mock_random_search.assert_called_once()
        self.assertEqual(best_estimator, mock_estimator)
        self.assertEqual(best_params, {'n_estimators': 200, 'max_depth': 20})
    
    def test_evaluate_model(self):
        # Sample data
        X = pd.DataFrame({'feature1': [1, 2, 3, 4]})
        y_true = pd.Series([10, 20, 30, 40])
        y_pred = np.array([12, 18, 33, 37])
        dataset_name = "Validation"
        
        # Initialize a mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = y_pred
        
        # Call the function
        metrics = evaluate_model(mock_model, X, y_true, dataset_name=dataset_name)
        
        # Expected metrics
        expected_mse = mean_squared_error(y_true, y_pred)
        expected_r2 = r2_score(y_true, y_pred)
        
        # Assertions
        self.assertAlmostEqual(metrics['MSE'], expected_mse)
        self.assertAlmostEqual(metrics['R2'], expected_r2)
        
        # Ensure that predict was called correctly
        mock_model.predict.assert_called_once_with(X)

if __name__ == '__main__':
    unittest.main()
