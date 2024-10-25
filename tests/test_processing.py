import unittest
import pandas as pd
import numpy as np
from src.data.processing import process_meteostat_data, compile_meteostat_data
import os
from unittest.mock import patch, MagicMock

class TestProcessingFunctions(unittest.TestCase):
    def test_process_meteostat_data(self):
        # Create a sample CSV in-memory
        data = {
            'time': ['1990-01-01', '1990-01-02'],
            'temperature_min': [-5, -3],
            'temperature_max': [5, 7],
            'precipitation_sum': [10, 20],
            'snow_depth': [50, 60],
            'other_column': [1, 2]
        }
        df_input = pd.DataFrame(data)
        csv_path = 'test_meteostat.csv'
        df_input.to_csv(csv_path, index=False)
        
        # Call function
        df_processed = process_meteostat_data(csv_path)
        
        # Expected DataFrame
        expected_data = {
            'date': pd.to_datetime(['1990-01-01', '1990-01-02']),
            'temperature_min': [-5, -3],
            'temperature_max': [5, 7],
            'precipitation_sum': [10, 20],
            'snow_depth': [50, 60]
        }
        df_expected = pd.DataFrame(expected_data)
        
        # Assertions
        pd.testing.assert_frame_equal(df_processed, df_expected)
        
        # Cleanup
        os.remove(csv_path)

    @patch('src.data.processing.process_meteostat_data')
    @patch('os.path.exists')
    @patch('pandas.DataFrame.to_csv')
    def test_compile_meteostat_data(self, mock_to_csv, mock_exists, mock_process):
        # Setup mock
        mock_exists.return_value = True
        mock_process.return_value = pd.DataFrame({
            'date': pd.to_datetime(['1990-01-01', '1990-01-02']),
            'temperature_min': [-5, -3],
            'temperature_max': [5, 7],
            'precipitation_sum': [10, 20],
            'snow_depth': [50, 60]
        })
        
        # Call function
        compile_meteostat_data('austrian_alps/st_anton', '/dummy/raw_dir', '/dummy/compiled.csv')
        
        # Assertions
        mock_process.assert_called_with('/dummy/raw_dir/austrian_alps_st_anton_1990_2023.csv')
        mock_to_csv.assert_called_with('/dummy/compiled.csv', index=False)

    @patch('src.data.processing.process_meteostat_data')
    def test_compile_meteostat_data_file_not_found(self, mock_process):
        # Call function with non-existent file
        compile_meteostat_data('austrian_alps/st_anton', '/dummy/raw_dir', '/dummy/compiled.csv')
        
        # Assertions
        mock_process.assert_not_called()

if __name__ == '__main__':
    unittest.main()
