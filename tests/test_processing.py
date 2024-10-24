import unittest
from src.data.processing import normalize_name, standardize_columns
import pandas as pd

class TestProcessingFunctions(unittest.TestCase):
    def test_normalize_name(self):
        self.assertEqual(normalize_name("St. Anton"), "st_anton")
        self.assertEqual(normalize_name("Val d'Isère & Tignes"), "val_d_isere_tignes")
        self.assertEqual(normalize_name("Kitzbühel"), "kitzbuhel")
        self.assertEqual(normalize_name("Sölden"), "solden")

    def test_standardize_columns(self):
        data = {'time': ['2023-12-01'], 'precipitation': [5], 'snowfall': [10]}
        df = pd.DataFrame(data)
        standardized_df = standardize_columns(df)
        expected_columns = ['date', 'precipitation_sum', 'snow_depth']
        self.assertListEqual(standardized_df.columns.tolist(), expected_columns)
        self.assertEqual(standardized_df['snow_depth'].iloc[0], 10)

if __name__ == '__main__':
    unittest.main()