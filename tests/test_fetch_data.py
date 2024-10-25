import unittest
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock
from src.data.fetch_data import get_nearest_station, download_meteostat_data

class TestFetchDataFunctions(unittest.TestCase):
    @patch('src.data.fetch_data.Stations')
    def test_get_nearest_station(self, mock_stations):
        # Setup mock
        mock_instance = mock_stations.return_value
        mock_instance.nearby.return_value = mock_instance
        mock_instance.inventory.return_value = mock_instance
        mock_instance.fetch.return_value = pd.DataFrame({
            'id': ['12345'],
            'name': ['Test Station']
        }, index=['12345'])
        
        # Call function
        station_id = get_nearest_station(47.1787, 10.3143, 1990, 2023)
        
        # Assertions
        self.assertEqual(station_id, '12345')
        mock_instance.nearby.assert_called_with(47.1787, 10.3143)
        mock_instance.inventory.assert_called_with('daily', (datetime(1990, 1, 1), datetime(2023, 12, 31)))
        mock_instance.fetch.assert_called_with(1)

if __name__ == '__main__':
    unittest.main()