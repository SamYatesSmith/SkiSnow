import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import os
import time

# Setup the Open-Meteo API client with cache and retry
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# List of resorts and their coordinates
resorts = [
    {'name': 'Chamonix', 'latitude': 45.9237, 'longitude': 6.8694, 'country': 'French Alps'},
    {'name': 'Val D\'Isère & Tignes', 'latitude': 45.5608, 'longitude': 6.5833, 'country': 'French Alps'},
    {'name': 'Les Trois Vallées', 'latitude': 45.6000, 'longitude': 6.6200, 'country': 'French Alps'},
    {'name': 'St. Anton', 'latitude': 47.1298, 'longitude': 11.2655, 'country': 'Austrian Alps'},
    {'name': 'Kitzbühel', 'latitude': 47.4475, 'longitude': 12.3853, 'country': 'Austrian Alps'},
    {'name': 'Sölden', 'latitude': 46.9871, 'longitude': 11.0050, 'country': 'Austrian Alps'},
    {'name': 'Zermatt', 'latitude': 46.0207, 'longitude': 7.7491, 'country': 'Swiss Alps'},
    {'name': 'St. Moritz', 'latitude': 46.4900, 'longitude': 9.8350, 'country': 'Swiss Alps'},
    {'name': 'Verbier', 'latitude': 46.0985, 'longitude': 7.2261, 'country': 'Swiss Alps'},
    {'name': 'Cortina d\'Ampezzo', 'latitude': 46.5394, 'longitude': 12.1356, 'country': 'Italian Alps'},
    {'name': 'Val Gardena', 'latitude': 46.5519, 'longitude': 11.7602, 'country': 'Italian Alps'},
    {'name': 'Sestriere', 'latitude': 45.4881, 'longitude': 7.6942, 'country': 'Italian Alps'},
    {'name': 'Kranjska Gora', 'latitude': 46.3998, 'longitude': 13.6772, 'country': 'Slovenian Alps'},
    {'name': 'Mariborsko Pohorje', 'latitude': 46.5500, 'longitude': 15.8000, 'country': 'Slovenian Alps'},
    {'name': 'Krvavec', 'latitude': 46.2194, 'longitude': 14.0958, 'country': 'Slovenian Alps'}
]

# Date range for historical data
start_date = '2019-09-01'
end_date = '2024-06-01'

# Key parameters for skiing conditions
daily_params = ["temperature_2m_max", "temperature_2m_min", "rain_sum", "snowfall_sum"]

# Directory to store the data
data_dir = 'data/raw/cds'

# Function to fetch data for a single resort
def fetch_resort_data(resort):
    name = resort['name']
    lat = resort['latitude']
    lon = resort['longitude']
    country = resort['country']
    
    print(f"Fetching data for {name} in {country}...")

    # Constructing API URL
    base_url = 'https://historical-forecast-api.open-meteo.com/v1/forecast'
    
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'daily': daily_params,
        'wind_speed_unit': "mph",
        'precipitation_unit': "inch",
        'timezone': 'auto'
    }
    
    # Fetch the data from the API
    response = openmeteo.weather_api(base_url, params=params)
    
    if response:
        # Process daily data
        daily = response[0].Daily()
        daily_data = {
            "date": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left"
            ),
            "temperature_2m_max": daily.Variables(0).ValuesAsNumpy(),
            "temperature_2m_min": daily.Variables(1).ValuesAsNumpy(),
            "rain_sum": daily.Variables(2).ValuesAsNumpy(),
            "snowfall_sum": daily.Variables(3).ValuesAsNumpy()
        }
        
        # Convert to DataFrame and save to CSV
        df = pd.DataFrame(daily_data)
        resort_dir = os.path.join(data_dir, country.replace(' ', '_').lower(), name.replace(' ', '_').lower())
        if not os.path.exists(resort_dir):
            os.makedirs(resort_dir)
        file_path = os.path.join(resort_dir, f"{name.replace(' ', '_').lower()}.csv")
        df.to_csv(file_path, index=False)
        print(f"Data for {name} saved to {file_path}")
    else:
        print(f"Failed to fetch data for {name}")

# Fetch data for all resorts
def fetch_all_resorts():
    for resort in resorts:
        fetch_resort_data(resort)
        time.sleep(65) # Introduce a 65-second delay to avoid rate limits

# Start data fetching process
fetch_all_resorts()

print("Data fetching completed.")
