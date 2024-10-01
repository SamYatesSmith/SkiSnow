import requests
import pandas as pd
import time
import os

resorts = [
    # French Alps
    {'name': 'Chamonix', 'latitude': 45.9237, 'longitude': 6.8694, 'country': 'French Alps'},
    {'name': 'Val D\'Isère & Tignes', 'latitude': 45.5608, 'longitude': 6.5833, 'country': 'French Alps'},
    {'name': 'Les Trois Vallées', 'latitude': 45.6000, 'longitude': 6.6200, 'country': 'French Alps'},

    # Austrian Alps
    {'name': 'St. Anton', 'latitude': 47.1298, 'longitude': 11.2655, 'country': 'Austrian Alps'},
    {'name': 'Kitzbühel', 'latitude': 47.4475, 'longitude': 12.3853, 'country': 'Austrian Alps'},
    {'name': 'Sölden', 'latitude': 46.9871, 'longitude': 11.0050, 'country': 'Austrian Alps'},

    # Swiss Alps
    {'name': 'Zermatt', 'latitude': 46.0207, 'longitude': 7.7491, 'country': 'Swiss Alps'},
    {'name': 'St. Moritz', 'latitude': 46.4900, 'longitude': 9.8350, 'country': 'Swiss Alps'},
    {'name': 'Verbier', 'latitude': 46.0985, 'longitude': 7.2261, 'country': 'Swiss Alps'},

    # Italian Alps
    {'name': 'Cortina d\'Ampezzo', 'latitude': 46.5394, 'longitude': 12.1356, 'country': 'Italian Alps'},
    {'name': 'Val Gardena', 'latitude': 46.5519, 'longitude': 11.7602, 'country': 'Italian Alps'},
    {'name': 'Sestriere', 'latitude': 45.4881, 'longitude': 7.6942, 'country': 'Italian Alps'},

    # Slovenian Alps
    {'name': 'Kranjska Gora', 'latitude': 46.3998, 'longitude': 13.6772, 'country': 'Slovenian Alps'},
    {'name': 'Mariborsko Pohorje', 'latitude': 46.5500, 'longitude': 15.8000, 'country': 'Slovenian Alps'},
    {'name': 'Krvavec', 'latitude': 46.2194, 'longitude': 14.0958, 'country': 'Slovenian Alps'}
]