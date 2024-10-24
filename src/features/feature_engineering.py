import pandas as pd
from datetime import datetime

def get_season_dates(year, open_mm_dd, close_mm_dd):
    """
    Given a year and open/close month-day strings, return datetime objects for open and close dates.
    Handles seasons that span across years.
    """
    open_month, open_day = map(int, open_mm_dd.split('-'))
    close_month, close_day = map(int, close_mm_dd.split('-'))
    
    open_date = pd.Timestamp(year=year, month=open_month, day=open_day)
    close_date = pd.Timestamp(year=year, month=close_month, day=close_day)
    
    # If close_date is earlier than open_date, it spans to the next year
    if close_date < open_date:
        close_date += pd.DateOffset(years=1)
    
    return open_date, close_date

def categorize_season(df, season_info, resort_key):
    """
    Categorize seasons based on operating dates and assign a season_id.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing 'date' column.
    - season_info (dict): Dictionary with 'open' and 'close' dates in 'MM-DD' format.
    - resort_key (str): Key to identify the resort (e.g., 'austrian_alps/st_anton').
    
    Returns:
    - pd.DataFrame: DataFrame with an added 'season_id' column.
    """
    if not season_info:
        # No season information provided
        df['season_id'] = None
        return df
    
    open_mm_dd = season_info['open']
    close_mm_dd = season_info['close']
    
    df = df.copy()
    df['season_id'] = None  # Initialize season identifier
    
    years = df['date'].dt.year.unique()
    
    for year in years:
        open_date, close_date = get_season_dates(year, open_mm_dd, close_mm_dd)
        
        # Filter rows within the current season
        season_mask = (df['date'] >= open_date) & (df['date'] <= close_date)
        season_label = f"{year}-{close_date.year}"
        
        df.loc[season_mask, 'season_id'] = season_label
    
    return df

def add_operating_season_indicator(df):
    """
    Adds a boolean column 'is_operating_season' indicating if the row is within an operating season.
    """
    df = df.copy()
    df['is_operating_season'] = df['season_id'].notnull()
    return df