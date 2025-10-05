"""
Weather Data Collection Module

This module provides functionality to collect historical weather data
from Visual Crossing Weather API for Hanoi temperature forecasting.
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import time
import logging
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherCollector:
    """
    A class to collect weather data from Visual Crossing Weather API.
    
    This class handles:
    - API authentication and rate limiting
    - Data collection for specified date ranges
    - Both daily and hourly data collection
    - Data validation and error handling
    """
    
    def __init__(self, api_key: Optional[str] = None, config_path: str = "config/config.yaml"):
        """
        Initialize the Weather Collector.
        
        Args:
            api_key: Visual Crossing API key. If None, will try to get from environment.
            config_path: Path to configuration file.
        """
        self.api_key = api_key or os.getenv("VISUAL_CROSSING_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set VISUAL_CROSSING_API_KEY environment variable.")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.base_url = self.config['data']['visual_crossing']['base_url']
        self.location = self.config['data']['location']
        self.rate_limit = self.config['data']['visual_crossing']['rate_limit']
        self.timeout = self.config['data']['visual_crossing']['timeout']
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 86400 / self.rate_limit  # seconds between requests
    
    def _wait_for_rate_limit(self):
        """Implement rate limiting to avoid API limits."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def fetch_daily_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch daily weather data for a date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with daily weather data
        """
        self._wait_for_rate_limit()
        
        url = f"{self.base_url}/{self.location}/{start_date}/{end_date}"
        
        params = {
            'unitGroup': 'metric',
            'include': 'days',
            'key': self.api_key,
            'contentType': 'json'
        }
        
        try:
            logger.info(f"Fetching daily data from {start_date} to {end_date}")
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            if 'days' not in data:
                logger.warning(f"No daily data found for {start_date} to {end_date}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data['days'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['location'] = self.location
            
            logger.info(f"Successfully fetched {len(df)} days of data")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error processing daily data: {e}")
            return pd.DataFrame()
    
    def fetch_hourly_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch hourly weather data for a date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with hourly weather data
        """
        self._wait_for_rate_limit()
        
        url = f"{self.base_url}/{self.location}/{start_date}/{end_date}"
        
        params = {
            'unitGroup': 'metric',
            'include': 'hours',
            'key': self.api_key,
            'contentType': 'json'
        }
        
        try:
            logger.info(f"Fetching hourly data from {start_date} to {end_date}")
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            if 'days' not in data:
                logger.warning(f"No hourly data found for {start_date} to {end_date}")
                return pd.DataFrame()
            
            # Extract hourly data from all days
            all_hours = []
            for day in data['days']:
                if 'hours' in day:
                    for hour in day['hours']:
                        hour['date'] = day['datetime']
                        hour['datetime'] = f"{day['datetime']} {hour['datetime']}"
                        all_hours.append(hour)
            
            if not all_hours:
                logger.warning(f"No hourly data found for {start_date} to {end_date}")
                return pd.DataFrame()
            
            df = pd.DataFrame(all_hours)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['location'] = self.location
            
            logger.info(f"Successfully fetched {len(df)} hours of data")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error processing hourly data: {e}")
            return pd.DataFrame()
    
    def collect_historical_data(self, years: int = 10, data_type: str = "daily") -> pd.DataFrame:
        """
        Collect historical weather data for specified number of years.
        
        Args:
            years: Number of years of historical data to collect
            data_type: Type of data to collect ("daily" or "hourly")
            
        Returns:
            DataFrame with historical weather data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        # Collect data in smaller chunks to avoid API limits
        chunk_size = 365 if data_type == "daily" else 30  # 1 year for daily, 1 month for hourly
        all_data = []
        
        current_date = start_date
        while current_date < end_date:
            chunk_end = min(current_date + timedelta(days=chunk_size), end_date)
            
            start_str = current_date.strftime("%Y-%m-%d")
            end_str = chunk_end.strftime("%Y-%m-%d")
            
            if data_type == "daily":
                chunk_data = self.fetch_daily_data(start_str, end_str)
            else:
                chunk_data = self.fetch_hourly_data(start_str, end_str)
            
            if not chunk_data.empty:
                all_data.append(chunk_data)
            
            current_date = chunk_end + timedelta(days=1)
        
        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            df = df.sort_values('datetime').reset_index(drop=True)
            logger.info(f"Collected {len(df)} records of {data_type} data over {years} years")
            return df
        else:
            logger.warning("No data collected")
            return pd.DataFrame()
    
    def save_data(self, df: pd.DataFrame, data_type: str = "daily", filename: Optional[str] = None):
        """
        Save collected data to file.
        
        Args:
            df: DataFrame to save
            data_type: Type of data ("daily" or "hourly")
            filename: Optional filename. If None, will generate automatically.
        """
        if df.empty:
            logger.warning("No data to save")
            return
        
        # Create directory if it doesn't exist
        if data_type == "daily":
            save_dir = self.config['data']['paths']['raw_daily']
        else:
            save_dir = self.config['data']['paths']['raw_hourly']
        
        os.makedirs(save_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hanoi_weather_{data_type}_{timestamp}.csv"
        
        filepath = os.path.join(save_dir, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
    
    def get_weather_features_info(self) -> Dict[str, str]:
        """
        Get information about available weather features.
        
        Returns:
            Dictionary with feature names and their descriptions
        """
        feature_descriptions = {
            # Temperature features
            'temp': 'Average temperature (°C)',
            'tempmax': 'Maximum temperature (°C)',
            'tempmin': 'Minimum temperature (°C)',
            'feelslike': 'Feels like temperature (°C)',
            
            # Atmospheric features
            'humidity': 'Relative humidity (%)',
            'pressure': 'Atmospheric pressure (hPa)',
            'visibility': 'Visibility (km)',
            'cloudcover': 'Cloud cover (%)',
            
            # Wind features
            'windspeed': 'Wind speed (km/h)',
            'winddir': 'Wind direction (degrees)',
            'windgust': 'Wind gust speed (km/h)',
            
            # Precipitation features
            'precip': 'Precipitation amount (mm)',
            'precipprob': 'Precipitation probability (%)',
            'preciptype': 'Precipitation type (rain, snow, etc.)',
            'snow': 'Snow amount (cm)',
            'snowdepth': 'Snow depth (cm)',
            
            # Solar and UV features
            'solarradiation': 'Solar radiation (W/m²)',
            'solarenergy': 'Solar energy (MJ/m²)',
            'uvindex': 'UV index',
            
            # Celestial features
            'moonphase': 'Moon phase (0-1, 0=new moon, 0.5=full moon)',
            'sunrise': 'Sunrise time',
            'sunset': 'Sunset time',
            
            # Weather condition features
            'conditions': 'Weather conditions description',
            'description': 'Detailed weather description',
            'icon': 'Weather icon identifier',
            
            # Additional features (for hourly data)
            'dew': 'Dew point temperature (°C)',
            'precipcover': 'Precipitation coverage (%)',
            'severerisk': 'Severe weather risk level',
        }
        
        return feature_descriptions


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect Hanoi weather data")
    parser.add_argument("--data-type", choices=["daily", "hourly"], default="daily",
                       help="Type of data to collect")
    parser.add_argument("--years", type=int, default=10,
                       help="Number of years of historical data to collect")
    parser.add_argument("--config", default="config/config.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    try:
        collector = WeatherCollector(config_path=args.config)
        
        logger.info(f"Starting {args.data_type} data collection for {args.years} years")
        df = collector.collect_historical_data(years=args.years, data_type=args.data_type)
        
        if not df.empty:
            collector.save_data(df, data_type=args.data_type)
            logger.info("Data collection completed successfully")
        else:
            logger.error("Data collection failed")
    
    except Exception as e:
        logger.error(f"Data collection failed with error: {e}")


if __name__ == "__main__":
    main()