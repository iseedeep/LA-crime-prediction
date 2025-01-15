import pandas as pd
import requests
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any
import json

class DataPipeline:
    def __init__(self, aws_connector):
        """Initialize pipeline with AWS connector"""
        self.aws = aws_connector
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def collect_fire_data(self, api_url: str, api_key: str = None) -> pd.DataFrame:
        """Collect fire incident data from API"""
        headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
        
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data)
            self.aws.upload_csv(df, 'fire', f'fire_incidents_{datetime.now().strftime("%Y%m%d")}')
            return df
            
        except Exception as e:
            self.logger.error(f"Error collecting fire data: {str(e)}")
            return None
    
    def collect_weather_data(self, api_key: str, locations: List[Dict[str, float]]) -> pd.DataFrame:
        """Collect weather data for specified locations"""
        weather_data = []
        
        for location in locations:
            try:
                url = f"https://api.openweathermap.org/data/2.5/weather"
                params = {
                    'lat': location['lat'],
                    'lon': location['lon'],
                    'appid': api_key,
                    'units': 'metric'
                }
                
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                weather_data.append({
                    'location': f"{location['lat']},{location['lon']}",
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'wind_speed': data['wind']['speed'],
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                
            except Exception as e:
                self.logger.error(f"Error collecting weather data for location {location}: {str(e)}")
        
        if weather_data:
            df = pd.DataFrame(weather_data)
            self.aws.upload_csv(df, 'weather', f'weather_data_{datetime.now().strftime("%Y%m%d")}')
            return df
        return None
    
    def collect_crime_data(self, api_url: str, api_key: str = None) -> pd.DataFrame:
        """Collect crime data from API"""
        headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
        
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data)
            self.aws.upload_csv(df, 'crime', f'crime_data_{datetime.now().strftime("%Y%m%d")}')
            return df
            
        except Exception as e:
            self.logger.error(f"Error collecting crime data: {str(e)}")
            return None
    
    def preprocess_data(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Preprocess collected data"""
        try:
            # Common preprocessing steps
            df = df.copy()
            df.dropna(inplace=True)
            
            if data_type == 'fire':
                # Fire-specific preprocessing
                df['datetime'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['datetime'].dt.hour
                df['day_of_week'] = df['datetime'].dt.dayofweek
                
            elif data_type == 'crime':
                # Crime-specific preprocessing
                df['datetime'] = pd.to_datetime(df['timestamp'])
                df['is_weekend'] = df['datetime'].dt.dayofweek.isin([5, 6])
                
            elif data_type == 'weather':
                # Weather-specific preprocessing
                df['temperature_fahrenheit'] = df['temperature'] * 9/5 + 32
                
            # Save processed data
            self.aws.upload_csv(
                df, 
                'processed', 
                f'processed_{data_type}_{datetime.now().strftime("%Y%m%d")}'
            )
            return df
            
        except Exception as e:
            self.logger.error(f"Error preprocessing {data_type} data: {str(e)}")
            return None
    
    def merge_datasets(self, 
                      fire_data: pd.DataFrame, 
                      crime_data: pd.DataFrame, 
                      weather_data: pd.DataFrame) -> pd.DataFrame:
        """Merge different datasets based on location and time"""
        try:
            # Convert all timestamps to datetime
            fire_data['datetime'] = pd.to_datetime(fire_data['timestamp'])
            crime_data['datetime'] = pd.to_datetime(crime_data['timestamp'])
            weather_data['datetime'] = pd.to_datetime(weather_data['timestamp'])
            
            # Merge datasets
            merged = pd.merge_asof(
                fire_data.sort_values('datetime'),
                weather_data.sort_values('datetime'),
                on='datetime',
                by='location',
                direction='nearest'
            )
            
            merged = pd.merge_asof(
                merged.sort_values('datetime'),
                crime_data.sort_values('datetime'),
                on='datetime',
                by='location',
                direction='nearest'
            )
            
            # Save merged dataset
            self.aws.upload_csv(
                merged, 
                'processed', 
                f'merged_data_{datetime.now().strftime("%Y%m%d")}'
            )
            return merged
            
        except Exception as e:
            self.logger.error(f"Error merging datasets: {str(e)}")
            return None
    
    def run_pipeline(self, 
                    fire_api_url: str, 
                    crime_api_url: str,
                    weather_api_key: str,
                    locations: List[Dict[str, float]],
                    fire_api_key: str = None,
                    crime_api_key: str = None) -> pd.DataFrame:
        """Run the complete data pipeline"""
        self.logger.info("Starting data pipeline...")
        
        # Collect data
        fire_data = self.collect_fire_data(fire_api_url, fire_api_key)
        crime_data = self.collect_crime_data(crime_api_url, crime_api_key)
        weather_data = self.collect_weather_data(weather_api_key, locations)
        
        if not all([fire_data is not None, 
                   crime_data is not None, 
                   weather_data is not None]):
            self.logger.error("Failed to collect all required data")
            return None
        
        # Preprocess each dataset
        fire_processed = self.preprocess_data(fire_data, 'fire')
        crime_processed = self.preprocess_data(crime_data, 'crime')
        weather_processed = self.preprocess_data(weather_data, 'weather')
        
        if not all([fire_processed is not None, 
                   crime_processed is not None, 
                   weather_processed is not None]):
            self.logger.error("Failed to preprocess all datasets")
            return None
        
        # Merge datasets
        merged_data = self.merge_datasets(
            fire_processed,
            crime_processed,
            weather_processed
        )
        
        self.logger.info("Data pipeline completed successfully")
        return merged_data

# Example usage
if __name__ == "__main__":
    from aws_connector import AWSConnector
    
    # Initialize AWS connector
    aws = AWSConnector()
    
    # Initialize pipeline
    pipeline = DataPipeline(aws)
    
    # Example locations
    locations = [
        {'lat': 34.0522, 'lon': -118.2437},  # Los Angeles
        {'lat': 34.1478, 'lon': -118.1445},  # Pasadena
    ]
    
    # Run pipeline
    merged_data = pipeline.run_pipeline(
        fire_api_url="https://api.example.com/fire",
        crime_api_url="https://api.example.com/crime",
        weather_api_key="your_weather_api_key",
        locations=locations
    )