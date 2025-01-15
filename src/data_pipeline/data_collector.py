import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.data_pipeline.aws_connector import AWSConnector

class SafetyDataCollector:
    def __init__(self, aws_connector):
        self.aws = aws_connector
        self.locations = [
            {'name': 'Downtown', 'lat': 34.0522, 'lon': -118.2437},
            {'name': 'Suburb', 'lat': 34.1478, 'lon': -118.1445}
        ]

    def generate_sample_fire_data(self, days=30):
        """Generate sample fire incident data"""
        data = []
        start_date = datetime.now() - timedelta(days=days)
        
        for _ in range(100):  # Generate 100 incidents
            location = random.choice(self.locations)
            incident_date = start_date + timedelta(
                days=random.randint(0, days),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            data.append({
                'timestamp': incident_date.strftime('%Y-%m-%d %H:%M:%S'),
                'location': location['name'],
                'latitude': location['lat'],
                'longitude': location['lon'],
                'incident_type': random.choice(['structural', 'vegetation', 'vehicle']),
                'severity': random.randint(1, 5),
                'response_time_minutes': random.randint(5, 30)
            })
        
        df = pd.DataFrame(data)
        self.aws.upload_csv(df, 'fire', 'sample_fire_incidents')
        return df

    def generate_sample_crime_data(self, days=30):
        """Generate sample crime data"""
        data = []
        start_date = datetime.now() - timedelta(days=days)
        
        for _ in range(150):  # Generate 150 incidents
            location = random.choice(self.locations)
            incident_date = start_date + timedelta(
                days=random.randint(0, days),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            data.append({
                'timestamp': incident_date.strftime('%Y-%m-%d %H:%M:%S'),
                'location': location['name'],
                'latitude': location['lat'],
                'longitude': location['lon'],
                'crime_type': random.choice(['theft', 'vandalism', 'assault']),
                'status': random.choice(['reported', 'under_investigation', 'resolved']),
                'priority_level': random.randint(1, 3)
            })
        
        df = pd.DataFrame(data)
        self.aws.upload_csv(df, 'crime', 'sample_crime_incidents')
        return df

    def generate_sample_weather_data(self, days=30):
        """Generate sample weather data"""
        data = []
        start_date = datetime.now() - timedelta(days=days)
        
        for location in self.locations:
            current_date = start_date
            while current_date <= datetime.now():
                # Generate 24 readings per day
                for hour in range(24):
                    data.append({
                        'timestamp': current_date.replace(hour=hour).strftime('%Y-%m-%d %H:%M:%S'),
                        'location': location['name'],
                        'latitude': location['lat'],
                        'longitude': location['lon'],
                        'temperature': random.uniform(60, 90),
                        'humidity': random.uniform(30, 70),
                        'wind_speed': random.uniform(0, 20),
                        'precipitation': random.uniform(0, 0.5)
                    })
                current_date += timedelta(days=1)
        
        df = pd.DataFrame(data)
        self.aws.upload_csv(df, 'weather', 'sample_weather_data')
        return df

def main():
    # Initialize AWS connector
    aws = AWSConnector()
    
    # Initialize data collector
    collector = SafetyDataCollector(aws)
    
    # Generate and upload sample data
    print("Generating fire incident data...")
    fire_data = collector.generate_sample_fire_data()
    print(f"Generated {len(fire_data)} fire incidents")
    
    print("\nGenerating crime data...")
    crime_data = collector.generate_sample_crime_data()
    print(f"Generated {len(crime_data)} crime incidents")
    
    print("\nGenerating weather data...")
    weather_data = collector.generate_sample_weather_data()
    print(f"Generated {len(weather_data)} weather records")
    
    print("\nData generation and upload complete!")

if __name__ == "__main__":
    main()