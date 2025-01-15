import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.data_pipeline.aws_connector import AWSConnector

class SafetyDataPreprocessor:
    def __init__(self, aws_connector):
        self.aws = aws_connector
        self.scalers = {
            'numerical': StandardScaler(),
            'temporal': StandardScaler()
        }
        self.label_encoders = {}
        
    def process_timestamps(self, df, timestamp_col='timestamp'):
        """Extract temporal features from timestamp"""
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Extract temporal features
        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['month'] = df[timestamp_col].dt.month
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        
        return df
    
    def process_fire_data(self, fire_df):
        """Process fire incident data"""
        df = self.process_timestamps(fire_df)
        
        # Encode categorical variables
        categorical_cols = ['incident_type', 'location']
        for col in categorical_cols:
            if col in df.columns:
                self.label_encoders[f'fire_{col}'] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[f'fire_{col}'].fit_transform(df[col])
        
        # Scale numerical features
        numerical_cols = ['severity', 'response_time_minutes']
        if numerical_cols:
            df[numerical_cols] = self.scalers['numerical'].fit_transform(df[numerical_cols])
        
        return df
    
    def process_crime_data(self, crime_df):
        """Process crime incident data"""
        df = self.process_timestamps(crime_df)
        
        # Encode categorical variables
        categorical_cols = ['crime_type', 'status', 'location']
        for col in categorical_cols:
            if col in df.columns:
                self.label_encoders[f'crime_{col}'] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[f'crime_{col}'].fit_transform(df[col])
        
        # Scale numerical features
        if 'priority_level' in df.columns:
            df[['priority_level']] = self.scalers['numerical'].fit_transform(df[['priority_level']])
        
        return df
    
    def process_weather_data(self, weather_df):
        """Process weather data"""
        df = self.process_timestamps(weather_df)
        
        # Scale weather features
        weather_features = ['temperature', 'humidity', 'wind_speed', 'precipitation']
        if all(col in df.columns for col in weather_features):
            df[weather_features] = self.scalers['numerical'].fit_transform(df[weather_features])
        
        return df
    
    def merge_datasets(self, fire_df, crime_df, weather_df):
        """Merge processed datasets based on location and time"""
        # Ensure all timestamps are datetime
        for df in [fire_df, crime_df, weather_df]:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Merge fire and weather data
        merged = pd.merge_asof(
            fire_df.sort_values('timestamp'),
            weather_df.sort_values('timestamp'),
            on='timestamp',
            by='location',
            direction='nearest',
            tolerance=pd.Timedelta('1H')
        )
        
        # Merge with crime data
        merged = pd.merge_asof(
            merged.sort_values('timestamp'),
            crime_df.sort_values('timestamp'),
            on='timestamp',
            by='location',
            direction='nearest',
            tolerance=pd.Timedelta('1H')
        )
        
        # Save merged dataset to processed bucket
        self.aws.upload_csv(
            merged, 
            'processed', 
            f'merged_safety_data_{datetime.now().strftime("%Y%m%d")}'
        )
        
        return merged
    
    def prepare_features(self, merged_df):
        """Prepare final feature matrix for ML model"""
        # Select relevant features
        feature_cols = [
            # Temporal features
            'hour', 'day_of_week', 'is_weekend', 'is_night',
            
            # Fire features
            'incident_type_encoded', 'severity',
            
            # Crime features
            'crime_type_encoded', 'priority_level',
            
            # Weather features
            'temperature', 'humidity', 'wind_speed', 'precipitation'
        ]
        
        # Ensure all feature columns exist
        available_features = [col for col in feature_cols if col in merged_df.columns]
        
        X = merged_df[available_features]
        
        # Handle any missing values
        X = X.fillna(X.mean())
        
        return X

def main():
    # Initialize AWS connector and preprocessor
    aws = AWSConnector()
    preprocessor = SafetyDataPreprocessor(aws)
    
    # Download data from S3
    print("Downloading data from S3...")
    fire_data = aws.download_csv('fire', 'sample_fire_incidents')
    crime_data = aws.download_csv('crime', 'sample_crime_incidents')
    weather_data = aws.download_csv('weather', 'sample_weather_data')
    
    # Process each dataset
    print("\nProcessing datasets...")
    processed_fire = preprocessor.process_fire_data(fire_data)
    processed_crime = preprocessor.process_crime_data(crime_data)
    processed_weather = preprocessor.process_weather_data(weather_data)
    
    # Merge datasets
    print("Merging datasets...")
    merged_data = preprocessor.merge_datasets(
        processed_fire,
        processed_crime,
        processed_weather
    )
    
    # Prepare features for ML
    print("Preparing feature matrix...")
    X = preprocessor.prepare_features(merged_data)
    
    print(f"\nFinal feature matrix shape: {X.shape}")
    print("\nFeature columns:")
    print(X.columns.tolist())
    
    print("\nPreprocessing complete!")

if __name__ == "__main__":
    main()