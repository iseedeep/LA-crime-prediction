import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data_pipeline.aws_connector import AWSConnector

# Define crime categories mapping
CRIME_CATEGORY_MAPPING = {
    # Violent Crimes
    'CRIMINAL HOMICIDE': 'VIOLENT_CRIME',
    'MANSLAUGHTER': 'VIOLENT_CRIME',
    'RAPE': 'VIOLENT_CRIME',
    'ASSAULT WITH DEADLY WEAPON': 'VIOLENT_CRIME',
    'INTIMATE PARTNER': 'VIOLENT_CRIME',
    'KIDNAPPING': 'VIOLENT_CRIME',
    'ROBBERY': 'VIOLENT_CRIME',
    'SHOTS FIRED': 'VIOLENT_CRIME',
    'BATTERY': 'VIOLENT_CRIME',
    
    # Property Crimes
    'BURGLARY': 'PROPERTY_CRIME',
    'VEHICLE - STOLEN': 'PROPERTY_CRIME',
    'BIKE - STOLEN': 'PROPERTY_CRIME',
    'THEFT': 'PROPERTY_CRIME',
    'VANDALISM': 'PROPERTY_CRIME',
    'STOLEN': 'PROPERTY_CRIME',
    'SHOPLIFTING': 'PROPERTY_CRIME',
    
    # Sexual Crimes
    'SEXUAL': 'SEXUAL_CRIME',
    'LEWD': 'SEXUAL_CRIME',
    'CHILD PORNOGRAPHY': 'SEXUAL_CRIME',
    'CRM AGNST CHLD': 'SEXUAL_CRIME',
    
    # Domestic Violence
    'VIOLATION OF RESTRAINING ORDER': 'DOMESTIC_VIOLENCE',
    'VIOLATION OF COURT ORDER': 'DOMESTIC_VIOLENCE',
    'CHILD ABUSE': 'DOMESTIC_VIOLENCE',
    'CHILD NEGLECT': 'DOMESTIC_VIOLENCE',
    
    # Weapons
    'WEAPON': 'WEAPONS_OFFENSE',
    'FIREARM': 'WEAPONS_OFFENSE',
    'BOMB': 'WEAPONS_OFFENSE',
    'BRANDISH': 'WEAPONS_OFFENSE',
    
    # Fraud
    'FRAUD': 'FRAUD',
    'COUNTERFEIT': 'FRAUD',
    'FORGERY': 'FRAUD',
    'IDENTITY': 'FRAUD',
    'BUNCO': 'FRAUD',
    'EMBEZZLEMENT': 'FRAUD',
    
    # Drugs
    'DRUGS': 'DRUG_OFFENSE',
    
    # Public Disorder
    'DISTURBING THE PEACE': 'PUBLIC_DISORDER',
    'TRESPASSING': 'PUBLIC_DISORDER',
    'DRINKING IN PUBLIC': 'PUBLIC_DISORDER',
    'DISRUPT': 'PUBLIC_DISORDER',
    'PROWLER': 'PUBLIC_DISORDER'
}

def preprocess_crime_data(bucket_name, s3_key, local_path):
    """
    Preprocesses the crime data from the given S3 bucket and key.
    """
    try:
        # Initialize AWS connector and download data
        aws = AWSConnector()
        aws.download_csv(bucket_name, s3_key, local_path)

        # Load the crime dataset
        crime_df = pd.read_csv(local_path)
        print("\nInitial columns:", crime_df.columns.tolist())
        print("\nSample of first few rows:")
        print(crime_df.head())

        # Strip any leading/trailing spaces from column names
        crime_df.columns = crime_df.columns.str.strip()

        # Drop unnecessary columns if they exist
        columns_to_drop = ["DR_NO", "Date Rptd", "LAT", "LON", "Crm Cd 1", "Crm Cd 2", "Crm Cd 3", "Crm Cd 4"]
        columns_to_drop = [col for col in columns_to_drop if col in crime_df.columns]
        crime_df.drop(columns=columns_to_drop, inplace=True)

        # Handle datetime features
        crime_df["Date Occurred"] = pd.to_datetime(crime_df["DATE OCC"], format='%m/%d/%Y %I:%M:%S %p')
        crime_df["Hour of Day"] = crime_df["Date Occurred"].dt.hour
        crime_df["Day of Week"] = crime_df["Date Occurred"].dt.dayofweek
        crime_df["Month"] = crime_df["Date Occurred"].dt.month
        crime_df["Year"] = crime_df["Date Occurred"].dt.year
        crime_df["Is Weekend"] = (crime_df["Day of Week"] >= 5).astype(int)
        crime_df["Is Night"] = ((crime_df["Hour of Day"] >= 22) | (crime_df["Hour of Day"] <= 5)).astype(int)

        # Drop original date columns
        crime_df.drop(columns=["DATE OCC", "Date Occurred"], inplace=True)

        # Handle location data
        if "LOCATION" in crime_df.columns:
            crime_df["Location_Hash"] = pd.util.hash_pandas_object(crime_df["LOCATION"])
            crime_df.drop(columns=["LOCATION"], inplace=True)

        # Map crimes to broader categories
        crime_df['Crime_Category'] = crime_df['Crm Cd Desc'].apply(
            lambda x: next((cat for key, cat in CRIME_CATEGORY_MAPPING.items() 
                          if key in x.upper()), 'OTHER')
        )

        # Create target variable
        le = LabelEncoder()
        crime_df['Target'] = le.fit_transform(crime_df['Crime_Category'])
        
        # Store and print target mapping
        target_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print("\nTarget variable mapping (Consolidated Categories):")
        for k, v in target_mapping.items():
            print(f"{k}: {v}")
        
        # Print category distribution
        print("\nCrime Category Distribution:")
        print(crime_df['Crime_Category'].value_counts())

        # Drop original crime description and category columns
        crime_df.drop(columns=["Crm Cd Desc", "Crime_Category"], inplace=True)

        # Identify categorical columns (object dtype)
        categorical_columns = crime_df.select_dtypes(include=['object']).columns
        print("\nCategorical columns found:", categorical_columns.tolist())

        # Encode all categorical columns
        for col in categorical_columns:
            if col != 'Target':  # Skip target variable
                le = LabelEncoder()
                # Handle missing values before encoding
                crime_df[col] = crime_df[col].fillna('MISSING')
                crime_df[col] = le.fit_transform(crime_df[col])
                print(f"\nEncoded {col} with {len(le.classes_)} unique values")

        # Handle missing values in numeric columns
        numeric_columns = crime_df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_columns) > 0:
            imputer = SimpleImputer(strategy='mean')
            crime_df[numeric_columns] = imputer.fit_transform(crime_df[numeric_columns])

        # Print final dataset info
        print("\nFinal dataset shape:", crime_df.shape)
        print("\nFinal columns:", crime_df.columns.tolist())
        
        # Check for any remaining missing values
        missing_values = crime_df.isnull().sum()
        print("\nMissing values in final dataset:\n", missing_values[missing_values > 0])

        # Upload processed data back to S3
        aws.upload_csv(crime_df, bucket_name, 'processed/processed_crime_data.csv')

        return crime_df

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    bucket_name = 'iseedeepproject1'
    s3_key = 'Crime_Data_from_2020_to_Present.csv.csv'
    local_path = 'data.csv'
    preprocessed_crime_df = preprocess_crime_data(bucket_name, s3_key, local_path)