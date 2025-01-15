import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from aws_connector import AWSConnector

def explore_data(df):
    """
    Explores the crime data with visualizations.

    Args:
      df: A pandas DataFrame with the preprocessed crime data.
    """
    # Visualizations
    plt.figure(figsize=(10, 6))
    top_crime_types = df['Crm Cd Desc'].value_counts().nlargest(20).index
    sns.countplot(y='Crm Cd Desc', data=df[df['Crm Cd Desc'].isin(top_crime_types)], order=top_crime_types)
    plt.title('Top 20 Crime Types')
    plt.xlabel('Count')
    plt.ylabel('Crime Type')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(df['Date Occurred'], bins=30, kde=True)
    plt.title('Distribution of Crimes Over Time')
    plt.xlabel('Date Occurred')
    plt.ylabel('Count')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Hour of Day', y='Crm Cd Desc', data=df[df['Crm Cd Desc'].isin(top_crime_types)])
    plt.title('Crime Occurrence by Hour of Day (Top 20 Crime Types)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Crime Type')
    plt.show()

if __name__ == "__main__":
    # Initialize AWS connector
    aws = AWSConnector()

    # Download the preprocessed data from S3
    bucket_name = 'iseedeepproject1'
    s3_key = 'Crime_Data_from_2020_to_Present.csv.csv'  # Ensure this matches the path in your S3 bucket
    local_path = 'processed_crime_data.csv'
    aws.download_csv(bucket_name, s3_key, local_path)

    # Load the preprocessed data
    preprocessed_crime_df = pd.read_csv(local_path)

    # Explore the data
    explore_data(preprocessed_crime_df)