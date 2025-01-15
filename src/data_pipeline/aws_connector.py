import boto3
from io import StringIO

class AWSConnector:
    def __init__(self, aws_access_key_id=None, aws_secret_access_key=None, region_name='us-east-1'):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )

    def create_bucket(self, bucket_name):
        try:
            self.s3.create_bucket(Bucket=bucket_name)
            print(f"Bucket '{bucket_name}' created successfully.")
        except self.s3.exceptions.BucketAlreadyExists as e:
            print(f"Bucket '{bucket_name}' already exists. Please choose a different name.")
        except self.s3.exceptions.BucketAlreadyOwnedByYou as e:
            print(f"Bucket '{bucket_name}' already owned by you.")

    def upload_csv(self, dataframe, bucket_name, s3_key):
        csv_buffer = StringIO()
        dataframe.to_csv(csv_buffer, index=False)
        self.s3.put_object(Bucket=bucket_name, Key=s3_key, Body=csv_buffer.getvalue())
        print(f"CSV uploaded to '{bucket_name}/{s3_key}' successfully.")

    def download_csv(self, bucket_name, s3_key, local_path):
        with open(local_path, 'wb') as f:
            self.s3.download_fileobj(bucket_name, s3_key, f)
        print(f"CSV downloaded from '{bucket_name}/{s3_key}' to '{local_path}' successfully.")

# Example usage
if __name__ == "__main__":
    aws = AWSConnector()
    aws.create_bucket('iseedeepproject1')