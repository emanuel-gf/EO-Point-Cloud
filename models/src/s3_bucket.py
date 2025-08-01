from loguru import logger
import io 
import numpy as np
from PIL import Image
import os
import boto3
import requests
from urllib.parse import parse_qs, urlparse
from lxml import html
from datetime import datetime, timedelta
from pystac_client import Client

def get_product(s3_resource, bucket_name, object_url, output_path):
    """
    Download a product from S3 bucket and create output directory if it doesn't exist.

    Args:
        s3_resource: boto3 S3 resource object
        bucket_name (str): Name of the S3 bucket
        object_url (str): Path to the object within the bucket
        output_path (str): Local directory to save the file

    Returns:
        str: Path to the downloaded file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Extract filename from the object URL
    _, filename = os.path.split(object_url)

    # Full path where the file will be saved
    local_file_path = os.path.join(output_path, filename)

    print(f"Downloading {object_url} to {local_file_path}...")

    try:
        # Download the file from S3
        s3_resource.Bucket(bucket_name).download_file(object_url, local_file_path)
        print(f"Successfully downloaded to {local_file_path}")
    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        raise

    return local_file_path

def get_product_content(s3_client, bucket_name: str, object_url: str) -> bytes:
    """
    Download the content of a product from an S3 bucket. Image as bytes in memory. 

    Args:
        s3_client: A boto3 S3 client object.
        bucket_name (str): The name of the S3 bucket.
        object_url (str): The path to the object within the bucket.

    Returns:
        bytes: The content of the downloaded file.

    Raises:
        Exception: If an error occurs during the download process.
    """
    logger.info(f"Downloading {object_url}")

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=object_url)
        content = response['Body'].read()
        logger.success(f"Successfully downloaded {object_url}")
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise

    return content


class S3Connector:
    """A clean connector for S3-compatible storage services."""

    def __init__(
        self,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
        region_name: str = 'default'
    ) -> None:
        """Initialize the S3Connector with connection parameters."""
        self.endpoint_url = endpoint_url
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.region_name = region_name

        # Create session
        self.session = boto3.session.Session()

        # Initialize S3 resource and client
        self.s3 = self._create_s3_resource()
        self.s3_client = self._create_s3_client()

    def _create_s3_resource(self):
        return self.session.resource(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region_name
        )

    def _create_s3_client(self):
        return self.session.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region_name
        )

    def get_s3_client(self):
        """Get the boto3 S3 client."""
        return self.s3_client

    def get_s3_resource(self):
        """Get the boto3 S3 resource."""
        return self.s3

    def get_bucket(self, bucket_name: str):
        """Get a specific bucket by name."""
        return self.s3.Bucket(bucket_name)

    def list_buckets(self) -> list:
        """List all available buckets."""
        response = self.s3_client.list_buckets()
        return [bucket['Name'] for bucket in response.get('Buckets', [])]


def connect_to_s3(endpoint_url: str, access_key_id: str, secret_access_key: str) -> tuple:
    """Connect to S3 storage."""
    try:
        connector = S3Connector(
            endpoint_url=endpoint_url,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            region_name='default'
        )
        logger.success(f"Successfully connected to {endpoint_url} ")
        return connector.get_s3_resource(), connector.get_s3_client()
    except Exception as e:
        logger.error(f"Failed to connect to S3 storage: {e}")
        return None, None
    

def extract_s3_path_from_url(url: str) -> str:
    """
    Extract the S3 object path from an S3 URL or URI.

    This function parses S3 URLs/URIs and returns just the object path portion,
    removing the protocol (s3://), bucket name, and any leading slashes.

    Args:
        url (str): The full S3 URI (e.g., 's3://eodata/path/to/file.jp2')

    Returns:
        str: The S3 object path (without protocol, bucket name and leading slashes)

    Raises:
        ValueError: If the provided URL is not an S3 URL.
    """
    if not url.startswith('s3://'):
        return url

    parsed_url = urlparse(url)

    if parsed_url.scheme != 's3':
        raise ValueError(f"URL {url} is not an S3 URL")

    object_path = parsed_url.path.lstrip('/')
    return object_path


 