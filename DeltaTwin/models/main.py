
import os
import sys
import time
from loguru import logger
from typing import Dict, List, Union
from dotenv import load_dotenv
import pystac_client
import psutil
import json
import cv2 as cv
from plyfile import PlyData, PlyElement
import numpy as np
import pandas as pd
#import pdal
import rasterio
import xarray as xr
from pyproj import Transformer
import subprocess
from loguru import logger
import yaml  
from urllib.parse import urlparse
import boto3
import pystac_client
from dotenv import load_dotenv
from PIL import Image
import requests
from lxml import html
from datetime import datetime, timedelta, date
from pystac_client import Client
from typing import Optional


## One page model  -----------------------------------------------------------------
def remove_last_segment_rsplit(sentinel_id: str) -> str:
    """
    Remove the last segment from a Sentinel ID by splitting at the last underscore.

    Args:
        sentinel_id (str): The Sentinel ID to process.

    Returns:
        str: The Sentinel ID without the last segment.
    """
    parts = sentinel_id.rsplit('_', 1)
    return parts[0]

def data_query_most_recent(catalog, bbox: list, max_cloud_cover: int= 20, default_timedelta: int = 360):
    """
    Fetch both L1C and L2A products from CDSE STAC catalog and find the most recent matching pair.
    By default, searches for images from the last 30 days.

    Args:
        catalog: STAC catalog client
        bbox: Bounding box coordinates [west, south, east, north]
        max_cloud_cover: Maximum cloud cover percentage
        start_date: Start date in format "YYYY-MM-DD" (optional, defaults to 30 days ago)
        end_date: End date in format "YYYY-MM-DD" (optional, defaults to today)

    Returns:
        tuple: (most recent matched L1C item, most recent matched L2A item)
    """
    try:
        # Look up for the most recent date 
        end_date = date.today().strftime("%Y-%m-%d")
        start_date = (date.today() - timedelta(days=default_timedelta)).strftime("%Y-%m-%d")
        
        logger.info(f"Using date range: {start_date} to {end_date}")

        # L2A products
        logger.info(f"Searching for L2A products from {start_date} to {end_date} in bbox {bbox}")
        l2a_items = catalog.search(
            collections=['sentinel-2-l2a'],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
            max_items=1000,
            sortby=["-datetime"] 
        ).item_collection()

        # Filter L2A items - remove those with high nodata percentage
        l2a_items = [item for item in l2a_items if item.properties.get("statistics", {}).get('nodata', 100) < 5]

        # Convert to dataframes 
        l2a_dicts = [item.to_dict() for item in l2a_items]

        df_l2a = pd.DataFrame(l2a_dicts)

        logger.info(f"Found {len(l2a_items)} L2A products")

        # Create unique ID
        df_l2a['id_key'] = df_l2a['id'].apply(remove_last_segment_rsplit)

        df_l2a['datetime'] = pd.to_datetime(df_l2a['properties'].apply(lambda x: x.get('datetime')))

        # Sort for most recent 
        df_l2a = df_l2a.sort_values('datetime', ascending=False).drop_duplicates(subset='id_key', keep='first')
        
        l2a_item = next((item for item in l2a_items if item.id == df_l2a.iloc[0]["id"]), None)

        if l2a_item:
            logger.success(f"L2A: {l2a_item.id} ({l2a_item.properties.get('datetime')})")
        else:
            logger.error("Failed to find corresponding STAC items for selected pair")
            return None, None

        return l2a_item

    except Exception as e:
        logger.error(f"Error fetching Sentinel data: {e}")
        return None, None
    
def load_dem_utm(url, bounds, width, height):
    """
    Loads the Copernicus DEM and selects the region of interest.

    Parameters:
        url (str): URL of the DEM dataset.
        bounds (rasterio.coords.BoundingBox): Bounding box with left,
            right, bottom, and top coordinates.
        width (int): Number of pixels (columns) in the target image.
        height (int): Number of pixels (rows) in the target image.

    Returns:
        xarray.DataArray: DEM region of interest.
    """
    # Load the dataset
    dem = xr.open_dataset(
        url,
        chunks={},
        storage_options={"client_kwargs": {"trust_env": True}},
        engine="zarr"
    )

    # Create UTM coordinate grid
    x = np.linspace(bounds.left, bounds.right, width)
    y = np.linspace(bounds.bottom, bounds.top, height)

    # Select the DEM region of interest using nearest interpolation
    dem_roi = dem.sel(x=x, y=y, method="nearest")

    return dem_roi

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

    logger.info(f"Downloading {object_url} to {local_file_path}...")

    try:
        # Download the file from S3
        s3_resource.Bucket(bucket_name).download_file(object_url, local_file_path)
        logger.success(f"Successfully downloaded to {local_file_path}")
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise

    return local_file_path


class Sentinel2Reader:
    """
    A class to read and preprocess Sentinel-2 satellite L2A product.

    This class loads Sentinel-2 L2A product, extracts metadata, and provides
    functionalities for preprocessing (flipping and transposing)
    and visualization.

    Attributes:
        filepath (str): Path to the Sentinel-2 image file.
        data (numpy.ndarray or None): Image after reading and preprocessing.
        bounds (rasterio.coords.BoundingBox or None): Bounding box.
        transform (Affine or None): Affine transformation matrix.
        height (int or None): Number of rows (pixels).
        width (int or None): Number of columns (pixels).
    """

    def __init__(self, filepath, preprocess=True):
        """
        Initializes the Sentinel2Reader instance and loads the image.

        Args:
            filepath (str): Path to the Sentinel-2 image file.
            preprocess (bool, optional): Whether to preprocess the image.
        """
        self.filepath = filepath
        self.data = None
        self.bounds = None
        self.transform = None
        self.height = None
        self.width = None

        self.read_image()
        if preprocess:
            self.preprocess()

    def read_image(self):
        """
        Reads the Sentinel-2 image and extracts metadata.

        This method loads the image using rasterio and extracts useful metadata such as
        image dimensions, bounds, and transformation matrix.
        """
        try:
            with rasterio.open(self.filepath) as src:
                self.data = src.read()
                self.bounds = src.bounds
                self.transform = src.transform
                self.height = src.height
                self.width = src.width
        except Exception as e:
            print(f"Error reading image: {e}")

    def preprocess(self):
        """
        Transposes and flips the image for correct visualization.

        Sentinel-2 images are stored as (Bands, Height, Width), so we transpose them
        to (Height, Width, Bands) for proper display. The image is then flipped
        vertically to match conventional visualization.
        """
        if self.data is not None:
            self.data = np.transpose(self.data, (1, 2, 0))  # Change dimensions
            self.data = cv.flip(self.data, 0)  # Flip vertically
        else:
            print("No image data to preprocess.")


class PcdGenerator:
    def __init__(self, sat_data, dem_data):
        """
        Initialize the PCD Generator.

        Args:
            sat_data (numpy.ndarray): Sentinel-2 RGB data with shape (H, W, 3).
            dem_data (xarray.DataArray): DEM data with coordinates.
        """
        self.sat_data = sat_data
        self.dem_data = dem_data
        self.point_cloud = None
        self.df = None  # Dataframe holding point cloud data

    def _normalize_rgb(self, rgb_array):
        """Normalize RGB values to range [0, 1]."""
        return np.array(rgb_array) / 255.0

    def generate_point_cloud(self):
        """Generate a point cloud from Sentinel-2 and DEM data."""
        # Extract lat/lon and DSM values
        lat_values = self.dem_data.coords['y'].values  # Latitude
        lon_values = self.dem_data.coords['x'].values  # Longitude
        dsm_values = self.dem_data.values  # Elevation

        # Reshape Sentinel-2 RGB data to (N, 3)
        tci_rgb = self.sat_data.reshape(-1, 3)
        rgb_tuples = [tuple(rgb) for rgb in tci_rgb]

        # Create a meshgrid for coordinates
        lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)

        # Flatten everything to 1D
        lon_flat = lon_grid.flatten()
        lat_flat = lat_grid.flatten()
        dsm_flat = dsm_values.flatten()

        # Create a DataFrame to store point cloud data
        self.df = pd.DataFrame({
            'x': lon_flat,
            'y': lat_flat,
            'z': dsm_flat,
            'color': rgb_tuples
        })
        logger.info(f"Total points before sampling: {len(self.df)}")

        # # Apply sampling
        # sample_size = int(self.sample_fraction * len(df) / 100)
        # self.df = df[:sample_size]
        # print(f"Sampled points: {len(self.df)}")

    def downsample(self, sample_fraction=0.1, random_state=42):
        """
        Downsample the point cloud by randomly sampling a fraction of the points.

        Args:
            sample_fraction (float): The fraction of points to sample. Default is 0.1.
            random_state (int): Seed for the random number generator. Default is 42.
        """

        # Perform random sampling
        self.df = self.df.sample(frac=sample_fraction, random_state=random_state, replace=True)
        logger.info(f"Point cloud downsampled with sample fraction {sample_fraction}")
        logger.info(f"Total point after downsampling: {len(self.df)}")


class PointCloudHandler:
    """Handler for point cloud data operations"""
    
    def __init__(self, pcd_df):
        """
        Initialize with point cloud dataframe
        
        Args:
            pcd_df: pandas DataFrame with columns like 'x', 'y', 'z' and optionally colors
        """
        self.df = pcd_df

        
    def to_ply_with_plyfile(self, filename):
        """
        Save point cloud data to PLY file using plyfile library.
        Replaces Open3D's save_point_cloud functionality.
        
        Args:
            filename: output PLY filename
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate input data
            if self.df is None or len(self.df) == 0:
                logger.error("No point cloud data to save")
                return False
            
            # Check for required coordinate columns
            required_cols = ['x', 'y', 'z']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            # Extract points
            points = self.df[['x', 'y', 'z']].values.astype(np.float32)
            logger.info(f"Extracting {len(points):,} points")
            
            # Start building the vertex data structure
            vertex_data = [
                ('x', 'f4'),
                ('y', 'f4'),
                ('z', 'f4')
            ]
            
            # Prepare the data tuple list
            data_tuples = []
            for i in range(len(points)):
                row = [points[i, 0], points[i, 1], points[i, 2]]
                data_tuples.append(tuple(row))
            
            # Check for color columns (handle different formats)
            colors = None
            
            # Method 1: Separate R, G, B columns
            separate_color_sets = [
                ['red', 'green', 'blue'],
                ['r', 'g', 'b'],
                ['R', 'G', 'B'],
                ['Red', 'Green', 'Blue']
            ]
            
            for color_set in separate_color_sets:
                if all(col in self.df.columns for col in color_set):
                    logger.info(f"Found separate color columns: {color_set}")
                    colors = self.df[color_set].values
                    break
            
            # Method 2: Single 'color' column with tuples
            if colors is None and 'color' in self.df.columns:
                logger.info("Found single 'color' column with RGB tuples")
                try:
                    # Extract RGB values from tuples
                    color_tuples = self.df['color'].values
                    # Convert tuples to numpy array
                    colors = np.array([list(color_tuple) for color_tuple in color_tuples])
                    logger.info(f"Extracted {len(colors)} color tuples")
                except Exception as e:
                    logger.warning(f"Could not parse color tuples: {e}")
                    colors = None
            
            # Method 3: Check for other single color column names
            if colors is None:
                for col_name in ['rgb', 'RGB', 'colors', 'colour']:
                    if col_name in self.df.columns:
                        logger.info(f"Found color column: {col_name}")
                        try:
                            color_data = self.df[col_name].values
                            if len(color_data[0]) == 3:  # Should be RGB triplet
                                colors = np.array([list(color_item) for color_item in color_data])
                                break
                        except:
                            continue
            
            if colors is not None:
                # Handle color range - ensure 0-255 range
                if colors.max() <= 1.0:
                    colors = (colors * 255).astype(np.uint8)
                else:
                    colors = colors.astype(np.uint8)
                
                logger.info(f"Color value range: {colors.min()} to {colors.max()}")
                
                # Add color data types
                vertex_data.extend([
                    ('red', 'u1'),
                    ('green', 'u1'),
                    ('blue', 'u1')
                ])
                
                # Rebuild data tuples with colors
                new_data_tuples = []
                for i, base_tuple in enumerate(data_tuples):
                    new_tuple = base_tuple + (colors[i, 0], colors[i, 1], colors[i, 2])
                    new_data_tuples.append(new_tuple)
                data_tuples = new_data_tuples
                
                logger.info(f"Added RGB colors for {len(colors):,} points")
            else:
                logger.info("No color columns found, saving points only")
            
            # Check for normal columns
            normal_cols = None
            possible_normal_sets = [
                ['nx', 'ny', 'nz'],
                ['normal_x', 'normal_y', 'normal_z']
            ]
            
            for normal_set in possible_normal_sets:
                if all(col in self.df.columns for col in normal_set):
                    normal_cols = normal_set
                    break
            
            if normal_cols:
                logger.info(f"Found normal columns: {normal_cols}")
                normals = self.df[normal_cols].values.astype(np.float32)
                
                # Add normal data types
                vertex_data.extend([
                    ('nx', 'f4'),
                    ('ny', 'f4'),
                    ('nz', 'f4')
                ])
                
                # Rebuild data tuples with normals
                new_data_tuples = []
                for i, base_tuple in enumerate(data_tuples):
                    new_tuple = base_tuple + (normals[i, 0], normals[i, 1], normals[i, 2])
                    new_data_tuples.append(new_tuple)
                data_tuples = new_data_tuples
            
            # Create the vertex array
            vertex_array = np.array(data_tuples, dtype=vertex_data)
            
            # Create PLY element
            vertex_element = PlyElement.describe(vertex_array, 'vertex')
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            
            # Write PLY file
            PlyData([vertex_element], text=True).write(filename)
            
            # Verify file was created successfully
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                logger.info(f"Successfully saved {len(points):,} points to {filename}")
                logger.info(f"File size: {file_size:,} bytes")
                return True
            else:
                logger.error(f"File was not created: {filename}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving PLY file: {e}")
            return False
    
    def save_point_cloud(self, filename):
        """
        Legacy method name for backward compatibility
        """
        return self.to_ply_with_plyfile(filename)
    
    def get_point_cloud_stats(self):
        """Get statistics about the point cloud"""
        if self.df is None or len(self.df) == 0:
            return "No point cloud data"
        
        stats = {
            'num_points': len(self.df),
            'columns': list(self.df.columns),
            'bounds': {}
        }
        
        # Calculate bounds for coordinate columns
        for col in ['x', 'y', 'z']:
            if col in self.df.columns:
                stats['bounds'][col] = {
                    'min': float(self.df[col].min()),
                    'max': float(self.df[col].max())
                }
        
        return stats
    

def parse_bbox(bbox_str: str) -> List[float]:
    """
    Parse bbox string into list of coordinates.
    
    Args:
        bbox_str: Comma-separated string of coordinates "min_lon,min_lat,max_lon,max_lat"
        
    Returns:
        List of four float coordinates
        
    Raises:
        ValueError: If bbox format is invalid
    """
    try:
        coords = [float(x.strip()) for x in bbox_str.split(',')]
        if len(coords) != 4:
            raise ValueError(
                f"bbox must contain exactly 4 coordinates, got {len(coords)}"
            )
        
        min_lon, min_lat, max_lon, max_lat = coords
        
        # Validate coordinate ranges
        if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
            raise ValueError(
                "Longitude must be between -180 and 180"
            )
        if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
            raise ValueError(
                "Latitude must be between -90 and 90"
            )
        if min_lon >= max_lon:
            raise ValueError(
                "min_lon must be less than max_lon"
            )
        if min_lat >= max_lat:
            raise ValueError(
                "min_lat must be less than max_lat"
            )
            
        return coords
    except ValueError as e:
        raise ValueError(
            f"Invalid bbox format. Expected 'min_lon,min_lat,max_lon,max_lat': {e}"
        )

def parse_arguments() -> Dict[str, Union[str, List[float], float]]:
    """
    Parse command line arguments using sys.argv.
    
    Returns:
        Dictionary with parsed bbox and cloud cover
    """
    logger.info(f"Command line arguments: {sys.argv}")
    logger.info(f"Number of arguments: {len(sys.argv)}")

    #if len(sys.argv) < 6:
        #raise ValueError("Missing required arguments: key_id, access_key, eh_token are required")
    
    key_id = sys.argv[1]
    logger.info(f"Key id: {key_id}")
    access_key = sys.argv[2] 
    logger.info(f"Access key provided: {'*' * len(access_key)}")  # Don't log actual key
    eh_token = sys.argv[3]
    logger.info(f"EH token provided: {'*' * len(eh_token)}")  # Don't log actual token
    
    bbox_str = sys.argv[4]
    logger.info(f"Using bbox from command line: {bbox_str}")
    
    # Parse and validate bbox
    bbox = parse_bbox(bbox_str)
    
    sampled_fraction = float(sys.argv[5])
    logger.info(f"Sampled fraction: {sampled_fraction} (type: {type(sampled_fraction)})")
    
    return {
        "key_id": key_id,
        "access_key": access_key,
        "eh_token": eh_token,
        "bbox": bbox,
        "sampled_fraction": sampled_fraction
    }

def initialize_env(key_id: str, access_key: str, eh_token: str,
                bbox_coords: List[float], sampled_fraction: float = 0.2) -> Dict[str, Union[List[float], float]]:
    """
    Initialize environment variables
    
    Args:
        key_id: S3 access key ID for authentication
        access_key: S3 secret access key for authentication  
        eh_token: Earth Hub authentication token
        bbox_coords: List of bounding box coordinates [min_lon, min_lat, max_lon, max_lat]
        sampled_fraction: Fraction for point cloud downsampling (default: 0.2)

    Returns:
        Dictionary with parsed bbox and cloud cover
    """
    try:
        load_dotenv()
        logger.info("Environment variables loaded successfully")

        return {
            "key_id": key_id,
            "access_key": access_key,
            "eh_token": eh_token,
            "bbox": bbox_coords,
            "sampled_fraction": sampled_fraction
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize environment: {e}")
        raise

def main() -> None:
    logger.info("=== STARTING POINT CLOUD GENERATOR ===")

    # Parse arguments with better error handling
    try:
        logger.info("Parsing command line arguments")
        args = parse_arguments()
        logger.info("Arguments parsed successfully")
    except ValueError as e:
        logger.error(f"Argument parsing failed: {e}")

    try:
        logger.info("Initializing environment")
        env = initialize_env(
            args["key_id"], 
            args["access_key"], 
            args["eh_token"], 
            args["bbox"],
            args["sampled_fraction"]
        )
        logger.info("Environment initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing environment: {e}")


    # Extract variables
    key_id = env["key_id"]
    access_key = env["access_key"]
    eh_token = env["eh_token"]
    bbox = env["bbox"]  
    sampled_fraction = env["sampled_fraction"]

    dir_path = os.getcwd()
    logger.info(f"Current working directory: {dir_path}")

    endpoint_url = 'https://eodata.dataspace.copernicus.eu'
    endpoint_stac= 'https://stac.dataspace.copernicus.eu/v1/'
    bucket_name= 'eodata'

    # Access STAC API
    try:
        logger.info("Connecting to STAC catalog")
        stac_url = endpoint_stac
        catalog = pystac_client.Client.open(stac_url)
        logger.info(f"Connected to STAC catalog: {stac_url}")
    except Exception as e:
        logger.error(f"Error connecting to STAC catalog: {e}")
        sys.exit(1)

    logger.info(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
    # Access Sentinel-2 L2A products - STAC 
    try:
        logger.info("Querying for most recent Sentinel-2 L2A data")
        l2a_item = data_query_most_recent(catalog, bbox, 20)
        logger.info(f"Found L2A item: {l2a_item.id} with datetime {l2a_item.properties.get('datetime')}")
    except Exception as e:
        logger.error(f"Error accessing L2A item: {e}")

    
    # Setup S3 connection
    try:
        logger.info("Setting up S3 connection")
        connector = S3Connector(
            endpoint_url=endpoint_url,
            access_key_id=key_id,
            secret_access_key=access_key,
            region_name='default'       
        )
        logger.info("S3 connector created successfully")
    except Exception as e:
        logger.error(f"Error creating S3 connector: {e}")
        sys.exit(1)

    s3 = connector.get_s3_resource()
    s3_client = connector.get_s3_client()
    buckets = connector.list_buckets()
    logger.info(f"Available buckets: {buckets}")
    
    # Extract and download Sentinel-2 data
    try:
        logger.info("Extracting product URL from STAC item")
        product_url = extract_s3_path_from_url(l2a_item.assets['TCI_20m'].href)
        logger.info(f"Extracted product URL: {product_url}")

        logger.info("Downloading Sentinel-2 image")
        file_path_s2 = get_product(s3_resource=s3, bucket_name=bucket_name,
                                 object_url=product_url,
                                 output_path=".")
        logger.info(f"S2 image saved at: {file_path_s2}")
    except Exception as e:
        logger.error(f"Error downloading Sentinel-2 data: {e}")
        sys.exit(1)
    
    # Process Sentinel-2 data
    try:
        logger.info("Processing Sentinel-2 data")
        reader = Sentinel2Reader(filepath=file_path_s2, preprocess=True)
        
        bounds = reader.bounds
        width = reader.width
        height = reader.height
        logger.info(f"Image bounds: {bounds}")
        logger.info(f"Image dimensions - Width: {width}, Height: {height}")
    except Exception as e:
        logger.error(f"Error processing Sentinel-2 data: {e}")

    # Download DEM data
    try:
        logger.info("Downloading DEM data from Earth Data Hub")
        data_url = f"https://edh:{eh_token}@data.earthdatahub.destine.eu/copernicus-dem-utm/GLO-30-UTM-v0/32N"
        
        dem_data = load_dem_utm(url=data_url,
                                bounds=bounds,
                                width=width,
                                height=height)
        logger.info("DEM data loaded successfully")
        logger.info(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
    except Exception as e:
        logger.error(f"Error loading DEM data: {e}")
    
    # Generate point cloud
    try:
        logger.info("Initializing Point Cloud Generator")
        pcd_gen = PcdGenerator(reader.data, dem_data["dem"])
        logger.info(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")

        logger.info("Generating point cloud")
        pcd_gen.generate_point_cloud()
        
        logger.info(f"Downsampling point cloud by factor: {sampled_fraction}")
        pcd_gen.downsample(sample_fraction=sampled_fraction)
        logger.info(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")

        logger.info("Point cloud downsampling completed")
    except Exception as e:
        logger.error(f"Error generating point cloud: {e}")



    try:
        logger.info("Processing point cloud for export")

        pcd_df = pcd_gen.df
        logger.info(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
        logger.info(f"Point cloud dataframe type: {type(pcd_df)}")
        logger.info(f"Dataframe shape: {pcd_df.shape}")
        logger.info(f"Dataframe columns: {list(pcd_df.columns)}")
        
        # Initialize handler
        handler = PointCloudHandler(pcd_df)
        logger.info(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
        
        # Get and log statistics
        stats = handler.get_point_cloud_stats()
        logger.info(f"Point cloud statistics: {stats}")
        
        # Save using plyfile (Docker-friendly)
        output_file = "./point_cloud.ply"
        success = handler.to_ply_with_plyfile(output_file)
        logger.info(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
        if success:
            logger.info(f"Point cloud saved successfully at: {output_file}")
            
            # Verify file was created and get details
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                logger.info(f"Output file size: {file_size:,} bytes")
                
                
                # # Optional: Quick verification read
                # try:
                #     from plyfile import PlyData
                #     plydata = PlyData.read(output_file)
                #     vertex_count = plydata['vertex'].count
                #     logger.info(f"Verification: PLY file contains {vertex_count:,} vertices")
                # except Exception as ve:
                #     logger.warning(f"Could not verify PLY file: {ve}")
            else:
                logger.error(f"Output file was not created: {output_file}")
                return False
        else:
            logger.error("Failed to save point cloud")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error processing and saving point cloud: {e}")
        return False
    

    
if __name__ == "__main__":
    main()