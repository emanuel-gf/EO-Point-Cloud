
import os
import sys
import datetime
import time

# Set environment variables early
os.environ['EGL_PLATFORM'] = 'surfaceless'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

from loguru import logger

from typing import Dict, List, Union
from dotenv import load_dotenv
import pystac_client
import traceback

from src.utils import (PcdGenerator, PointCloudHandler, Sentinel2Reader,load_config,
                          load_dem_utm)
from src.s3_bucket import S3Connector, extract_s3_path_from_url, get_product
from src.stac import data_query_most_recent

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

    if len(sys.argv) < 6:
        raise ValueError("Missing required arguments: key_id, access_key, eh_token are required")
    
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
        traceback.print_exc()
        sys.exit(1)

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
        traceback.print_exc()
        sys.exit(1)

    # Extract variables
    key_id = env["key_id"]
    access_key = env["access_key"]
    eh_token = env["eh_token"]
    bbox = env["bbox"]  
    sampled_fraction = env["sampled_fraction"]

    dir_path = os.getcwd()
    logger.info(f"Current working directory: {dir_path}")

    # Load configuration
    try:
        query_cfg = load_config(f"{dir_path}/src/query_config.yml")
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

    # Access STAC API
    try:
        logger.info("Connecting to STAC catalog")
        stac_url = query_cfg["endpoint_stac"]
        catalog = pystac_client.Client.open(stac_url)
        logger.info(f"Connected to STAC catalog: {stac_url}")
    except Exception as e:
        logger.error(f"Error connecting to STAC catalog: {e}")
        sys.exit(1)

    # Access Sentinel-2 L2A products - STAC 
    try:
        logger.info("Querying for most recent Sentinel-2 L2A data")
        l2a_item = data_query_most_recent(catalog, bbox, 20)
        logger.info(f"Found L2A item: {l2a_item.id} with datetime {l2a_item.properties.get('datetime')}")
    except Exception as e:
        logger.error(f"Error accessing L2A item: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Setup S3 connection
    try:
        logger.info("Setting up S3 connection")
        connector = S3Connector(
            endpoint_url=query_cfg["endpoint_url"],
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
        file_path_s2 = get_product(s3_resource=s3, bucket_name=query_cfg["bucket_name"],
                                 object_url=product_url,
                                 output_path=".")
        logger.info(f"S2 image saved at: {file_path_s2}")
    except Exception as e:
        logger.error(f"Error downloading Sentinel-2 data: {e}")
        traceback.print_exc()
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
        traceback.print_exc()
        sys.exit(1)

    # Download DEM data
    try:
        logger.info("Downloading DEM data from Earth Data Hub")
        data_url = f"https://edh:{eh_token}@data.earthdatahub.destine.eu/copernicus-dem-utm/GLO-30-UTM-v0/32N"
        
        dem_data = load_dem_utm(url=data_url,
                                bounds=bounds,
                                width=width,
                                height=height)
        logger.info("DEM data loaded successfully")
    except Exception as e:
        logger.error(f"Error loading DEM data: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Generate point cloud
    try:
        logger.info("Initializing Point Cloud Generator")
        pcd_gen = PcdGenerator(reader.data, dem_data["dem"])

        logger.info("Generating point cloud")
        pcd_gen.generate_point_cloud()
        
        logger.info(f"Downsampling point cloud by factor: {sampled_fraction}")
        pcd_gen.downsample(sample_fraction=sampled_fraction)
        logger.info("Point cloud downsampling completed")
    except Exception as e:
        logger.error(f"Error generating point cloud: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Process and save point cloud
    try:
        logger.info("Processing Open3D point cloud")
        pcd_df = pcd_gen.df
        logger.info(f"Point cloud dataframe type: {type(pcd_df)}")
        
        handler = PointCloudHandler(pcd_df)
        handler.to_open3d()
        
        output_file = "./point_cloud.ply"
        handler.save_point_cloud(output_file)
        logger.info(f"Point cloud saved successfully at: {output_file}")
        
        # Verify file was created
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            logger.info(f"Output file size: {file_size} bytes")
        else:
            logger.error(f"Output file was not created: {output_file}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error processing and saving point cloud: {e}")
        traceback.print_exc()
        sys.exit(1)

    logger.info("=== POINT CLOUD GENERATION COMPLETED SUCCESSFULLY ===")
    
if __name__ == "__main__":
    main()