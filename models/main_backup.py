import os
import sys
import datetime
import time
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['EGL_PLATFORM'] = 'surfaceless'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

from loguru import logger

from typing import Dict, List, Union
from dotenv import load_dotenv
import pystac_client
import subprocess
import traceback

from src.utils import (PcdGenerator, PointCloudHandler, Sentinel2Reader,load_config,
                          load_dem_utm)
from src.s3_bucket import S3Connector, extract_s3_path_from_url, get_product
from src.stac import data_query_most_recent

def log(level, message, **kwargs):
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    extra = " ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else ""
    log_line = f"{timestamp} [{level}] {message} {extra}".strip()
    
    if level in ["ERROR", "CRITICAL"]:
        print(log_line, file=sys.stderr, flush=True)
    else:
        print(log_line, file=sys.stdout, flush=True)

def log_info(message, **kwargs):
    log("INFO", message, **kwargs)

def log_error(message, **kwargs):
    log("ERROR", message, **kwargs)

def log_debug(message, **kwargs):
    log("DEBUG", message, **kwargs)


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
    logger.info(f" access key: { access_key}")
    eh_token = sys.argv[3]
    logger.info(f"eh token {eh_token}")
    # Get bbox from sys.argv[1] or use default

    bbox_str = sys.argv[4]
    logger.info(f"Using bbox from command line: {bbox_str}")
    
    # Parse and validate bbox
    bbox = parse_bbox(bbox_str)
    
    sampled_fraction = float(sys.argv[5])
    logger.info(f"sampled fa")
    logger.info(f" Sampled fraction {sampled_fraction} as type {type(sampled_fraction)}")
    return {
        "key_id": key_id,
        "access_key": access_key,
        "eh_token": eh_token,
        "bbox": bbox,
        "sampled_fraction":sampled_fraction
    }

def initialize_env(key_id:str,access_key:str,eh_token:str,
                bbox_coords: List[float], sampled_fraction:float=0.2) -> Dict[str, Union[List[float], float]]:
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
        logger.success("Loaded environment variables")

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
    print("Aqui ja dentro do loop do main")
    print("Esse eh um print de teste dessa porra do caralho",flush=True)
    print("Ja esse print aqui eh sem flush",flush=False)
    print("Esse print do caralho eh loggado no sterror", file=sys.stderr)
    print("Ja esse ultimo aqui eh o demonio no std out", file=sys.stdout)
     # Remove the file logger and configure for stdout

    
    os.environ['PYTHONUNBUFFERED'] = '1'
    logger.info("Starting main function")
    

    ##### TESTES 
    log_info("Parsing command line arguments")
    log_info(f"Command line arguments: {sys.argv}")
    log_info(f"Number of arguments: {len(sys.argv)}")

    print("=== KUBERNETES LOG TEST START ===", flush=True)

    for i in range(5):
        print(f"INFO: Test message {i+1}", flush=True)
        print(f"ERROR: Test error {i+1}", file=sys.stderr, flush=True)
        time.sleep(1)

    print("=== KUBERNETES LOG TEST END ===", flush=True)

    # Parse arguments with better error handling
    try:
        args = parse_arguments()
        print("Sucess with parsing argument", file=sys.stdout)
    except ValueError as e:
        logger.error(f"Argument parsing failed: {e}")
        print("Error:", e, file=sys.stdout)
        traceback.print_exc()
        sys.exit(1)

    try:
        env = initialize_env(
            args["key_id"], 
            args["access_key"], 
            args["eh_token"], 
            args["bbox"],
            args["sampled_fraction"]
        )


    except ValueError as e:
        logger.error(f"Error at initializing env")
        sys.exit(1)


    # Current Dir
    dir_path = os.getcwd() 

    key_id = env["key_id"]
    access_key  = env["access_key"]
    eh_token = env["eh_token"]
    bbox = env["bbox"]  
    sampled_fraction = env["sampled_fraction"]

    dir_path = os.getcwd()
    logger.info(f"Current working directory: {dir_path}")

    query_cfg = load_config(f"{dir_path}/src/query_config.yml")

    ## Access STAC API
    try:
        stac_url = query_cfg["endpoint_stac"]
        catalog = pystac_client.Client.open(stac_url)
    except ValueError as e:
        logger.error(f"Error at stac catalog")
        sys.exit(1)

    
    ## Access Sentinel-2 L2A products - STAC 
    
    try:
        l2a_item  = data_query_most_recent(catalog, bbox,  20)
        logger.info(f"Found L2A item: {l2a_item.id} with datetime {l2a_item.properties.get('datetime')}")
    except Exception as e:
        logger.error(f"Error accessing L2A item: {e}")
        sys.exit(1)  # Exit with error code
        return None
    
    try:
        connector = S3Connector(
            endpoint_url=query_cfg["endpoint_url"],
            access_key_id=key_id,
            secret_access_key=access_key,
            region_name='default'       
        )
    except Exception as e:
        logger.error(f"Error at S3Connector")
        sys.exit(1)

    s3 = connector.get_s3_resource()
    s3_client = connector.get_s3_client()
    buckets = connector.list_buckets()
    logger.info(f"Available buckets: {buckets}")
    
    # Extract url 
    product_url = extract_s3_path_from_url(l2a_item.assets['TCI_20m'].href)
    logger.info(f"Extracted product URL: {product_url}")

    ## Download Image and save in the path 
    file_path_s2 = get_product(s3_resource=s3, bucket_name=query_cfg["bucket_name"],
                             object_url=product_url,
                             output_path=".")
    logger.info(f"S2 image saved at: {file_path_s2}")
    
    ##Extract information from S2 
    reader = Sentinel2Reader(filepath=file_path_s2,
                             preprocess=True)
    
    bounds = reader.bounds
    width = reader.width
    height = reader.height
    logger.info(f"Bounds:{bounds}")
    logger.info(f"Width: {width}, heigh:{height}")

    ## Get DEM
    ## Path for the Earth Hub DEM Data 
    data_url = f"https://edh:{eh_token}@data.earthdatahub.destine.eu/copernicus-dem-utm/GLO-30-UTM-v0/32N"
    
    logger.info(f"Downloading DEM data")
    dem_data = load_dem_utm(url=data_url,
                            bounds=bounds,
                            width=width,
                            height=height)
    logger.success(f"DEM loaded sucessfully")
    
    logger.info(f"Initialize Point Cloud handler")
    ## Initialize and generate point cloud
    pcd_gen = PcdGenerator(reader.data,
                            dem_data["dem"])

    # downsample point cloud data using Open3D functionality
    pcd_gen.generate_point_cloud()
    pcd_gen.downsample(sample_fraction=sampled_fraction)
    logger.info(f"Points downsampled by {sampled_fraction}")


    logger.info(f"Handling Open3D point cloud creation")
    # # Process and save point cloud and mesh
    pcd_df = pcd_gen.df
    print(f"Type of pcd_df {pcd_df}")
    handler = PointCloudHandler(pcd_df)
    handler.to_open3d()
    #handler.generate_mesh(depth=7)

    handler.save_point_cloud("./point_cloud.ply")
    logger.success(f"Point cloud saved at: {"./point_cloud.py"}")

    #handler.save_mesh("./mesh.glb")
    #logger.success(f"Mesh saved at: {"./mesh.glb"}")

    logger.success("Completely Sucesfully ")
    
if __name__ == "__main__":
    print("Esse eh um print de teste dessa porra do caralho",flush=True)
    print("Ja esse print aqui eh sem flush",flush=False)
    print("Esse print do caralho eh loggado no sterror", file=sys.stderr)
    print("Ja esse ultimo aqui eh o demonio no std out", file=sys.stdout)
    logger.remove()  # Remove default handler
    logger.add(sys.stdout, level="INFO", format="{time} | {level} | {message}")
    main()