import os
import dotenv
from models.utils import (PcdGenerator, PointCloudHandler, Sentinel2Reader,
                          load_dem_utm)
from loguru import logger
from typing import Dict, List, Union
from dotenv import load_dotenv


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

def parse_arguments() -> Dict[str, Union[List[float], int]]:
    """
    Parse command line arguments using sys.argv.
    
    Returns:
        Dictionary with parsed bbox and cloud cover
    """
    logger.info(f"Command line arguments: {sys.argv}")
    logger.info(f"Number of arguments: {len(sys.argv)}")
    
    # Default values
    default_bbox = "3.2833,45.3833,11.2,50.1833"
    default_max_cloud_cover = 20
    
    # Get bbox from sys.argv[1] or use default
    if len(sys.argv) > 1:
        bbox_str = sys.argv[1]
        logger.info(f"Using bbox from command line: {bbox_str}")
    else:
        bbox_str = default_bbox
        logger.info(f"Using default bbox: {bbox_str}")
    
    # Get max_cloud_cover from sys.argv[2] or use default
    if len(sys.argv) > 2:
        max_cloud_cover = int(sys.argv[2])
        logger.info(f"Using max_cloud_cover from command line: {max_cloud_cover}")
    else:
        max_cloud_cover = default_max_cloud_cover
        logger.info(f"Using default max_cloud_cover: {max_cloud_cover}")
    
    # Parse and validate bbox
    bbox = parse_bbox(bbox_str)
    
    return {
        "bbox": bbox,
        "max_cloud_cover": max_cloud_cover
    }


def initialize_env(eh_token:str, bbox_coords: List[float], max_cloud_cover: int) -> Dict[str, Union[List[float], int]]:
    """
    Initialize environment variables
    
    Args:
        bbox_str: List of bounding box coordinates: min_lon,min_lat,max_lon,max_lat
        max_cloud_cover: Maximum cloud cover percentage
        
    Returns:
        Dictionary with parsed bbox and cloud cover
    """
    try:
        load_dotenv()
        logger.success("Loaded environment variables")

        return {
            eh_token: eh_token,
            "bbox": bbox_coords,
            "max_cloud_cover": max_cloud_cover
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize environment: {e}")
        raise

def main() -> None:
    logger.info("Starting main function")
    logger.add("EO-point-cloud.log", rotation="5MB")
    
    args = parse_arguments()

    env  = initialize_env(args["bbox"], args["max_cloud_cover"])

    dir_path = os.getcwd() 

    eh_token = env["eh_token"]
    bbox = env["bbox"]
    max_cloud_cover = env["max_cloud_cover"]   

    ## Path for the Earth Hub DEM Data 
    data_url = f"https://edh:{eh_token}@data.earthdatahub.destine.eu/copernicus-dem-utm/GLO-30-UTM-v0/32N"
    
    ## Download Sentinel-TCI data 
    
    product_path = "/home/ubuntu/project/destine-godot-mvp/src/sentinel2-data/T32TLR_20241030T103151_TCI_20m.jp2"
    reader = Sentinel2Reader(filepath=product_path, preprocess=True)
    bounds = reader.bounds
    width = reader.width
    height = reader.height
    dem_data = load_dem_utm(url=data_url, bounds=bounds, width=width, height=height)
    # Initialize and generate point cloud
    pcd_gen = PcdGenerator(reader.data, dem_data["dem"])

    # downsample point cloud data using Open3D functionality
    pcd_gen.generate_point_cloud()
    pcd_gen.downsample(sample_fraction=0.2)

    # Process and save point cloud and mesh
    handler = PointCloudHandler(pcd_gen.df)
    handler.to_open3d()
    handler.generate_mesh(depth=9)
    handler.save_point_cloud("point_cloud.ply")
    handler.save_mesh("mesh.glb")