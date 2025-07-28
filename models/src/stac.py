import io
import os
from urllib.parse import urlparse
import random

from loguru import logger
import boto3
import pystac_client
from dotenv import load_dotenv
from PIL import Image
import datetime 
import numpy as np
import pandas as pd 



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
        end_date = datetime.date.today().strftime("%Y-%m-%d")
        start_date = (datetime.date.today() - datetime.timedelta(days=default_timedelta)).strftime("%Y-%m-%d")
        
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