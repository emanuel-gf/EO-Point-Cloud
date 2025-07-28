#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================================================
File Downloader Tool | Sentinel-2, Copernicus DEM and LiDAR Data Processing Tool
===========================================================

This script provides a function to download files from a given URL using wget.
The primary functionality includes:

- Downloading files from a URL
- Saving the downloaded files to a specified directory

This script provides a set of classes and functions to process
Copernicus DEM, Sentinel-2 L2A product, IGN LiDAR point cloud data.
The primary functionalities include:

- Loading and preprocessing Sentinel-2 imagery
- Extracting and processing DEM data
- Reading, transforming, and filtering LiDAR point cloud data
- Generating and visualizing point clouds and 3D meshes
- Downsampling and filtering datasets for optimized processing


Author: Sébastien Tétaud
Modified by: Emanuel Goulart
Date: 2025-03-14
License: Apache 2.0
"""
import json

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import pdal
import rasterio
import xarray as xr
from pyproj import Transformer
import xarray as xr
import numpy as np
import os
import subprocess
from loguru import logger
import yaml  

def load_config(config_path: str = "cfg/config.yaml") -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): The path to the configuration YAML file.

    Returns:
        dict: The loaded configuration dictionary.
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        logger.success(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file {config_path} not found")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML configuration: {e}")
        return {}
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return {}
    
def download_file(url, download_dir):
    """
    Downloads a file from a given URL using wget.

    Parameters:
        url (str): The URL of the file to download.
        download_dir (str): The directory where the file will be saved.

    Returns:
        str: Path to the downloaded file, or None if the download fails.
    """
    file_name = os.path.join(download_dir, os.path.basename(url))

    try:
        # Construct and execute the wget command
        subprocess.run(['wget', '-O', file_name, url], check=True)
        print(f"Downloaded: {file_name}")
        return file_name
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {url}: {e}")
        return None
 

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

    def show_image(self):
        """
        Displays the processed Sentinel-2 image.
        """
        if self.data is not None:
            plt.imshow(self.data)
            plt.axis("off")
            plt.show()
        else:
            print("No image data loaded.")


class IGNLidarProcessor:
    """
    A class to process LiDAR point cloud data from IGN.

    Attributes:
        input_file (str): Path to the input LiDAR file.
        df (pd.DataFrame): DataFrame to store the point cloud data.
    """

    def __init__(self, input_file: str):
        """
        Initialize the IGNLidarProcessor with the input file path.

        Args:
            input_file (str): Path to the input LiDAR file.
        """
        self.input_file = input_file
        self.df = None

    def read_point_cloud(self):
        """
        Read the point cloud data from the input file using PDAL.

        Raises:
            Exception: If there is an error in reading the file.
        """
        # Define a PDAL pipeline to read the file
        pipeline = {
            "pipeline": [
                {
                    "type": "readers.las",
                    "filename": self.input_file
                }
            ]
        }
        # Convert the pipeline to JSON format
        pipeline_json = json.dumps(pipeline)
        # Initialize the PDAL pipeline
        p = pdal.Pipeline(pipeline_json)
        # Execute the pipeline
        _ = p.execute()
        # Extract the point cloud data
        pcd = p.arrays
        self.df = pd.DataFrame(pcd[0])

    def transform_coordinates(self, src_crs: str = "epsg:2154", dst_crs: str = "epsg:32632"):
        """
        Transform the coordinates from the source CRS to the destination CRS.

        Args:
            src_crs (str): Source coordinate reference system. Default is "epsg:2154".
            dst_crs (str): Destination coordinate reference system. Default is "epsg:32632".

        Raises:
            ValueError: If the point cloud data has not been read yet.
        """
        if self.df is None:
            raise ValueError("Point cloud data has not been read yet.")

        # Initialize the transformer
        transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        # Convert X, Y coordinates
        utm_easting, utm_northing = transformer.transform(self.df["X"].values,
                                                          self.df["Y"].values)
        # Add transformed coordinates to DataFrame
        self.df["UTM_Easting"] = utm_easting
        self.df["UTM_Northing"] = utm_northing

        self.df = pd.DataFrame(data={
            "x": self.df["UTM_Easting"],
            "y": self.df["UTM_Northing"],
            "z": self.df["Z"],
            'classification': self.df["Classification"]
        })

    def downsample(self, sample_fraction=0.1, random_state=42):
        """
        Downsample the point cloud by randomly sampling a fraction of the points.

        Args:
            sample_fraction (float): The fraction of points to sample. Default is 0.1.
            random_state (int): Seed for the random number generator. Default is 42.
        """
        if self.df is None:
            raise ValueError("Point cloud data not generated. Run `generate_point_cloud()` first.")

        # Perform random sampling
        self.df = self.df.sample(frac=sample_fraction, random_state=random_state)
        print(f"Point cloud downsampled with sample fraction {sample_fraction}")

    def classes_filtering(self, list_classes=[9, 6]):
        """
        Filter the point cloud data based on specified classification values.

        Args:
            list_classes (list): List of classification values to keep. Default is [9, 6].
        """
        if self.df is None:
            raise ValueError("Point cloud data has not been read yet.")

        self.df = self.df[self.df['classification'].isin(list_classes)]

    def generate_color(self):
        """
        Generate unique colors for each classification in the point cloud data.

        Returns:
            dict: A dictionary mapping classification values to RGB colors.

        Raises:
            ValueError: If the point cloud data has not been read yet.
        """
        if self.df is None:
            raise ValueError("Point cloud data has not been read yet.")

        # Get unique classification values
        unique_classes = self.df["classification"].unique()
        num_classes = len(unique_classes)
        cmap = plt.cm.get_cmap("tab10", num_classes)  # Choose a colormap with enough distinct colors
        class_colors = {cls: tuple((np.array(cmap(i)[:3]) * 255).astype(int)) for i, cls in enumerate(unique_classes)}
        self.df["color"] = self.df["classification"].map(class_colors)

    def assign_custom_colors(self):
        """
        Assign custom colors for specific classification categories.
        """
        if self.df is None:
            raise ValueError("Point cloud data has not been read yet.")

        color_mapping = {
            1: (200, 200, 200),   # Non classé (Gray)
            2: (139, 69, 19),     # Sol (Brown)
            3: (34, 139, 34),     # Végétation basse (Green)
            4: (0, 100, 0),       # Végétation moyenne (Dark Green)
            5: (0, 50, 0),        # Végétation haute (Darker Green)
            6: (255, 0, 0),       # Bâtiment (Red)
            66: (255, 20, 147)    # Points virtuels (Pink)
        }

        self.df["color"] = self.df["classification"].map(lambda x: np.array(color_mapping.get(x, (0, 0, 0)), dtype=np.int64))
        self.df["color"] = self.df["color"].apply(
            lambda x: np.array(x[:3]) if isinstance(x, (list, tuple, np.ndarray)) and len(x) >= 3 else np.array([0, 0, 0])
        )


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
        print(f"Total points before sampling: {len(self.df)}")

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
        self.df = self.df.sample(frac=sample_fraction, random_state=random_state)
        print(f"Point cloud downsampled with sample fraction {sample_fraction}")
        logger.info(f"Point cloud downsampled with sample fraction {sample_fraction}")


class PointCloudHandler:
    """
    A class to handle operations related to the Open3D point cloud object.

    Attributes:
        df (pd.DataFrame): DataFrame holding point cloud data.
        point_cloud (o3d.geometry.PointCloud): Open3D PointCloud object.
    """

    def __init__(self, df):
        """
        Initialize the PointCloudHandler with a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame holding point cloud data.
        """
        self.df = df
        self.point_cloud = None
        self.mesh = None  # Store mesh object

    def downsample(self, sample_fraction=0.1, random_state=42):
        """
        Downsample the point cloud by randomly sampling a fraction of the points.

        Args:
            sample_fraction (float): The fraction of points to sample. Default is 0.1.
            random_state (int): Seed for the random number generator. Default is 42.
        """
        if self.df is None:
            raise ValueError("Point cloud data not available.")

        self.df = self.df.sample(frac=sample_fraction,
                                 random_state=random_state)
        print(f"Point cloud downsampled (fraction: {sample_fraction})")

    def to_open3d(self):
        """Convert the DataFrame to an Open3D PointCloud object, normalize, and assign colors."""
        if self.df is None:
            raise ValueError("Point cloud data not available.")

        # Extract XYZ coordinates
        xyz = self.df[['x', 'y', 'z']].values.astype(np.float64)

        # Normalize: Center and Scale
        center = np.mean(xyz, axis=0)
        xyz -= center  # Center the point cloud
        scale_factor = np.max(np.abs(xyz))  # Get the largest absolute value
        xyz /= scale_factor  # Scale to fit within a reasonable range

        # Extract and normalize colors
        colors = np.stack(self.df['color'].apply(lambda x: np.array(x) / 255.0))
        colors = np.clip(colors, 0.0, 1.0)
        # Create Open3D PointCloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        self.point_cloud = pcd
        print("Open3D PointCloud object created successfully.")
        return pcd

    def generate_mesh(self, depth=9):
        """Generate a 3D mesh from the point cloud using Poisson reconstruction."""
        if self.point_cloud is None:
            raise ValueError("Point cloud not generated yet.")

        self.point_cloud.estimate_normals()
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.point_cloud, depth=depth)

        if mesh.is_empty():
            raise RuntimeError("Surface reconstruction failed")

        self.mesh = mesh
        print("Mesh generated successfully.")
        return mesh

    def save_point_cloud(self, filename="point_cloud.ply"):
        """Save the point cloud to a file."""
        if self.point_cloud is None:
            raise ValueError("Point cloud not generated yet.")

        o3d.io.write_point_cloud(filename, self.point_cloud)
        print(f"Point cloud saved to {filename}")

    def save_mesh(self, filename="mesh.glb"):
        """Save the mesh as GLB format."""
        if self.mesh is None:
            raise ValueError("Mesh not generated yet.")

        o3d.io.write_triangle_mesh(filename, self.mesh)
        print(f"Mesh saved to {filename} (GLB format).")

    def visualize(self):
        """Visualize the point cloud using Open3D."""
        if self.point_cloud is None:
            raise ValueError("Point cloud not generated yet.")

        o3d.visualization.draw_geometries([self.point_cloud])


class PcdFilter:
    """
    A class to filter and concatenate DataFrames based on bounding box coordinates.

    Attributes:
        df (pd.DataFrame): The DataFrame to be filtered.
        df_target (pd.DataFrame): The target DataFrame to compute the bounding box from.
        bbox (dict): A dictionary containing the bounding box coordinates.
    """

    def __init__(self, df, df_target):
        """
        Initialize the PcdFilter with a DataFrame and a target DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to be filtered.
            df_target (pd.DataFrame): The target DataFrame to compute the bounding box from.
        """
        self.df = df
        self.df_target = df_target
        self.bbox = None

    def set_bounding_box_from_target(self, margin=2):
        """
        Set the bounding box coordinates based on a target DataFrame with an optional margin.

        Args:
            margin (float): Margin to extend the bounding box. Default is 2.
        """
        x_min, x_max = self.df_target["x"].min(), self.df_target["x"].max()
        y_min, y_max = self.df_target["y"].min(), self.df_target["y"].max()
        x_tile = x_max - x_min
        y_tile = y_max - y_min

        self.bbox = {
            "x_min": x_min - margin * x_tile,
            "x_max": x_max + margin * x_tile,
            "y_min": y_min - margin * y_tile,
            "y_max": y_max + margin * y_tile
        }

    def filter_data(self):
        """
        Filter the DataFrame based on the bounding box coordinates.

        Returns:
            pd.DataFrame: The filtered DataFrame.

        Raises:
            ValueError: If the bounding box is not set.
        """
        if self.bbox is None:
            raise ValueError("Bounding box is not set. Use `set_bounding_box_from_target` method first.")

        filtered_df = self.df[
            (self.df["x"] >= self.bbox["x_min"]) & (self.df["x"] <= self.bbox["x_max"]) &
            (self.df["y"] >= self.bbox["y_min"]) & (self.df["y"] <= self.bbox["y_max"])
        ]
        return filtered_df

    def concatenate_dataframes(self, df_list):
        """
        Concatenate a list of DataFrames.

        Args:
            df_list (list): A list of DataFrames to concatenate.

        Returns:
            pd.DataFrame: The concatenated DataFrame.
        """
        return pd.concat(df_list, axis=0)

