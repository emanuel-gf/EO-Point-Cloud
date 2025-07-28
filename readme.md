# EO Point Cloud Generation

This repository contains a streaming of Earth Observation (EO) data for creating Point Cloud and 3D visualizations. The repository is focused on the implementation of a Delta Twin component. [Delta Twin](https://deltatwin.destine.eu/) is a project that belongs to [Destionation Earth](https://destination-earth.eu/) initiative. It allows modeling activities of digital twins and aims to offer a collaborative environment for building and running multi-scale and composable workflows.

This repository is
This repository contains a collection of Jupyter notebooks and utilities for generating 3D point clouds from Earth Observation (EO) data, including Digital Elevation Models (DEM), Sentinel-2 imagery, and LiDAR data.

## Overview

The project demonstrates various techniques for processing and visualizing geospatial data in 3D, with a focus on:

1. Accessing and visualizing DEM data from the DestinE Earth Data Hub (https://earthdatahub.destine.eu/getting-started)
2. Combining DEM data with Sentinel-2 imagery to create colorized 3D point clouds
3. Processing IGN HD LiDAR data for high-resolution terrain modeling
4. Integrating multiple data sources (DEM, Sentinel-2, and LiDAR) for comprehensive 3D models

## Repository Structure

```bash
├── LICENSE
├── README.md
├── notebooks
│   ├── basic_part_1.ipynb          # Basic DEM visualization
│   ├── basic_part_2.ipynb          # Point cloud generation from DEM and Sentinel-2
│   ├── basic_part_3.py.py          # IGN HD LiDAR processing
│   ├── intermediate_part_1.ipynb   # Integration of multiple data sources
└── src
    ├── data                        # Data storage directory
    │   ├── grille.zip
    │   ├── ign/
    │   └── sentinel2/
    │       └── T32TLR_20241030T103151_TCI_20m.jp2
    ├── point_cloud_generator.py
    └── util/                       # Utility functions
        ├── file_downloader.py      # Utility to download files
        └── general.py              # Common processing functions
```

## Notebooks

### Basic Part 1: DEM Visualization

The Source code do:
- Access Copernicus DEM data from the Earth Data Hub
- Access and Download it Sentinel-2 data using AWS S3 connectors.
- Allows the selection of the region of interest through a bbox.
- Retrieves the most recent image for a filtered cloud cover lesser than 20%. 
- Create a point cloud ready to be visualized in Open3d.

## Utility Functions

The repository includes several utility classes to streamline data processing:

- `Sentinel2Reader`: For reading and preprocessing Sentinel-2 imagery
- `PcdGenerator`: For creating point clouds from DEM and imagery data
- `PointCloudHandler`: For manipulating and saving point clouds


## Getting Started

1. Clone this repository

```bash
git 
```
2. Install the required dependencies

```bash
conda create --name pointcloud python==3.13.1
conda activate env
```
Run cli to install all the packages

```bash
sh install_packages.sh
```

3. Run the main.py at the root of the folder :
```bash
python main.py your_cdse_id your_cdse_access_key your_earth_data_hub_api_key bbox sampled_fraction
```


## License

Apache License 2.0

## Acknowledgments

 The majority part of the handler classes were developt by Sebastien Tetauld. 
 - [Sebastien Tetauld](https://github.com/sebastien-tetaud)
- [Copernicus DEM](https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model)
- [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)
- [DestinE Earth Data Hub](https://earthdatahub.destine.eu/)