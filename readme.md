# EO Point Cloud Generation

[fig-point-cloud](figs/tyrol2.jpg)

This repository aims on the creation Earth Observation Point Cloud. The repository is mainly focus at the implementation of EO-PointCloud as a component into the [DeltaTwin](https://deltatwin.destine.eu/) service of [Destination Earth](https://platform.destine.eu/). [Delta Twin](https://deltatwin.destine.eu/) allows modeling activities of digital twins and aims to offer a collaborative environment for building and running multi-scale and composable workflows.

This repository contains two implementation of the EO-PointCloud. One using OPEN3D as the core of PointCloud, and the other one is a workaround with plyfile library which is capable of being injected inside the Component of DeltaTwin. 

![Point CLoud](figs/gif_image.gif)

## Overview

The project demonstrates how to create a DeltaTwin component and generate a EO-PointCloud File.

1. Access [DeltaTwin](/DeltaTwin/) and see the three mandatory files to create a component: 
- [Manifest](/DeltaTwin/manifest.json)
- [Workflow](/DeltaTwin/workflow.yml)
- [Models](/DeltaTwin/models/)

In case of doubt and further explanation of how to build a DeltaTwin component, please refers to: [DeltaTwin Documentation](https://deltatwin.destine.eu/docs)

2. The folder [EO-PointCloud](/EO-Point-Cloud/) holds the implementation of EO-PointCloud with OPEN3D. For a proper installation of OPEN3D please check the file: [install_packages](/initial_setup/install_packages.sh)


## Quick start

To install the correct environment, it is recomended to use Conda and Conda-Forge. 
Please follow the steps given at [install_packages](/initial_setup/install_packages.sh)

## Getting Required API keys: 

To run any model is necessary having:
 - An account at [CDSE](https://dataspace.copernicus.eu/).
 - An account at [Destination Earth]((https://platform.destine.eu/))  

 ### CDSE API - Key and Secret 

- [CDSE](https://dataspace.copernicus.eu/) key and secret. 
This is related to fetching the Sentinel-2 images from CDSE ecosystem. 
Creating an account and fetching data is free for any user!
In case of problems finding the key and secret, check it here: [FAQ: key and secret](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html) 

### Earth Data Hub
- [Earth Data Hub](https://earthdatahub.destine.eu/) API-key. 
This is service inside Destination Earth, by having an account with any user has already access to the Earth Data Hub service. 
Get your API key at: 
1 - Log in
2 - Click in your username (upper-right corner)
3 - Account Settings 
4 - Add personal access token
5 - Done! Save carefully this token and pass into the Earth_data_hub_api for the main.py 


## Getting Started

1. Clone this repository

```bash
git clone https://github.com/emanuel-gf/EO-Point-Cloud.git
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

3. Run the main.py either at the DeltaTwin folder or EO-PointCloud folder. :
```bash
python main.py your_cdse_id your_cdse_access_key your_earth_data_hub_api_key bbox sampled_fraction
```

[DeltaTwin](figs/20250828-1413-50.2334437.mp4)

# How to create a DeltaTwin component 

For a detailed explanation, see at: [DeltaTwin Doc](https://deltatwin.destine.eu/docs/introduction)

### Deltatwin

1. Install deltatwin
```
pip install deltatwin
```
2. Log in 
```
deltatwin login username password -a https://api.deltatwin.destine.eu/
```
3. Refresh token
```
deltatwin login
```
4. Goes to the Folder [Deltatwin](/DeltaTwin/), fill the [inputs_file](/DeltaTwin/inputs_file.json) and start by running locally
```
deltatwin run start_local -i inputs_file.json --debug 
```
If you get an error, it may be because the name in the manifest: "name": "eo-point-cloud-generator", should be unique! So you have to create your own name.

5. Push the Component to the Cloud and the Web-UI
```
deltatwin component publish -t whatever-tag 0.0.1 
```

[DeltaTwin](figs/eo-workflow.jpg)

## Repository Structure

```bash
pcddeltatwin/
├─ DeltaTwin/  ## DELTATWIN COMPONENT OF EO-POINTCLOUD
│  ├─ models/
│  │  ├─ find_bug.ipynb
│  │  └─ main.py
│  ├─ inputs_file.json
│  ├─ manifest.json
│  └─ workflow.yml
├─ EO-Point-Cloud/  ##EO-POINTCLOUD 
│  ├─ src/
│  │  ├─ query_config.yml
│  │  ├─ s3_bucket.py
│  │  ├─ stac.py
│  │  ├─ utils_no2.py
│  │  └─ utils.py
│  ├─ debug_.py
│  ├─ main.py
│  ├─ test_apt_requirements.py
│  └─ tryer.py
├─ initial_setup/
│  └─ install_packages.sh  ## Information on how to properly install OPEN3D and all libraries of EO-PointCloud generation
├─ notebooks/
│  └─ draw_bbox.ipynb ## Quick way to create a bbox and get its boundaries to pass inside the EO-PointCloud arguments
├─ save/
│  ├─ dockerfile  ##Docker with OPEN3D
│  ├─ manifest_bug.json ## Save copy of manifest to workaround with likely bug regarding DeltaTwin and OPEN3D
│  ├─ manifest_test2.json ## Save copy
│  ├─ requirements.txt
│  ├─ t-requirements.txt
├  ├─ copy_manifest.json ## same copy
├  ├─ dockerfile.open3d 
│  └─ test.ipynb  ## Run tests on STAC Catalogue
├─ .gitignore
├─ LICENSE
├─ manifest.json
├─ readme.md
├─ requirements.txt
```

## Main.py
You can run any main.py in your environment to test it before trying out the DeltaTwin. 
Any file [main.py](/DeltaTwin/models/main.py) requires 5 inputs to run.
```
main.py -your_cdse_key -your_cdse_secret -your_earth_hub_api_key -bbox -sample_fraction
```

## Utility Functions 

The repository includes several utility classes to streamline data processing:

- `Sentinel2Reader`: For reading and preprocessing Sentinel-2 imagery
- `PcdGenerator`: For creating point clouds from DEM and imagery data
- `PointCloudHandler`: For manipulating and saving point clouds

[fig-eo](figs/tyrol.jpg)

## License

Apache License 2.0

## Acknowledgments
This dataset is a personal project developed during my internship under Destination Earth initiative within [ESA](https://www.esa.int/). 
 The majority part of the classes for creation and handling PointClouds were developt by Sebastien Tetauld. 
- [Sebastien Tetauld](https://github.com/sebastien-tetaud)
- [Copernicus DEM](https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model)
- [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)
- [DestinE Earth Data Hub](https://earthdatahub.destine.eu/)