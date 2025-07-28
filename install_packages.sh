
# Create a new conda environment with Python 3.13.1
# conda create --name env python=3.13.1 -y

# Install xarray with complete dependencies
python -m pip install "xarray[complete]"

# Install GDAL from conda-forge
conda install -c conda-forge gdal -y

# Install PDAL from conda-forge
conda install -c conda-forge python-pdal -y

# Install Open3D from conda-forge
conda install -c conda-forge open3d -y

pip install -r requirements.txt

echo "Installation complete."