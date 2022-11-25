# Simple Install
~~~
pip install https://github.com/thomaspingel/smrf/archive/refs/heads/main.zip
~~~

# Full Lidar processing environment
~~~
conda create --name lidar -c conda-forge geopandas laspy scipy jupyterlab spyder imageio rasterio python=3.9 python-pdal --yes
conda activate lidar
pip install laszip open3d pystac-client planetary_computer
pip install https://github.com/thomaspingel/smrf/archive/refs/heads/main.zip
~~~

# Examples
Examples for use at https://github.com/thomaspingel/smrf/tree/main/examples
