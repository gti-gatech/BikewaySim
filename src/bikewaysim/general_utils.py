# General utility functions that can be use across the project
from shapely.geometry import Polygon, box
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
from importlib import import_module

# TODO throw this in utils
def run_notebook(notebook_path, timeout=600, kernel_name='python3'):
    """Runs a Jupyter Notebook without saving the output."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    
    executor = ExecutePreprocessor(timeout=timeout, kernel_name=kernel_name)
    executor.preprocess(notebook, {'metadata': {'path': os.path.dirname(notebook_path)}})
    
    # Optionally save the executed notebook
    with open(notebook_path, "w") as f:
        nbformat.write(notebook, f)

    print(f"Executed {os.path.basename(notebook_path)}")

def run_notebooks_in_order(notebook_list,dir):
    """Runs a list of Jupyter notebooks or Python scripts sequentially."""
    for notebook in notebook_list:
        if notebook.endswith('.ipynb'):
            # If it's a notebook, run it
            run_notebook(os.path.join(dir, notebook))
        elif notebook.endswith('.py'):
            # If it's a Python script, import and run it
            module_name = notebook[:-3]
            import_module(module_name)
            print(f"Executed {module_name}.py")

def print_elapsed_time(seconds):
    # Round the total seconds at the start
    seconds = round(seconds)
    
    # Calculate the elapsed days, hours, and minutes
    days = seconds // 86400  # 1 day = 86400 seconds
    hours = (seconds % 86400) // 3600  # 1 hour = 3600 seconds
    minutes = (seconds % 3600) // 60

    # Build the time string
    if days > 0:
        elapsed_time = f"{days:02} days {hours:02} hours {minutes:02} minutes"
    else:
        elapsed_time = f"{hours:02} hours {minutes:02} minutes"

    # Return the formatted elapsed time
    return elapsed_time

def chunks(l,n):
    '''
    Splits a list into groups of n for parallel processing operations
    '''
    n = max(1,n)
    return (l[i:i+n] for i in range(0,len(l),n))

def bbox_to_gdf(input:list):
    """
    Converts a bounding box in GeoJSON format from https://boundingbox.klokantech.com
    """
    
    input = input[0]
    polygon = Polygon([tuple(i) for i in input])
    gdf = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon])

    # [[[-85.0782381338,32.3978375774],[-84.9093067807,32.3978375774],[-84.9093067807,32.5314503075],[-85.0782381338,32.5314503075],[-85.0782381338,32.3978375774]]]
    
    return gdf

def bounds_to_polygon(gdf):
    """
    Converts the total bounds of a GeoDataFrame into a Polygon.
    
    Parameters:
    gdf (GeoDataFrame): The GeoDataFrame whose total bounds will be converted.
    
    Returns:
    Polygon: A Polygon representing the total bounds of the GeoDataFrame.
    """
    minx, miny, maxx, maxy = gdf.total_bounds
    return Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny), (minx, miny)])

def ckdnearest(gdA, gdB, return_dist=True):  
    """
    Take in two geometry columns and find nearest gdB point from each
    point in gdA. Returns the matching distance too.
    
    MUST BE PROJECTED COORDINATE SYSTEM
    """
    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].reset_index(drop=True)
    
    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest,
            pd.Series(dist, name='dist')
        ], 
        axis=1)
    
    if return_dist == False:
        gdf = gdf.drop(columns=['dist'])
    
    return gdf

def create_grid(gdf,cell_size_mi):
    '''
    Turns a gdf into a grid. Assumes a projected coordinate system in feet.
    '''
    
    cell_size_ft = cell_size_mi * 5280
    xmin, ymin, xmax, ymax = gdf.total_bounds
    grid_cells = []

    for x0 in np.arange(xmin, xmax+cell_size_ft, cell_size_ft):
        for y0 in np.arange(ymin, ymax+cell_size_ft, cell_size_ft):
            x1 = x0 - cell_size_ft
            y1 = y0 + cell_size_ft
            grid_cells.append(box(x0,y0,x1,y1))
    cell = gpd.GeoDataFrame(grid_cells,columns=['geometry'],crs=gdf.crs)
    return cell