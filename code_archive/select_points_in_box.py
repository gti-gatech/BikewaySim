# -*- coding: utf-8 -*-
"""
Created on Tue May 18 19:26:59 2021

@author: tpassmore6
"""

import geopandas as gpd
import contextily as ctx
import pandas as pd
import os


#change this to where you stored this folder
os.chdir(r'C:\Users\tpassmore6\Documents\GitHub\BikewaySim_Network_Processing')


minx = 2231492.083
miny = 1374382.322

maxx = 2234918.803
maxy = 1375960.721

bbox = [minx, miny, maxx, maxy]

nodes = gpd.read_file(
    r'Base_Shapefiles/arc/ABM2020-TIP20-2020-150kShapefiles-outputs.gdb',
    layer= 'DAILY_Node',
    bbox=bbox)

nodes.to_file(r'C:\Users\tpassmore6\OneDrive - Georgia Institute of Technology\BikewaySim\BikewaySim Slides\abm_examine\abm_nodes.geojson', drive='GeoJSON')

