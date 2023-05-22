# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 16:03:55 2022

@author: tpassmore6
"""

import pandas as pd
import geopandas as gpd
import os
from pathlib import Path

#make directory/pathing more intuitive later
user_directory = os.fspath(Path.home()) #get home directory and convert to path string
file_directory = r"/Documents/BikewaySimData" #directory of bikewaysim outputs
os.chdir(user_directory+file_directory)

#import lines
dist_lines = gpd.read_file(r'trb2023/trip_lines.gpkg',layer='dist',driver='GPKG')
per_dist_lines = gpd.read_file(r'trb2023/trip_lines.gpkg',layer='per_dist',driver='GPKG')
imp_dist_lines = gpd.read_file(r'trb2023/trip_lines.gpkg',layer='imp_dist',driver='GPKG')

#only trips greater than a mile
dist_lines = dist_lines[dist_lines.length > 5280]
per_dist_lines = per_dist_lines[per_dist_lines.length > 5280]
imp_dist_lines = imp_dist_lines[per_dist_lines.length > 5280]

#selected trips
trip_ids=['529_512','542_550','551_544']


for trip_id in trip_ids:

    #filter
    selected_dist = dist_lines[dist_lines['trip_id'] == trip_id]
    selected_per_dist = per_dist_lines[per_dist_lines['trip_id'] == trip_id]
    selected_imp_dist = imp_dist_lines[imp_dist_lines['trip_id'] == trip_id]

    #export 
    fp = r'trb2023/routes.gpkg'
    selected_dist.to_file(fp,layer=trip_id+'dist')
    selected_per_dist.to_file(fp,layer=trip_id+'per_dist')
    selected_imp_dist.to_file(fp,layer=trip_id+'imp_dist')
