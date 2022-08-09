# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 21:19:26 2022

@author: tpassmore6
"""
import geopandas as gpd
import pandas as pd
import numpy as np
import os
from pathlib import Path

#make directory/pathing more intuitive later
user_directory = os.fspath(Path.home()) #get home directory and convert to path string
file_directory = r"/Documents/BikewaySimData" #directory of bikewaysim outputs
os.chdir(user_directory+file_directory)

#%%

#bring in conflated network
conflated_links = gpd.read_file('processed_shapefiles/conflation/finalized_networks/trb.gpkg',layer='links')


#bring in osm data
osm = 

#filter
osm = osm[['']]


check_if_null = ["cycleway", "cycleway:left", "cycleway:right", "cycleway:right:width", "cycleway:buffer", "cycleway:left:buffer",
                 "cycleway:right:buffer", "cycleway:both:buffer"]
filter_columns = []

for column_names in check_if_null:
    if column_names in gdf_osm.columns:
        filter_columns.append(column_names)

#main filter
osm_bike_facilities_1 = (gdf_osm['highway'] == "cycleway") | (gdf_osm['highway_1'] == "cycleway")

osm_bike_facilities_2 = gdf_osm[filter_columns].isna().all(axis=1) == False

osm_bike_facilities_3 = (gdf_osm != "shared_lane").all(axis=1)

osm_bike_facilities = gdf_osm[ (osm_bike_facilities_1 | osm_bike_facilities_2) & osm_bike_facilities_3]


#osm_bike_facilities_2 = osm_bike_facilities.dropna(axis=0, how='all', subset = filter_columns)

#osm_bike_facilities = osm_bike_facilities_1.append(osm_bike_facilities_2)

gdf_osm = gdf_osm.filter(["highway","highway_1", "cycleway", "cycleway:left", "cycleway:right", "cycleway:right:width", "cycleway:buffer", "cycleway:left:buffer",
                 "cycleway:right:buffer", "cycleway:both:buffer", "bicycle","geometry"])

gdf_osm.to_file(r'Processed_Shapefiles/bike_inventories/osm_bike_examine.geojson', driver = 'GeoJSON')

osm_bike_facilities = osm_bike_facilities.filter(["highway","highway_1", "cycleway", "cycleway:left", "cycleway:right", "cycleway:right:width", "cycleway:buffer", "cycleway:left:buffer",
                 "cycleway:right:buffer", "cycleway:both:buffer", "bicycle","geometry"])

osm_bike_facilities.to_file(r'Processed_Shapefiles/bike_inventories/osm_bike_cleaned.geojson', driver = 'GeoJSON')