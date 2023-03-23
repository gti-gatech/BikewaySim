# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 15:29:50 2021

@author: tpassmore6
"""
import geopandas as gpd
import pandas as pd
import os
from pathlib import Path

#make directory/pathing more intuitive later
user_directory = os.fspath(Path.home()) #get home directory and convert to path string
file_directory = "\Documents\GitHub\BikewaySim_Network_Processing" #directory of bikewaysim network processing code

#change this to where you stored this folder
os.chdir(user_directory+file_directory)

#study area to clip with
gdf_clip = gpd.read_file(r"Base_Shapefiles/bikewaysim_study_area/bikewaysim_study_area.shp").to_crs(epsg=2240)

#%%city of atlanta bike lanes

gdf_coa = gpd.read_file(r"Base_Shapefiles/coa/bicycle_routes.shp")

shared_coa = [
    'Enhanced shared Roadway',
    'Enhanced Shared Roadway',
    'Neghborhood Greenway',
    'Neighborhood Greenway',
    'Neighborhood greenway',
    'Sharrows'
    ]
bike_lanes_coa = [
    'Bike Lane',
    'Buffered Bike Lane',
    'Buffered Bike Lane / Bike Lane', 
    'Buffered Bike Lanes',
    'Buffered Contraflow Bike Lane',
    'Uphill Bike Lane / Downhill Sharrows',
    'Uphill Bike Lane | Downhill Sharrows',
    'Uphill bike lane downhill shared lane markings',
    'Uphill bike lane Downhill Sharrows',
    'Uphill Bike Lane, Downhill SLMS',
    'Uphill Bike Lanes / Downhill Sharrows',
    'Uphill Buffered Bike Lane | Downhill Sharrows',
    'Uphill Buffered Bike Lanes | Downhill Sharrows'
    ]

seperated_bike_lanes_coa = [
    'Protected Bike Lane',
    'Raised Bike Lane',
    'Seperated Bike Lane',
    'Seperated Bike Lanes',
    'Seperated WB Lane/Buffered EB Lane',
    'Two Way Cycle Track'
    ]

bike_paths_coa = [
    'Curbless Shared Bike/Ped Street (no cars)',
    'Hard Surface Multi-Use Path',
    'Multi-Use Path',
    'Shared Use Path',
]

gdf_coa['facility_type_simplified'] = None

#shared facility
gdf_coa.loc[
    (gdf_coa['FACILITY_T'].isin(shared_coa)), 'facility_type_simplified'] = "Shared Facility"

#bike lane
gdf_coa.loc[
   (gdf_coa['FACILITY_T'].isin(bike_lanes_coa)), 'facility_type_simplified'] = "Bike Lane"

#seperated bike lane
gdf_coa.loc[
   (gdf_coa['FACILITY_T'].isin(seperated_bike_lanes_coa)), 'facility_type_simplified'] = "Protected Bike Lane"

#multi-use path
gdf_coa.loc[
   (gdf_coa['FACILITY_T'].isin(bike_paths_coa)), 'facility_type_simplified'] = "Multi-Use Path"

#filter out to only currently built
gdf_coa = gdf_coa[gdf_coa['STATUS'] == "Existing"]

# Clip by study area boundary
gdf_coa = gpd.clip(gdf_coa,gdf_clip)

# export
gdf_coa = gdf_coa.to_file(r'Processed_Shapefiles/bike_inventories/coa_bike_cleaned.geojson', driver = 'GeoJSON')

#%% ARC study area

gdf_arc = gpd.read_file(r'Base_Shapefiles/arc/Regional_Bikeway_Inventory_2021.geojson').to_crs(epsg=2240)

gdf_arc = gpd.clip(gdf_arc, gdf_clip)

gdf_arc.to_file(r'Processed_Shapefiles/bike_inventories/arc_bike_cleaned.geojson', driver = 'GeoJSON')

#%% OSM bikes, not sure what do with this one yet

gdf_osm = gpd.read_file(r'Base_Shapefiles/osm/osm_links_arc.geojson').to_crs(epsg=2240)

gdf_osm = gdf_osm[gdf_osm.geometry.type == "LineString"]


gdf_osm = gpd.clip(gdf_osm,gdf_clip)


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

#%% Analysis

arc_length = round(gdf_arc.length.sum() / 5280, 0)

osm_length = round(osm_bike_facilities.length.sum() / 5280,0)

diff = osm_length/arc_length

print(f'ARC has {arc_length} miles and OSM has {osm_length} miles. Matchup percentage of {diff}%')

bike_lanes = round(gdf_arc[gdf_arc['facil'] == "Bike Lane"].length.sum() / 5280,0)
multi_use = round(gdf_arc[gdf_arc['facil'] == "Mixed Use Path"].length.sum() / 5280,0)
protec_lane = round(gdf_arc[gdf_arc['facil'] == "Protected Bike Lane"].length.sum() / 5280,0)

print(f'There are {bike_lanes} miles of bike lane, {multi_use} miles of mixed use paths, and {protec_lane} miles of protected bike lanes.')