# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 16:34:15 2022

@author: tpassmore6
"""
#%%

import geopandas as gpd
import pandas as pd
import numpy as np
import os
from pathlib import Path

#make directory/pathing more intuitive later
user_directory = os.fspath(Path.home()) #get home directory and convert to path string
file_directory = r"/Documents/BikewaySimData" #directory of bikewaysim outputs
os.chdir(user_directory+file_directory)

#settings
desired_crs = 'epsg:2240'
here_attr = ['here_A_B','ST_NAME','SPEED_CAT','LANE_CAT','DIR_TRAVEL','FUNC_CLASS']
studyareaname = 'marta'
#%%

#bring in filtered links
filtered_links = gpd.read_file(f'processed_shapefiles/{studyareaname}/here_network.gpkg',layer='road_links')
filtered_nodes = gpd.read_file(f'processed_shapefiles/{studyareaname}/here_network.gpkg',layer='road_nodes')

#bring in attribute data
raw_links = gpd.read_file(f'processed_shapefiles/{studyareaname}/here_network.gpkg',layer='base_links',ignore_geometry=True,rows=0)
ignore_fields = [x for x in list(raw_links.columns) if x not in here_attr]
raw_links = gpd.read_file(f'processed_shapefiles/{studyareaname}/here_network.gpkg',layer='base_links',ignore_fields=ignore_fields,ignore_geometry=True)

#attach attribute data to filtered links
links = filtered_links.merge(raw_links,on='here_A_B')

#%% speed categories

#these are all the possible speeds
# here_speed_bins = {
#     '1': '> 80 MPH',
#     '2': '65-80 MPH',
#     '3': '55-64 MPH',
#     '4': '41-54 MPH',
#     '5': '31-40 MPH',
#     '6': '21-30 MPH',
#     '7': '6-20 MPH',
#     '8': '< 6 MPH'
#     }

#simplified speeds
here_speed_bins = {
    '1': '> 30',
    '2': '> 30',
    '3': '> 30',
    '4': '> 30',
    '5': '> 30',
    '6': '25-30',
    '7': '< 25',
    '8': '< 25'
    }

#replace speed categories
links['SPEED_CAT'] = links['SPEED_CAT'].map(here_speed_bins)

#%% lanes

# all possible lane combos
# here_lane_bins = {
#     '1': 'one lane',
#     '2': 'two or three lanes',
#     '3': 'four or more'
#     }

here_lane_bins = {
    '1': '1',
    '2': '2-3',
    '3': '> 4'
    }

links['LANE_CAT'] = links['LANE_CAT'].map(here_lane_bins)

#%% directionality

# all possible directions
# B	Both Directions
# F	From Reference Node
# T	To Reference Node
# N	Closed in both directions

here_oneway_bins = {
    'B':'both',
    'F':'oneway',
    'T':'wrongway',
    'N': 'NA'
    }

links['DIR_TRAVEL'] = links['DIR_TRAVEL'].map(here_oneway_bins) 

#%% here functional class (does not correspond to FHWA or HFCS)

func_class = {
    '1':'highways',
    '2':'major arterials',
    '3':'collectors/minor arterials',
    '4':'collectors/minor atrerials',
    '5':'local'
    }

links['FUNC_CLASS'] = links['FUNC_CLASS'].map(func_class)

#%% multi use paths

# multi_use_path = {
#     'N': True,
#     'Y': False
#     }

# here['mu'] = 0
# here.loc[here['AR_AUTO']=='N','mu'] = 1

#%%

#export for conflation
links.to_file(f'to_conflate/{studyareaname}.gpkg',layer='here_links')
filtered_nodes.to_file(f'to_conflate/{studyareaname}.gpkg',layer='here_nodes')
