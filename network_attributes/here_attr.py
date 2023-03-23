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

#settings
studyarea_name = 'bikewaysim'
fp = Path.home() / Path(f'Documents/NewBikewaySim/{studyarea_name}')

#revelant attributes
here_attr = ['here_A_B','ST_NAME','SPEED_CAT','LANE_CAT','DIR_TRAVEL','FUNC_CLASS']

#import network
links = gpd.read_file(fp / Path('filtered_network.gpkg'),layer='here_roadbike_links')
nodes = gpd.read_file(fp / Path('filtered_network.gpkg'),layer='here_roadbike_links')

#bring in attribute data
attr = pd.read_pickle(fp / Path('here.pkl'))[here_attr]

#attach attribute data to filtered links
links = pd.merge(links,attr,on='here_A_B')

#%% speed categories
here_speed_bins = {
    '1': '> 30', # '> 80 MPH',
    '2': '> 30', # '65-80 MPH',
    '3': '> 30', # '55-64 MPH',
    '4': '> 30', # '41-54 MPH',
    '5': '> 30', # '31-40 MPH',
    '6': '25-30', # '21-30 MPH',
    '7': '< 25', # '6-20 MPH',
    '8': '< 25' # '< 6 MPH'
    }
links['SPEED_CAT'] = links['SPEED_CAT'].map(here_speed_bins)

#%% number of lanes
here_lane_bins = {
    '1': '1', # 'one lane',
    '2': '2-3', # 'two or three lanes',
    '3': '> 4' # 'four or more'
    }
links['LANE_CAT'] = links['LANE_CAT'].map(here_lane_bins)

#%% road directionality
here_oneway_bins = {
    'B':'both', # Both Directions
    'F':'oneway', # From Reference Node
    'T':'wrongway', # To Reference Node
    'N': 'NA' # Closed in both directions
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

#%% export for conflation
links.to_file(fp / Path('to_conflate.gpkg'),layer='here_links')
nodes.to_file(fp / Path('to_conflate.gpkg'),layer='here_nodes')