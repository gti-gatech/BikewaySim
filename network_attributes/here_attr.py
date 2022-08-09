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

#%%

#bring in attribute data
here = gpd.read_file('processed_shapefiles/here/here_bikewaysim_network.gpkg',layer='base_links',ignore_geometry=False)

#filter to certain columns
here = here[['here_A_B','ST_NAME','SPEED_CAT','LANE_CAT','DIR_TRAVEL','AR_AUTO']]



#%% speed cat

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

here['below25'] = 0
bins = ['8','7']
here.loc[here['SPEED_CAT'].isin(bins),'below25'] = 1

here['25-30'] = 0 
bins = ['6']
here.loc[here['SPEED_CAT'].isin(bins),'25-30'] = 1

here['above30'] = 0
bins = ['1','2','3','4','5']
here.loc[here['SPEED_CAT'].isin(bins),'above30'] = 1


#%% lanes

# here_lane_bins = {
#     '1': 'one lane',
#     '2': 'two or three lanes',
#     '3': 'four or more'
#     }

here['1laneper'] = 0
here.loc[here['LANE_CAT']=='1','1laneper'] = 1

here['2to3lanesper'] = 0
here.loc[here['LANE_CAT']=='2','2to3lanesper'] = 1

here['4ormorelanesper'] = 0
here.loc[here['LANE_CAT']=='3','4ormorelanesper'] = 1

#%% directionality

# B	Both Directions
# F	From Reference Node
# T	To Reference Node
# N	Closed in both directions

here['oneway'] = 0
bins = ['F','T']
here.loc[here['DIR_TRAVEL'].isin(bins),'oneway'] = 1

here['wrongway'] = 0
bins = ['T']
here.loc[here['DIR_TRAVEL'].isin(bins),'wrongway'] = 1

#%% multi use paths

here['mu'] = 0
here.loc[here['AR_AUTO']=='N','mu'] = 1

#%% filter

here.drop(columns=['SPEED_CAT','LANE_CAT','DIR_TRAVEL'],inplace=True)

#%%


#bring in conflated network
#conflated_links = gpd.read_file('processed_shapefiles/conflation/finalized_networks/trb.gpkg',layer='links')
conflated_links = gpd.read_file('processed_shapefiles/here/here_bikewaysim_network.gpkg',layer='roadbike_links')

#merge
conflated_links = pd.merge(conflated_links,here,on='here_A_B',how='left')

#add arc bike attributes



#export
#conflated_links.to_file('processed_shapefiles/conflation/finalized_networks/trb.gpkg',layer='links')
conflated_links.to_file('processed_shapefiles/conflation/finalized_networks/here_trb.gpkg',layer='links')
