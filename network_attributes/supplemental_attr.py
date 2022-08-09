# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 04:52:07 2022

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

#%% add via overlap

def attr_overlap(base_links, join_links, buffer_ft):
    
    #give base_links a temp column so each row has unique identifier (index basically does this too)
    # A_B doesn't work because there might be duplicates from the split links step
    base_links['temp_ID'] = np.arange(base_links.shape[0]).astype(str)
    
    #calculate original base_links link length
    base_links['length'] = base_links.geometry.length
    
    #create copy of join links to use for bufferring
    buffered_links = join_links.copy()
    
    #buffer the join links by tolerance_ft
    buffered_links['geometry'] = buffered_links.buffer(buffer_ft)
    
    #make sure it's the active geometry
    buffered_links.set_geometry('geometry',inplace=True)
    
    #intersect join buffer and base links
    overlapping_links = gpd.overlay(base_links, buffered_links, how='intersection')
    
    #calculate the link length to see which join buffers had the greatest overlap of base links
    overlapping_links['percent_overlap'] = (overlapping_links.geometry.length / overlapping_links['length'] )
    
    #select matches with greatest link overlap
    max_overlap_idx = overlapping_links.groupby('temp_ID')['percent_overlap'].idxmax().to_list()
    
    #get matching links from overlapping links
    match_links = overlapping_links.loc[max_overlap_idx]
    
    #say link has to have at least this much overlap
    match_links = match_links[match_links['percent_overlap'] > 0.80]
    
    #filter
    match_links = match_links[['temp_ID','facil']]
    
    #join with base links
    base_links = pd.merge(base_links,match_links,on='temp_ID',how='left')

    #drop
    base_links.drop(columns=['temp_ID','length'],inplace=True)
    
    return base_links



#%%

#bring in conflated network
#conflated_links = gpd.read_file('processed_shapefiles/conflation/finalized_networks/trb.gpkg',layer='links')
conflated_links = gpd.read_file('processed_shapefiles/conflation/finalized_networks/here_trb.gpkg',layer='links')

arc = gpd.read_file('processed_shapefiles/bike_inventories/arc2022.gpkg')
arc = arc[['facil','geometry']]

#%%

conflated_links = attr_overlap(conflated_links,arc,40)

#%%

conflated_links['pbl']=0
conflated_links.loc[conflated_links['facil']=='Protected Bike Lane','pbl'] = 1

conflated_links['bl']=0
conflated_links.loc[conflated_links['facil']=='Bike Lane','bl'] = 1

#conflated_links['mu']=0
conflated_links.loc[(conflated_links['facil']=='Multi-Use Path'),'mu'] = 1

conflated_links.drop(columns=['facil'],inplace=True)

#%% export
conflated_links.to_file('processed_shapefiles/conflation/finalized_networks/here_trb.gpkg',layer='links')
#conflated_links.to_file('processed_shapefiles/conflation/finalized_networks/trb.gpkg',layer='links')