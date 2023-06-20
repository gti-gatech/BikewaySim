#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 13:09:28 2022

@author: tannerpassmore
"""

import os
from pathlib import Path
import pandas as pd
import geopandas as gpd


user_directory = os.fspath(Path.home()) #get home directory and convert to path string
file_directory = r"/Documents/BikewaySimData" #directory of bikewaysim network processing code

#change this to where you stored this folder
os.chdir(user_directory+file_directory)

#%% bring in links
links = gpd.read_file(rf'processed_shapefiles/prepared_network/dist/links/links.geojson')

#%% start from distance col

#modify primary links (with no bike lane) as 4x the distance if bikelane present just use regular distance
stress_links = ['primary','primary_link','secondary','secondary_link','trunk','trunk_link']
links['stress_cond'] = links['highway'].isin(stress_links)

#recalculate per distance
links['per_dist'] = links['distance'] * (1-0.5*links['bike_lane']+1*links['stress_cond'])

#%% make network improvment

#what road to improve?
#st_names = [['10th Street Northwest','10th Street Northeast'],['Piedmont Avenue Northeast'],['']]
st_names = ['Piedmont Avenue Northeast']


#make improvement
links.loc[links['name'].isin(st_names),'bike_lane'] = 1

#recalculate per distance
links['improvement'] = links['distance'] * (1-0.5*links['bike_lane']+1*links['stress_cond'])

#export improved links
links[links['improvement'] < links['per_dist']].to_file('trb2023/improvements.geojson',driver='GeoJSON')

#%% export this

links.to_file('trb2023/new_network.geojson',driver='GeoJSON')

