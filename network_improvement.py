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

dist = gpd.read_file(r'prepared_network/dist/links/links.geojson')
per = gpd.read_file(r'prepared_network/per_dist/links/links.geojson')

#%% drop duplicates?
dist = dist.drop_duplicates()
per = per.drop_duplicates()

#rename dist column in per
per = per.rename(columns={'distance':'per_distance'})

#merge by a_b
comb = pd.merge(dist,per[['A_B','per_distance']],on=['A_B'])

#recreate bike lane columns
comb['mod'] = 1
comb.loc[comb['distance'] > comb['per_distance'],'mod'] = 0.25
comb.loc[comb['distance'] < comb['per_distance'],'mod'] = 4

#put a bike lane on 10th
comb.loc[comb.name == '10th Street Northwest','mod'] = 0.25

#create new per_distance column
comb['per_distance'] = comb['distance'] * comb['mod']

#export
comb = comb[per.columns.to_list()]
comb = comb.rename(columns={'per_distance':'distance'})
comb.to_file(r'prepared_network/improvement/links/links.geojson')

#put a bike lane on boulevard


#%% check dist columns

