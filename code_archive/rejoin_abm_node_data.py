# -*- coding: utf-8 -*-
"""
Created on Wed May 19 09:34:06 2021

@author: tpassmore6
"""

import geopandas as gpd
import pandas as pd
import os

#change this to where you stored this folder
os.chdir(r'C:\Users\tpassmore6\Documents\GitHub\BikewaySim_Network_Processing')


nodes = gpd.read_file(r'Processed_Shapefiles/abm/abm_study_area_road_nodes.geojson')
nodes_complete = gpd.read_file(r'Base_Shapefiles/arc/ABM2020-TIP20-2020-150kShapefiles-outputs.gdb',
                             layer = 'DAILY_Node')

nodes_complete['match'] = nodes_complete['N'].astype(str) + '_abm'

nodes_w_info = pd.merge(nodes, nodes_complete, left_on='ID_abm', right_on='match')


#what columns are all zeros
zero_columns = nodes_w_info.loc[:,(nodes_w_info == 0).all(axis=0)]

zero_columns.head()

#column names all
list(nodes_w_info.columns)

#column names zero
list(zero_columns.columns)

#what columns aren't filled with only zeroes
nodes_with_info = nodes_w_info.loc[:,(nodes_w_info != 0).any(axis=0)]