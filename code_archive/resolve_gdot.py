# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:45:48 2021

@author: tpassmore6
"""

import geopandas as gpd
import pandas as pd
import fiona
import os
from functools import partial, reduce                                                                              
import time

start_time = time.time()                                                                                                              
        
#change this to where you stored this folder
os.chdir(r'C:\Users\tpassmore6\Documents\GitHub\BikewaySim_Network_Processing')

#fields to ignore
ignore = ['DATA_ITEM','Shape_Length','BEG_POINT','END_POINT','MILEAGE']

#rc routes filepath
rc_routesfp = r'Base_Shapefiles/gdot/Road_Inventory_Geodatabase/Road_Inventory_2019.gdb'
data_frames = [gpd.read_file(rc_routesfp, layer = x, ignore_fields = ignore) for x in range(0,len(fiona.listlayers(rc_routesfp)))]
print(f'done... now merging...')    

rc_routes = reduce(lambda left,right: pd.merge(left,right,how='outer'), data_frames)
print('done... now writing...')


print(rc_routes['ROUTE_ID'].nunique())

rc_routes.to_file(r'Base_Shapefiles\gdot\rc_routes.geojson', driver = 'GeoJSON')
print(f'Total Time: {round(time.time() - start_time, 2)} seconds')
