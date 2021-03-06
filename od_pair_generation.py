# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 14:41:04 2021

@author: tpassmore6
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
from pathlib import Path

#make directory/pathing more intuitive later
user_directory = os.fspath(Path.home()) #get home directory and convert to path string
file_directory = "\Documents\GitHub\BikewaySim_Network_Processing" #directory of bikewaysim network processing code

#change this to where you stored this folder
os.chdir(user_directory+file_directory)

#%% Method 1 random OD pairs
def get_random_OD_pairs(poly, num_of_pairs):
    
     #simplify gdf to poly
     poly_simp = poly.geometry[0]
     minx, miny, maxx, maxy = poly_simp.bounds #get the min and max extents of the polygon
     
     #initialize empty lists to contain od pairs
     trip_id = []
     origin_geo = []
     dest_geo = []
     
     x = 1
     while x < (num_of_pairs+1):
         y = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
         z = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
         if poly_simp.contains(y) and poly_simp.contains(z):
             trip_id.append(x)
             origin_geo.append(y)
             dest_geo.append(z)
             x += 1
         else:
             x = x
             
     #turn lists into GeoSeries
     origin_geo = gpd.GeoSeries(origin_geo)
     dest_geo = gpd.GeoSeries(dest_geo)
     
     points = pd.DataFrame({'trip_id':trip_id,'ori_lat':origin_geo.geometry.y,'ori_lon':origin_geo.x,
                                'dest_lat':dest_geo.y,'dest_lon':dest_geo.x, 'origin_geo':origin_geo})
     
     #export         
     points.to_csv('od_pairs/od_pairs.csv')
     
     #plotting
     fig, ax = plt.subplots()

     ax.set_aspect('equal')

     poly.plot(ax=ax, color ='white', edgecolor='black')
     origin_geo.plot(ax=ax, marker='o', color='red', markersize=5)
     dest_geo.plot(ax=ax, marker='o', color='green', markersize=5)
     plt.show();
     
     return points

#study_area = gpd.read_file(r'Processed_Shapefiles/study_areas/study_area.shp').to_crs(epsg=4326)

#od_pairs = get_random_OD_pairs(study_area,1000)


#%%method 2, TAZ centroids
abm_nodes = gpd.read_file(r'Processed_Shapefiles/abm/abm_study_area_base_nodes.geojson')
taz = gpd.read_file(r'Base_Shapefiles/arc/Model_Traffic_Analysis_Zones_2020/Model_Traffic_Analysis_Zones_2020.shp')

trim_node_id = lambda row: row['abm_ID'].split("100")[1]
abm_nodes['abm_ID'] = abm_nodes.apply(trim_node_id, axis = 1)
            
taz['FID_1'] = taz['FID_1'].astype(str)
            
taz_centroids = pd.merge(abm_nodes, taz['FID_1'], left_on = 'abm_ID', right_on = 'FID_1')


#create list of od pairs
list_of_ids = taz_centroids['abm_ID'].tolist()

ori_id = []
dest_id = []

#https://stackoverflow.com/questions/28757389/pandas-loc-vs-iloc-vs-at-vs-iat
for x in list_of_ids:
    for y in list_of_ids:
        if y != x:
            ori_id.append(x)
            dest_id.append(y)
od_pairs = pd.DataFrame({'ori_id':ori_id,'dest_id':dest_id})          
            
# merge geometry info back in and create other columns
od_pairs = pd.merge(od_pairs,taz_centroids[['abm_ID','geometry']], left_on = 'ori_id', right_on = 'abm_ID', how='left')
od_pairs = od_pairs.rename(columns={'geometry':'ori_geometry'}).drop(columns={'abm_ID'})

od_pairs = pd.merge(od_pairs,taz_centroids[['abm_ID','geometry']], left_on = 'dest_id', right_on = 'abm_ID', how='left')
od_pairs = od_pairs.rename(columns={'geometry':'dest_geometry'}).drop(columns={'abm_ID'})

od_pairs = od_pairs.set_geometry('ori_geometry').to_crs(epsg=4326)
od_pairs['ori_lat'] = od_pairs.geometry.y
od_pairs['ori_lon'] = od_pairs.geometry.x

od_pairs = od_pairs.set_geometry('dest_geometry').to_crs(epsg=4326)
od_pairs['dest_lat'] = od_pairs.geometry.y
od_pairs['dest_lon'] = od_pairs.geometry.x

od_pairs['trip_id'] = od_pairs['ori_id'] + '_' + od_pairs['dest_id']
od_pairs = od_pairs[['trip_id','ori_lat','ori_lon','dest_lat','dest_lon']]

#export         
od_pairs.to_csv('od_pairs/od_pairs.csv', index = False)

