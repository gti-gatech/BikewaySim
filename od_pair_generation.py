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
file_directory = "\Documents\BikewaySimData" #directory of bikewaysim network processing code

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


#%%method 2, TAZ centroids, deprecated
#abm_nodes = gpd.read_file(r'processed_shapefiles/abm/abm_bikewaysim_base_nodes.geojson')

# bikewaysim_studyarea = gpd.read_file(r'processed_shapefiles/study_areas/study_area.geojson').to_crs('epsg:4326')

# tazs = gpd.read_file(r'base_shapefiles/arc/Model_Traffic_Analysis_Zones_2020/Model_Traffic_Analysis_Zones_2020.shp',
#                      mask=bikewaysim_studyarea)



#tazs.to_file('tu_delft/tazs.gpkg',driver='GPKG',layer='tazs')

tazs = gpd.read_file('trb2023/tazs.gpkg',layer='centroids') 

#get centroid
#tazs['geometry'] = tazs.geometry.centroid


tazs['FID_1'] = tazs['FID_1'].astype(str)

#export centroids
#tazs.to_file('tu_delft/tazs.gpkg',driver='GPKG',layer='centroids')


#%%
#create list of od pairs
list_of_ids = tazs['FID_1'].tolist()

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
od_pairs = pd.merge(od_pairs,tazs[['FID_1','geometry']], left_on = 'ori_id', right_on = 'FID_1', how='left')
od_pairs = od_pairs.rename(columns={'geometry':'ori_geometry'}).drop(columns={'FID_1'})

od_pairs = pd.merge(od_pairs,tazs[['FID_1','geometry']], left_on = 'dest_id', right_on = 'FID_1', how='left')
od_pairs = od_pairs.rename(columns={'geometry':'dest_geometry'}).drop(columns={'FID_1'})

od_pairs = od_pairs.set_geometry('ori_geometry').to_crs(epsg=4326)
od_pairs['ori_lat'] = od_pairs.geometry.y
od_pairs['ori_lon'] = od_pairs.geometry.x

od_pairs = od_pairs.set_geometry('dest_geometry').to_crs(epsg=4326)
od_pairs['dest_lat'] = od_pairs.geometry.y
od_pairs['dest_lon'] = od_pairs.geometry.x

od_pairs['trip_id'] = od_pairs['ori_id'] + '_' + od_pairs['dest_id']
od_pairs = od_pairs[['trip_id','ori_id','dest_id','ori_lat','ori_lon','dest_lat','dest_lon']]

#export
od_pairs.to_csv('bikewaysim_outputs/samples_in/all_tazs.csv', index = False)


#%% alt method 2

#tazs = gpd.read_file('demonstration_viz/tazs.geojson', ignore_geometry=True)

#get rid of excess columns


#%% create OD folders

#export         
#od_pairs.to_csv('od_pairs/od_pairs.csv', index = False)

#%% for viz

#for aggregate impedance
#single_taz_to_all = od_pairs[od_pairs['ori_id'] == '538']
#
#export
#single_taz_to_all.to_csv('bikewaysim_outputs/samples_in/single_taz_to_all.csv', index = False)
