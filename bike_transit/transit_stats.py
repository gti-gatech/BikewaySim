# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 02:22:20 2023

@author: tpassmore6
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd
import pickle
from tqdm import tqdm

#%% distribution of times


walk = gpd.read_file(r"C:/Users/tpassmore6/Documents/bad_transfers/TransitSimData/Data/walk_dist/visuals/553.gpkg",layer='centroids_viz')
bike = gpd.read_file(r"C:/Users/tpassmore6/Documents/bad_transfers/TransitSimData/Data/bike_dist/visuals/553.gpkg",layer='centroids_viz')
bikewalk = gpd.read_file(r"C:/Users/tpassmore6/Documents/bad_transfers/TransitSimData/Data/bikewalk_dist/visuals/553.gpkg",layer='centroids_viz')

walk['time_bin'].value_counts()
bikewalk['time_bin'].value_counts()
bike['time_bin'].value_counts()

#%%

bike = pd.read_parquet(r"C:\Users\tpassmore6\Documents\TransitSimData\Data\bike_dist\trips\553.parquet")
stops = gpd.read_file(r"C:/Users/tpassmore6/Documents/TransitSimData/Data/base_layers.gpkg",layer='route_and_stop')

#%%

bike = pd.read_pickle(r"C:\Users\tpassmore6\Documents\TransitSimData\Data\bike_dist\raptor_results\1071\8_15.pkl")
walk = pd.read_pickle(r"C:\Users\tpassmore6\Documents\TransitSimData\Data\walk_dist\raptor_results\1071\8_15.pkl")

bike = bike.loc[bike[bike['dest_taz']=='461']['travel_time'].idxmin()]
walk = walk.loc[walk[walk['dest_taz']=='461']['travel_time'].idxmin()]

#%%
bike = pd.read_pickle(r"C:\Users\tpassmore6\Documents\TransitSimData\Data\bike_dist\raptor_results\1071\8_0.pkl")
walk = pd.read_pickle(r"C:\Users\tpassmore6\Documents\TransitSimData\Data\walk_dist\raptor_results\1071\8_0.pkl")

bike = bike.loc[bike[bike['dest_taz']=='1033']['travel_time'].idxmin()]
walk = walk.loc[walk[walk['dest_taz']=='1033']['travel_time'].idxmin()]

#%%


walk = walk[walk['status']=='success']

remove_buses = walk[walk.apply(lambda row: len([x[-2] for x in row['edge_list'] if x[-2] == 'bus']) <= 1 ,axis=1)]
                          
                          )



#%%
taz = '1377'
walkraw = gpd.read_file(Path.home() / f'Documents/TransitSimData/Data/walk_dist/visuals/{taz}.gpkg',layer='tazs_viz',ignore_geometry=True)
bikewalkraw = gpd.read_file(Path.home() / f'Documents/TransitSimData/Data/bikewalk_dist/visuals/{taz}.gpkg',layer='tazs_viz',ignore_geometry=True)
bikeraw = gpd.read_file(Path.home() / f'Documents/TransitSimData/Data/bike_dist/visuals/{taz}.gpkg',layer='tazs_viz',ignore_geometry=True)

#%% check difference in number of accessible tazs
walkraw.shape[0]
bikewalkraw.shape[0]
bikeraw.shape[0]

#wait time
walkraw['avg_wait_time'].mean()
bikewalkraw['avg_wait_time'].mean()
bikeraw['avg_wait_time'].mean()

#%%find all accessible
access = walkraw.loc[walkraw['OBJECTID'].isin(bikeraw['OBJECTID']) & walkraw['OBJECTID'].isin(bikewalkraw['OBJECTID']),'OBJECTID'].tolist()

#filter each one
bike = bikeraw[bikeraw['OBJECTID'].isin(access)]
bikewalk = bikewalkraw[bikewalkraw['OBJECTID'].isin(access)]
walk = walkraw[walkraw['OBJECTID'].isin(access)]

#make index
bike.index = bike['OBJECTID']
bikewalk.index = bikewalk['OBJECTID']
walk.index = walk['OBJECTID']

#%% check difference in travel time
(walk['avg_transit_time'] - bikewalk['avg_transit_time']).mean()
(walk['avg_transit_time'] - bike['avg_transit_time']).mean()


#wait time
(walk['avg_wait_time'] - bikewalk['avg_wait_time']).mean()
(walk['avg_wait_time'] - bike['avg_wait_time']).mean()


#transfer eliminated
(((walk['min_transfers']==1).sum() - (bikewalk['min_transfers']==1).sum()) / (walk['min_transfers']==1).sum()) *100
(((walk['min_transfers']==1).sum() - (bike['min_transfers']==1).sum()) / (walk['min_transfers']==1).sum()) *100


#%% bike and bike-walk accessible

access = bikeraw.loc[bikeraw['OBJECTID'].isin(bikewalkraw['OBJECTID']),'OBJECTID'].tolist()

#filter each one
bike = bikeraw[bikeraw['OBJECTID'].isin(access)]
bikewalk = bikewalkraw[bikewalkraw['OBJECTID'].isin(access)]

#make index
bike.index = bike['OBJECTID']
bikewalk.index = bikewalk['OBJECTID']

#%% check difference in travel time
(bikewalk['avg_transit_time'] - bike['avg_transit_time']).mean()

#wait time
(bikewalk['avg_wait_time'] - bike['avg_wait_time']).mean()

#transfer eliminated
(((bikewalk['min_transfers']==1).sum() - (bike['min_transfers']==1).sum()) / (bikewalk['min_transfers']==1).sum()) *100



#%% differenc in 




tazs = gpd.read_file(Path.home() / 'Documents/TransitSimData/Data/base_layers.gpkg',layer='tazs')
centroids = gpd.read_file(Path.home() / 'Documents/TransitSimData/Data/base_layers.gpkg',layer='centroids')        
        
#%%

bike = pd.read_pickle(Path.home() / 'Documents/TransitSimData/Data/bike_dist/raptor_results/1071/8_45.pkl')
bikewalk = pd.read_pickle(Path.home() / 'Documents/TransitSimData/Data/bikewalk_dist/raptor_results/1071/8_0.pkl')
walk = pd.read_pickle(Path.home() / 'Documents/TransitSimData/Data/walk_dist/raptor_results/1071/8_0.pkl')


#%%

bike['spd'] = bike['dist_last_leg'] / bike['dist_from_time'].apply(lambda x: x.total_seconds()) * 60 *60 / 5280
bikewalk['spd'] = bikewalk['dist_last_leg'] / bikewalk['dist_from_time'].apply(lambda x: x.total_seconds()) * 60 *60 / 5280


#%%

test1 = bikewalk.loc[bikewalk.loc[bikewalk['dest_taz']=='420','travel_time'].idxmin()]
test2 = bike.loc[bike.loc[bike['dest_taz']=='420','travel_time'].idxmin()]


#%%

bike = pd.read_parquet(Path.home() / 'Documents/TransitSimData/Data/bike_dist/trips/553.parquet')