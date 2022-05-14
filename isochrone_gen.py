# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 17:37:50 2022

@author: tpassmore6
"""
#imports
import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd

#%%bring in network data

data = pd.read_pickle(r'C:/Users/tpassmore6/Documents/BikewaySimData/base_shapefiles/osm/osm_attributes_bikewaysim.pickle')

columns = ['osmid','highway','highway_1','bicycle',
            'footway','cycleway','cycleway:both',
            'cycleway:est_width','cycleway:left','cycleway:right'
            ]
data = data[columns]

links = gpd.read_file(r'C:/Users/tpassmore6/Documents/BikewaySimData/processed_shapefiles/prepared_network/links/links.geojson')
nodes = gpd.read_file(r'C:/Users/tpassmore6/Documents/BikewaySimData/processed_shapefiles/prepared_network/nodes/nodes.geojson')

links = links[['A','B','A_B','osmid','name','distance','geometry']]

links = pd.merge(links,data,on=['osmid']).drop_duplicates()

#%% find osm bike infra

check_if_null = ["cycleway", "cycleway:left", "cycleway:right", "cycleway:right:width", "cycleway:buffer", "cycleway:left:buffer",
                 "cycleway:right:buffer", "cycleway:both:buffer"]
filter_columns = []

for column_names in check_if_null:
    if column_names in links.columns:
        filter_columns.append(column_names)

#main filter
osm_bike_facilities_1 = (links['highway'] == "cycleway") | (links['highway_1'] == "cycleway")

osm_bike_facilities_2 = links[filter_columns].isna().all(axis=1) == False

osm_bike_facilities_3 = (links != "shared_lane").all(axis=1)

osm_bike_facilities = links[ (osm_bike_facilities_1 | osm_bike_facilities_2) & osm_bike_facilities_3]

osm_bike_facilities.to_file(r'C:/Users/tpassmore6/Downloads/osm_bike.geojson', driver = 'GeoJSON')

#bike lane/yes no
links['bike_lane'] = 0
links.loc[links['A_B'].isin(osm_bike_facilities['A_B']),'bike_lane'] = 1


#%% percieved cost
# create percieved link
links['per_distance'] = links['distance']

#modify bike lanes (perceive as half the distance)
links.loc[links['bike_lane']==1,'per_distance'] = links['per_distance'] * 0.5

#modify primary links (with no bike lane) as 4x the distance if bikelane present just use regular distance
stressful_links = ['primary','primary_link','secondary','secondary_link','trunk','trunk_link']
stressful_cond = links['highway'].isin(stressful_links)


links.loc[(links['bike_lane']==0) & stressful_cond,'per_distance'] = links['distance'] * 4
links.loc[(links['bike_lane']==1) & stressful_cond,'per_distance'] = links['distance']

#%%lts planning version
stressful_links = ['primary','primary_link','secondary','secondary_link']
condition = (links['bike_lane']==0) & links['highway'].isin(stressful_links)
links_lowstress = links[-condition]

#%% create the impedance columns

#%% export networks

links_per = links.copy()
links_per['distance'] = links_per['per_distance']
links_per.drop(columns=['per_distance'])
links_per[['A','B','A_B','osmid','name','distance','geometry']].to_file(r'C:/Users/tpassmore6/Documents/BikewaySimData/processed_shapefiles/prepared_network/links/links_per.geojson',driver='GeoJSON')

#%%

def make_bikeshed(links,nodes,n,time_min, impedance_col,bk_speed_mph):
    DGo = nx.DiGraph()  # create directed graph
    for ind, row2 in links.iterrows():
        # forward graph, time stored as minutes
        DGo.add_weighted_edges_from([(str(row2['A']), str(row2['B']), float(row2[impedance_col] / bk_speed_mph * 60))])
                                    #weight='forward', name=row2['name'])
        
    # for ind, row2 in links.iterrows():
    #     # forward graph, time stored as minutes
    #     DGo.add_weighted_edges_from([(str(row2['A']), str(row2['B']), float(row2[impedance_col] / bk_speed_mph * 60))])
    #                                 #weight='forward', name=row2['name'])    
    #create bikeshed
    #https://networkx.org/documentation/stable/reference/generated/networkx.generators.ego.ego_graph.html
    #https://geonetworkx.readthedocs.io/en/latest/5_Isochrones.html
    bikeshed = nx.ego_graph(DGo, radius=time_min, n=n, distance='weight')
            
    #merge
    links['A_B_tup'] = list(zip(links['A'],links['B']))

    #
    bikeshed = links[links['A_B_tup'].isin(list(bikeshed.edges))]
    bikeshed_node = nodes[nodes['N']==n]
    
    #drop tuple column
    bikeshed.drop(columns=['A_B_tup'],inplace=True)
    
    return bikeshed, bikeshed_node

#n = '3069614561'
#n = '3069238372'
#n = '307752789090'
n = '3069322975'

bikeshed, bikeshed_node = make_bikeshed(links,nodes,n,15,'distance',8)
bikeshed_per, _ = make_bikeshed(links,nodes,n,15,'per_distance',8)
bikeshed_lts, _ = make_bikeshed(links_lowstress,nodes,n,15,'distance',8)

#%%

#calculate lengths
bikeshed.length.sum() /5280 /2
bikeshed_per.length.sum()/5280 /2
bikeshed_lts.length.sum()/5280 /2

#export

#%% isochrone/ego-graph
import os
from pathlib import Path

user_directory = os.fspath(Path.home()) #get home directory and convert to path string
fp = user_directory+r'\Documents\BikewaySimData\tu_delft\bikeshed.gpkg'

bikeshed.to_file(fp,layer='bikeshed',driver='GPKG')
bikeshed_node.to_file(fp,layer='node',driver='GPKG')
bikeshed_per.to_file(fp,layer='bikeshed_per',driver='GPKG')
bikeshed_lts.to_file(fp,layer='lts',driver='GPKG')

