# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 12:14:15 2021

@author: tpassmore6
"""

import os
from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

#make directory/pathing more intuitive later
user_directory = os.fspath(Path.home()) #get home directory and convert to path string
file_directory = r"\Documents\GitHub\BikewaySim_Network_Processing" #directory of bikewaysim network processing code

#change this to where you stored this folder
os.chdir(user_directory+file_directory)

#used for masking
#gdf = gpd.read_file(r'Base_Shapefiles/bikewaysim_study_area/bikewaysim_study_area.shp').to_crs(epsg=2240)

#osm_links = gpd.read_file(r'Base_Shapefiles/osm/osm_links_arc.geojson').to_crs(epsg=2240)

osm_links = gpd.read_file(r'Processed_Shapefiles/osm/test.geojson').to_crs(epsg=2240)

#if there are features that enclose themselves (usually sidewalk blocs and circular paths in parks), convert them to linestring by taking the exterior
osm_polygons_to_links = osm_links[osm_links.geometry.type == "Polygon"].boundary.geometry

osm_links.loc[osm_polygons_to_links.index,'geometry'] = osm_polygons_to_links

osm_nodes = gpd.read_file(r'Base_Shapefiles/osm/osm_nodes_arc.geojson').to_crs(epsg=2240)


#osm_nodes.to_file(r'Base_Shapefiles/osm/osm_nodes_split.geojson', driver='GeoJSON')


#%% Extract Start and End Points Code from a LineString as tuples
def start_node_point(row, geom):
   return (Point(row[geom].coords.xy[0][0], row[geom].coords.xy[1][0])) #basically look at x and then y coord, use apply to do this for every row of a dataframe

def end_node_point(row, geom):
   return (Point(row[geom].coords.xy[0][-1], row[geom].coords.xy[1][-1]))

#%%create buffer and start/end node geometry columns

def add_nodeids_to_links_old(links,nodes,node_id,link_id,network_name):
   #this function takes in a set of links and creates an A and B column in the link gdf
   #using the specified id column in the nodes gdf


   #create starting and ending node columns in the links gdf
   links['start_node'] = links.apply(start_node_point, geom=links.geometry.name, axis=1)
   links['end_node'] = links.apply(end_node_point, geom= links.geometry.name, axis=1)

   #create very small buffer on nodes in nodes gdf and set active geo to buffer
   nodes['buffer'] = nodes.buffer(1)
   nodes = nodes.filter([f'{node_id}','buffer']).rename(columns={f'{node_id}':'node_id'})
   nodes = nodes.set_geometry('buffer')

   #start nodes first
   osm_links_start = links.set_geometry('start_node')

   #intersect start nodes with nodes
   osm_links_start = gpd.overlay(osm_links_start, nodes, how="intersection")

   #this isn't working switch to merge
   links = pd.merge(links, osm_links_start[[f'{link_id}','node_id']], on = f'{link_id}', how = 'left').rename(columns={'node_id':'A'})


   #end nodes next
   osm_links_end = links.set_geometry('end_node')

   osm_links_end = gpd.overlay(osm_links_end, nodes, how='intersection')

   links = pd.merge(links, osm_links_end[[f'{link_id}','node_id']], on = f'{link_id}', how = 'left').rename(columns={'node_id':'B'})

   #drop the start/end node columns so gdf can be exported
   links = links.drop(columns={'start_node','end_node'})

   return links

#%%
#osm_links = add_nodeids_to_links(osm_links,osm_nodes,'id','id','osm')

#osm_links.to_file('Base_Shapefiles/osm/osm_links_w_node_ids.geojson', driver = 'GeoJSON')


def add_nodeids_to_links(links,nodes,node_id,network_name):
   #this function takes in a set of links and creates an A and B column in the link gdf
   #using the specified id column in the nodes gdf
   #currently used for OSM
   
   #turn link index into column to serve as link id
   links['link_id'] = links.index
   
   #create very small buffer on nodes in nodes gdf and set active geo to buffer
   nodes['buffer'] = nodes.buffer(0.001)
   nodes = nodes.filter([f'{node_id}','buffer']).rename(columns={f'{node_id}':'node_id'})
   nodes = nodes.set_geometry('buffer')

   #start nodes first
   links_start = links.copy()
   links_start['start_node'] = links_start.apply(start_node_point, geom=links_start.geometry.name, axis=1)
   links_start = links_start.set_geometry('start_node').set_crs(epsg=2240)

   #intersect start nodes with nodes
   links_start = gpd.overlay(links_start, nodes, how="intersection")

   #merge with links and rename to A
   links = pd.merge(links, links_start[['link_id','node_id']], on = 'link_id', how = 'left').rename(columns={'node_id':'A'})

   #end nodes next
   links_end = links.copy()
   links_end['end_node'] = links_end.apply(end_node_point, geom= links_end.geometry.name, axis=1)
   links_end = links_end.set_geometry('end_node').set_crs(epsg=2240)
   links_end = gpd.overlay(links_end, nodes, how='intersection')
   links = pd.merge(links, links_end[['link_id','node_id']], on = 'link_id', how = 'left').rename(columns={'node_id':'B'})

   return links

osm_links_new = add_nodeids_to_links(osm_links,osm_nodes,'id','osm')

#%%

from shapely.geometry import Polygon
polys1 = gpd.GeoSeries([Polygon([(0,0), (2,0), (2,2), (0,2)]),
                               Polygon([(2,2), (4,2), (4,4), (2,4)])])
polys2 = gpd.GeoSeries([Polygon([(1,1), (3,1), (3,3), (1,3)]),
                              Polygon([(3,3), (5,3), (5,5), (3,5)])])
df1 = gpd.GeoDataFrame({'geometry': polys1, 'df1_data':[1,2]})
df2 = gpd.GeoDataFrame({'geometry': polys2, 'df2_data':[1,2]})

test = gpd.overlay(df1,df2,how='union')


#%%
#turned below into function

#create starting and ending node columns
osm_links['start_node'] = osm_links.apply(start_node_point, geom=osm_links.geometry.name, axis=1)
osm_links['end_node'] = osm_links.apply(end_node_point, geom= osm_links.geometry.name, axis=1)

#create very small buffer on nodes
osm_nodes['buffer'] = osm_nodes.buffer(0.0001)
osm_nodes = osm_nodes.filter(['id','buffer']).rename(columns={'id':'node_id'})
osm_nodes = osm_nodes.set_geometry('buffer')

#%% run up till here

#start nodes first
osm_links_start = osm_links.set_geometry('start_node')

#intersect start nodes with nodes
osm_links_start = gpd.overlay(osm_links_start, osm_nodes, how="intersection")

#this isn't working switch to merge
osm_links = pd.merge(osm_links, osm_links_start[['id','node_id']], on = 'id', how = 'left').rename(columns={'node_id':'A'})


#end nodes next
osm_links_end = osm_links.set_geometry('end_node')

osm_links_end = gpd.overlay(osm_links_end, osm_nodes, how='intersection')

osm_links = pd.merge(osm_links, osm_links_end[['id','node_id']], on = 'id', how = 'left').rename(columns={'node_id':'B'})

osm_links = osm_links.drop(columns={'start_node','end_node'})

#%% trim node id down to number and export

trim_node_id = lambda row: row.split("/")[1]

osm_links['A'] = osm_links['A'].apply(trim_node_id)
osm_links['B'] = osm_links['B'].apply(trim_node_id)



