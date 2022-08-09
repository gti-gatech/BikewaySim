# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 14:09:06 2021

@author: tpassmore6
"""

#%%import cell
import geopandas as gpd
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' #get rid of copy warning
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  
import os
import time
import pickle
from shapely.geometry import Point, LineString

#%% helper functions

# Extract Start and End Points as tuples and round to reduce precision
def start_node(row, geom):
   #basically look at x and then y coord, use apply to do this for every row of a dataframe
   return (round(row[geom].coords.xy[0][0],5), round(row[geom].coords.xy[1][0],5))
def end_node(row, geom):
   return (round(row[geom].coords.xy[0][-1],5), round(row[geom].coords.xy[1][-1],5))

# Extract start and end points but turn them into shapely Points
def start_node_geo(row, geom):
   return (Point(row[geom].coords.xy[0][0], row[geom].coords.xy[1][0])) 
def end_node_geo(row, geom):
   return (Point(row[geom].coords.xy[0][-1], row[geom].coords.xy[1][-1]))

#%%

def create_node_ids(links, network_name, network_mapper):
    #need to create new nodes IDs for OSM and TIGER networks
    #extract nodes, find unique nodes, number sequentially, match ids back to links start end nodes
    
    #turn to unprojected and then round
    orig_crs = links.crs
    links = links.to_crs("epsg:4326")
    
    #add start and end coordinates to each line so we can match them
    links['start_node'] = links.apply(start_node, geom= links.geometry.name, axis=1)
    links['end_node'] = links.apply(end_node, geom= links.geometry.name, axis=1)

    #create one long list of nodes to find unique nodes
    nodes_coords = links['start_node'].append(links['end_node'])
    
    #turn series into data frame
    nodes = pd.DataFrame({f'{network_name}_coords':nodes_coords})
    
    #find number of intersecting links
    nodes[f'{network_name}_num_links'] = 1 #this column will be used to count the number of links
    
    #for each unique ID and coordinate, count number of links
    nodes = nodes.groupby([f'{network_name}_coords'], as_index=False).count()
    
    #give nodes row number name
    #CAUTION, node IDs will change based on study area, and/or ordering of the data 
    nodes[f'{network_name}_ID'] = np.arange(nodes.shape[0]).astype(str)
    nodes[f'{network_name}_ID'] = network_mapper[network_name] + network_mapper['generated'] + nodes[f'{network_name}_ID']
    
    #extract only the ID and coords column for joining
    nodes = nodes[[f'{network_name}_ID',f'{network_name}_coords']]
    
    #turn nodes into gdf
    nodes = gpd.GeoDataframe(nodes,geometry=f'{network_name}_coords',crs="epsg:4326").to_crs(orig_crs)
    
    #perform join back to original dataframe
    #rename id to be A, rename coords to match start_node
    joining = nodes.rename(columns={f'{network_name}_ID':f'{network_name}_A',f'{network_name}_coords':'start_node'})
    links = pd.merge(links, joining, how = 'left', on = 'start_node' )

    #rename id to be B, rename coords to match end_node
    joining = nodes.rename(columns={f'{network_name}_ID':f'{network_name}_B',f'{network_name}_coords':'end_node'})
    links = pd.merge(links, joining, how = 'left', on = 'end_node' )
    links = links.rename(columns={f'{network_name}_ID':f'{network_name}_B'})
    
    #drop tuple columns
    links = links.drop(columns=['start_node','end_node'])

    #create an A_B column
    links[f'{network_name}_A_B'] = links[f'{network_name}_A'] + '_' + links[f'{network_name}_B']

    return links, nodes

#%% rename node ID's function
def rename_refcol(df, network_name, A, B, network_mapper):  
    #renames node ID column name to network name _ A/_B
    df = df.rename(columns={A:f'{network_name}_A',B:f'{network_name}_B'})

    #adds numbers to the beginning from network mapper
    df[f'{network_name}_A'] = network_mapper[network_name] + network_mapper['original'] + df[f'{network_name}_A'].astype(str)
    df[f'{network_name}_B'] = network_mapper[network_name] + network_mapper['original'] + df[f'{network_name}_B'].astype(str)
    #create an A_B column
    df[f'{network_name}_A_B'] = df[f'{network_name}_A'] + '_' + df[f'{network_name}_B']
    return df

def rename_nodes(df, network_name, node_id, network_mapper):  
    #renames node ID column name to network name _ A/_B
    df = df.rename(columns={node_id:f'{network_name}_ID'})
    #adds numbers to the beginning from network mapper
    df[f'{network_name}_ID'] = network_mapper[network_name] + network_mapper['original'] + df[f'{network_name}_ID'].astype(str)
    return df

#%% Create node layer from lines
#do this from the links to make sure that all nodes are included even if they would have been clipped
#by the study area  
def make_nodes(df, network_name): 

    #turn to unprojected and then round
    orig_crs = df.crs
    df = df.to_crs("epsg:4326")

    #extract start and end node, eliminate duplicates, turn into points    
    #add start and end coordinates to each line
    df['start_node'] = df.apply(start_node, geom= df.geometry.name, axis=1)
    df['end_node'] = df.apply(end_node, geom= df.geometry.name, axis=1)

    #stack start/end node coords and IDs on top of each other
    nodes_id = df[f'{network_name}_A'].append(df[f'{network_name}_B'])
    nodes_coords = df['start_node'].append(df['end_node'])
    
    #turn into dataframe
    nodes = pd.DataFrame({f"{network_name}_ID":nodes_id,f"{network_name}_coords":nodes_coords})
    
    #turn the coordinates into points so we can do spatial mapping
    nodes[f'{network_name}_coords'] = nodes.apply(lambda row: Point([row[f'{network_name}_coords']]), axis=1)
    
    #convert to GeoDataFrame and set geo and CRS
    nodes = gpd.GeoDataFrame(nodes).set_geometry(f'{network_name}_coords').set_crs("epsg:4326").to_crs(orig_crs)
    
    return nodes

#%% filter nodes

def filter_nodes(links,nodes,network_name):
    #remove nodes that aren't in filtered links
    nodes_in = links[f'{network_name}_A'].append(links[f'{network_name}_B']).unique()
    nodes_filt = nodes[nodes[f'{network_name}_ID'].isin(nodes_in)]
    return nodes_filt

