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

#%%

def add_nodes(links, studyarea_name, network_name, link_type, network_mapper, A = None, B = None):
    	
    #figure out where network mapper needs to be added
    if A is not None and B is not None:
    	#first rename the nodes and column names
        links = rename_nodes(links, network_name, link_type, A, B, network_mapper)   
    else:
        #need to create new node id's
        links = create_node_ids(links, network_name, link_type, network_mapper)
    
    #create nodes
    nodes = make_nodes(links, network_name, link_type, studyarea_name)
    
    #remove tuple
    links = links.drop(columns=['start_node','end_node'])
    
    return nodes, links


# Extract Start and End Points Code from a LineString as tuples
def start_node(row, geom):
   return (row[geom].coords.xy[0][0], row[geom].coords.xy[1][0]) #basically look at x and then y coord, use apply to do this for every row of a dataframe

def end_node(row, geom):
   return (row[geom].coords.xy[0][-1], row[geom].coords.xy[1][-1])


def create_node_ids(links, network_name, link_type, network_mapper):
    #need to create new nodes IDs for OSM and TIGER networks
    #extract nodes, find unique nodes, number sequentially, match ids back to links start end nodes
    
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

    return links

#%% rename node ID's function
def rename_nodes(df, network_name, link_type, A, B, network_mapper):  
    #renames node ID column name to network name _ A/_B
    df = df.rename(columns={A:f'{network_name}_A',B:f'{network_name}_B'})

    #adds numbers to the beginning from network mapper
    df[f'{network_name}_A'] = network_mapper[network_name] + network_mapper['original'] + df[f'{network_name}_A'].astype(str)
    df[f'{network_name}_B'] = network_mapper[network_name] + network_mapper['original'] + df[f'{network_name}_B'].astype(str)
    #create an A_B column
    df[f'{network_name}_A_B'] = df[f'{network_name}_A'] + '_' + df[f'{network_name}_B']
    return df


#%% Create node layer from lines
#do this from the links to make sure that all nodes are included even if they would have been clipped
#by the study area  
def make_nodes(df, network_name, link_type, studyarea_name): 

    #extract start and end node, eliminate duplicates, turn into points    

    #add start and end coordinates to each line
    df['start_node'] = df.apply(start_node, geom= df.geometry.name, axis=1)
    df['end_node'] = df.apply(end_node, geom= df.geometry.name, axis=1)

    #stack start/end node coords and IDs on top of each other
    nodes_id = df[f'{network_name}_A'].append(df[f'{network_name}_B'])
    nodes_coords = df['start_node'].append(df['end_node'])
    
    #turn into dataframe
    nodes = pd.DataFrame({f"{network_name}_ID":nodes_id,f"{network_name}_coords":nodes_coords})
    
    #find number of intersecting links
    nodes[f'{network_name}_num_links'] = 1
    
    #for each unique ID and coordinate, count number of links
    nodes = nodes.groupby([f'{network_name}_ID',f'{network_name}_coords'], as_index=False).count()
    
    #turn the coordinates into points so we can do spatial mapping
    nodes[f'{network_name}_coords'] = nodes.apply(lambda row: Point([row[f'{network_name}_coords']]), axis=1)
    
    #convert to GeoDataFrame and set geo and CRS
    nodes = gpd.GeoDataFrame(nodes).set_geometry(f'{network_name}_coords').set_crs(epsg=2240)
    
    return nodes
