# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 13:22:22 2021

@author: tpassmore6
"""

#%%import cell
import geopandas as gpd
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' #get rid of copy warning
import numpy as np
#np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  
import time
from shapely.geometry import Point, box
from pathlib import Path
import contextily as cx
import fiona
import warnings

# Suppress the shapely warning (not working)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from helper_functions import *

def import_study_area(settings):
    if settings['studyarea_layer'] is None:
        studyarea = gpd.read_file(settings['studyarea_fp'])
    else:
        studyarea = gpd.read_file(settings['studyarea_fp'],layer=settings['studyarea_layer'])
    
    #dissolve
    studyarea = studyarea.dissolve()

    #re-project if neccessary
    if studyarea.crs != settings['crs']:
        studyarea.to_crs(settings['crs'],inplace=True)

    # calculate the area of the study area in square miles
    sqmi = round(float(studyarea['geometry'].area / 5280**2),2) 
    print(f"The study area is {sqmi} square miles.")

    # print to examine
    studyarea.plot(figsize=(10,10),alpha=0.5,edgecolor='k')
    #ax = studyarea.plot(figsize=(10,10),alpha=0.5,edgecolor='k')
    #cx.add_basemap(ax, crs=studyarea.crs)

    return studyarea

#master function used to run all the filter functions
def filter_networks(settings:dict,network_dict:dict):
    '''
    This function runs all the other functions in this file. It takes in two settings dictionaries.

    settings:


    network_dict:
    
    "studyarea": studyarea, #geodataframe of the study area
    "networkfp": networkfp, #filepath for the network, specified earlier
    "network_name": 'abm', #name for the network being evaluated
    "network_mapper": network_mapper, #leave this, edit in the block above
    "layer": 0 #if network has layers, then specify which layer to look at; if no layers then leave as 0 
    "desired_crs": "desired_crs", # leave this, edit in the CRS block
    "nodesfp":None, # specify the nodes path if available, otherwise leave as None
    "node_id": None, # specify the column in the nodes file that has the node id information, if not available leave as 0
    "A": "A", #column with the starting node id; replace with None if there isn't a column
    "B": "B", #column with the ending node id; replace with None if there isn't a column
    "bbox": True, #use the bounding box of the study area as the mask for bringing in features instead of the polygon boundaries
    

    '''
    #record the starting time
    tot_time_start = time.time()
    
    network_name = network_dict['network_name']
    print(f'Filtering the {network_name} network.')

    #import the network
    links, nodes = filter_to_general(settings,network_dict)  
    
    #print the total time it took to run the code
    print(f'{network_name} imported... took {round(((time.time() - tot_time_start)/60), 2)} minutes')

    return links, nodes

#use this to create a complete clean network
def filter_to_general(settings:dict,network_dict:dict):
    
    '''
    This first filter function removes links
    that are not legally traversable by bike
    (e.g., Interstates, sidewalks, private drives)
    '''
    studyarea = settings['studyarea']
    links_fp = network_dict['links_fp']
    network_name = network_dict['network_name']

    if network_dict['links_layer'] is None:
        network_dict['links_layer'] = 0
    
    # use bounding box to mask instead of polygon boundaries
    if settings['use_bbox']:
        links = gpd.read_file(links_fp,bbox=tuple(studyarea.total_bounds),layer=network_dict['links_layer'])
    else:
        links = gpd.read_file(links_fp,mask=studyarea,layer=network_dict['links_layer'])
    
    # if network_dict['links_layer'] is None:
    #     links = gpd.read_file(links_fp, mask = mask, layer = 0)
    # else:
    #     links = gpd.read_file(links_fp, mask = mask, layer = network_dict['links_layer'])

    if links.crs != settings['crs']:
        links.to_crs(settings['crs'],inplace=True)

    #create linkids to distinguish links when they have the same start/end node
    #prefer pre-established linkid but if linkid is not unique create new linkids
    if links.columns.isin(['linkid']).any():
        print('Column named linkid detected but not set as linkid column in settings.')
        links.rename(columns={'linkid':'undefined_linkid'},inplace=True)
    
    if network_dict['linkid'] is not None:
        if ~links.columns.isin([network_dict['linkid']]).any():
            print('Provided linkid is not in the columns.')
            network_dict['linkid'] = None
        
        if links[network_dict['linkid']].duplicated().any():
            print('Provided linkid is not unique.')
            network_dict['linkid'] = None
            
    if network_dict['linkid'] is not None:
        links.rename(columns={network_dict['linkid']:f'{network_name}_linkid'},inplace=True)
    else:
        print('Generating unique link ids.')
        links.reset_index(inplace=True)
        links.rename(columns={'index':f'{network_name}_linkid'},inplace=True)

    #bring in or create nodes and add reference ids to links
    links, nodes = creating_nodes(links,settings,network_dict)

    #export the attributes
    links.drop(columns=['geometry']).to_pickle(settings['output_fp']/f'{network_name}_attr.pkl')

    #initialize a link type column
    links['link_type'] = np.nan

    return links, nodes

def creating_nodes(links:gpd.GeoDataFrame,settings:dict,network_dict:dict):
    '''
    This function creates a node layer for the links.
    If a nodes layer is already available, then use that.

    It will also assign reference ids to links if they aren't provided.
    '''
    
    #TODO have cleaner way of extracting variables from dict
    nodes_fp = network_dict['nodes_fp']
    A = network_dict['A']
    B = network_dict['B']
    nodes_layer = network_dict['nodes_layer']
    nodes_id = network_dict['nodes_id']
    network_name = network_dict['network_name']

    #conditions to check
    a_and_b = (A is not None) & (B is not None)
    nodeid_check = nodes_id is not None

    if nodes_fp is not None:
        print('There is a nodes layer...')
        
        if nodes_layer is None:
            nodes = gpd.read_file(nodes_fp)
        else:
            nodes = gpd.read_file(nodes_fp,layer=nodes_layer)

        if nodes.crs != settings['crs']:
            nodes.to_crs(settings['crs'],inplace=True)

        #check if there is a node id column
        if nodeid_check is None:
            print('Setting index as the node IDs.')
            nodes[f'{network_name}_N'] = nodes.index
        else:
            nodes.rename(columns={nodes_id:f'{network_name}_N'},inplace=True)

        if a_and_b:
            print("and links and nodes have reference ids.")
            links.rename(columns={A:f'{network_name}_A',B:f'{network_name}_B'},inplace=True)
        else:
            print('but no reference ids for links.')
            links = add_ref_ids(links,nodes,network_name)

    # no nodes layer
    else:
        print('no nodes layer')
        if a_and_b:
            print('but links have reference ids')
            links.rename(columns={A:f'{network_name}_A',B:f'{network_name}_B'},inplace=True)
            nodes = make_nodes_refid(links,network_name)
        else:
            print('and links dont have reference ids')
            nodes = make_nodes(links,network_name)
            #add node ids to links
            links = add_ref_ids(links,nodes,network_name)

    return links, nodes
    

def remove_directed_links(links, network_name):
    #remove directed links
    df_dup = pd.DataFrame(
        np.sort(links[[f"{network_name}_A",f"{network_name}_B"]], axis=1),
            columns=[f"{network_name}_A",f"{network_name}_B"]
            )

    #preserve directionality in column called 'bothways'
    df_dup['bothways'] = df_dup.duplicated(keep=False)
    df_dup = df_dup.drop_duplicates()
    links = pd.merge(links,df_dup,how='inner', left_index = True, right_index = True,
                    suffixes=(None,'_drop')).drop(columns={f'{network_name}_A_drop',f'{network_name}_B_drop'})
    
    return links

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

def add_ref_ids(links,nodes,network_name):
    '''
    This function adds reference columns to links from the nodes id column
    '''
    for_matching = links.copy()

    #make the first point the active geometry
    for_matching['pt_geometry'] = for_matching.apply(start_node_geo, geom='geometry', axis=1)
    for_matching.set_geometry('pt_geometry',inplace=True)
    for_matching.drop(columns='geometry',inplace=True)
    #find nearest node from starting node and add to column
    links[f'{network_name}_A'] = ckdnearest(for_matching,nodes,return_dist=False)[f'{network_name}_N']

    #repeat for end point
    for_matching = links.copy()
    #make the first point the active geometry
    for_matching['pt_geometry'] = for_matching.apply(end_node_geo, geom='geometry', axis=1)
    for_matching.set_geometry('pt_geometry',inplace=True)
    for_matching.drop(columns='geometry',inplace=True)
    #find nearest node from starting node and add to column
    links[f'{network_name}_B'] = ckdnearest(for_matching,nodes,return_dist=False)[f'{network_name}_N']

    #check for missing reference ids
    if links[f'{network_name}_A'].isnull().any() | links[f'{network_name}_B'].isnull().any():
        print("There are missing reference ids")
    else:
        print("Reference IDs successfully added to links.")
    return links

def make_nodes_refid(links, network_name): 
    '''
    This function creates a nodes layer from links with reference ids.
    '''
    #starting point
    nodes_A = links.copy()
    nodes_A.geometry = nodes_A.apply(start_node_geo,geom='geometry',axis=1)
    nodes_A = nodes_A[[f'{network_name}_A','geometry']]
    nodes_A.rename(columns={f'{network_name}_A':f'{network_name}_N'},inplace=True)
    #ending point
    nodes_B = links.copy()
    nodes_B.geometry = nodes_B.apply(end_node_geo,geom='geometry',axis=1)
    nodes_B = nodes_B[[f'{network_name}_B','geometry']]
    nodes_B.rename(columns={f'{network_name}_B':f'{network_name}_N'},inplace=True)

    #append
    nodes = pd.concat([nodes_A,nodes_B],ignore_index=True)

    #drop duplicates
    nodes.drop_duplicates(subset=[f'{network_name}_N'],inplace=True)

    return nodes

def make_nodes(links, network_name):
    links_copy = links.copy()

    #turn to unprojected CRS
    orig_crs = links_copy.crs
    links_copy.to_crs("epsg:4326",inplace=True)

    #extract start and end node, eliminate duplicates, turn into points    
    #add start and end coordinates to each line
    start_nodes = links_copy.apply(start_node, geom='geometry', axis=1)
    end_nodes = links_copy.apply(end_node, geom='geometry', axis=1)

    nodes = pd.concat([start_nodes,end_nodes],ignore_index=True)
    nodes.drop_duplicates(inplace=True)

    #turn into dataframe
    nodes = pd.DataFrame({f'{network_name}_N':range(0,len(nodes)),'geometry':nodes})
    
    #turn the WKT coordinates into points
    nodes['geometry'] = nodes.apply(lambda row: Point([row['geometry']]), axis=1)
        
    #convert to GeoDataFrame and set geo and CRS
    nodes = gpd.GeoDataFrame(nodes,geometry='geometry',crs="epsg:4326").to_crs(orig_crs)
    
    return nodes

def filter_nodes(links,nodes,network_name):
    #remove nodes that aren't in the filtered links
    nodes_in = set(pd.concat([links[f'{network_name}_A'],links[f'{network_name}_B']],ignore_index=True))
    nodes_filt = nodes[nodes[f'{network_name}_N'].isin(nodes_in)]
    return nodes_filt

def export(links,nodes,network_name,settings):
    start = time.time()
    #remove excess columns for now
    links = links[[f'{network_name}_A',f'{network_name}_B',f'{network_name}_linkid','link_type','geometry']]
    nodes = nodes[[f'{network_name}_N','geometry']]
    #export
    export_fp = settings['output_fp'] / 'filtered.gpkg'
    links.to_file(export_fp,layer=f'{network_name}_links')
    nodes.to_file(export_fp,layer=f'{network_name}_nodes')
    end = time.time()
    print('Export took',np.round((end-start)/60,2),'minutes')
    return

def summary(settings):
    '''
    Look at and summurize features in a filter.gpkg file.
    '''

    #summary table
    #can add other metrics of interest in the future
    summary_table = pd.DataFrame(columns=['num_links','num_nodes','tot_link_length','avg_link_length'])
    
    #expected link types
    layers = fiona.listlayers(settings['output_fp']/ 'filtered.gpkg')

    #remove node layers
    layers = [x for x in layers if 'node' not in x]

    #go through each network
    for network in layers:
        links = gpd.read_file(settings['output_fp']/'filtered.gpkg',layer=network)
    
        #get network name
        network_name = network.split('_')[0]
    
        #how many links
        num_links = len(links)
    
        #how many nodes
        nodes = pd.concat([links[f'{network_name}_A'],links[f'{network_name}_B']],ignore_index=True)
        num_nodes = len(nodes.unique())

        #total length
        length_mi = links.geometry.length / 5280 # create a new distance column and calculate mileage of each link
        tot_link_length = round(length_mi.sum(),0)

        #average link length
        avg_link_length = round(links.geometry.length.mean(),1)

        #create data frame
        summary_table.loc[network,:] = [num_links,num_nodes,tot_link_length,avg_link_length]

    #export summary table
    summary_table.to_csv(settings['output_fp']/ "network_summary.csv")
   
    print(summary_table)
