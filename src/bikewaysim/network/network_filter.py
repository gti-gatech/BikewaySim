# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 13:22:22 2021

@author: tpassmore6
"""

#%%import cell
import geopandas as gpd
import pandas as pd
import numpy as np
import time
from shapely.geometry import Point
from pathlib import Path
import pickle

from bikewaysim import general_utils

def snap_to_network(to_snap,network_nodes_raw):
    #record the starting time
    time_start = time.time()
    
    #create copy of network nodes
    network_nodes = network_nodes_raw.copy()
    
    #rename geometry columns
    to_snap.rename(columns={'geometry':'original'},inplace=True)
    to_snap.set_geometry('original',inplace=True)
    network_nodes.rename(columns={'geometry':'snapped'},inplace=True)
    network_nodes.set_geometry('snapped',inplace=True)
    
    #find closest network node from each orig/dest
    snapped_nodes = general_utils.ckdnearest(to_snap, network_nodes)

    #filter columns
    snapped_nodes = snapped_nodes[to_snap.columns.to_list()+['N','dist']]
        
    #drop geo column
    snapped_nodes.drop(columns=['original'],inplace=True)
    
    print(f'snapping took {round(((time.time() - time_start)/60), 2)} minutes')
    return snapped_nodes

def import_study_area(settings):
    
    if settings['studyarea_layer'] is None:
        studyarea = gpd.read_file(settings['studyarea_filepath'])
    else:
        studyarea = gpd.read_file(settings['studyarea_filepath'],layer=settings['studyarea_layer'])
    
    #dissolve all polygons included in the layer
    studyarea = studyarea.dissolve()

    #re-project if neccessary
    if studyarea.crs != settings['project_crs']:
        studyarea.to_crs(settings['project_crs'],inplace=True)

    # calculate the area of the study area in square miles
    sqmi = int(studyarea.geometry.area / 5280**2)
    print(f"The study area is {sqmi} square miles.")

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

def network_import(settings:dict,network_dict:dict):

    studyarea = settings['studyarea']
    links_fp = network_dict['edges_filepath']
    network_name = network_dict['network_name']

    # use bounding box to mask instead of polygon boundaries
    if settings['use_bbox']:
        studyarea_bounds = studyarea.dissolve().envelope
    else:
        studyarea_bounds = studyarea.dissolve().convex_hull

    if links_fp.suffix in ['.gpkg','.shp','.gdb','.gpkg']:

        if network_dict['edges_layer'] is None:
            network_dict['edges_layer'] = 0
        
        links = gpd.read_file(links_fp,mask=studyarea_bounds,layer=network_dict['edges_layer'])

        if links.crs != settings['project_crs']:
            links.to_crs(settings['project_crs'],inplace=True)

    #TODO mask this data (DO NOT CLIP)
    elif links_fp.suffix in ['.pkl']:
        #pickle all attributes as is
        with links_fp.open('rb') as fh:
            links = pickle.load(fh)
            
    if links.crs != settings['project_crs']:
        links.to_crs(settings['project_crs'],inplace=True)

    # #create filepath and add base_layers.gpkg
    # if Path(settings['project_filepath'],'networks').exists() == False:
    #     Path(settings['project_filepath'],'networks').mkdir()

    studyarea_bounds.to_file(Path(settings['project_filepath'],'base_layers.gpkg'),layer='studyarea_bounds')

    return links
        
#use this to create a complete clean network
def filter_to_general(settings:dict,network_dict:dict):
    
    '''
    This function is used to filter a network to a study area and create a nodes layer if one is not provided.
    '''

    network_name = network_dict['network_name']

    links = network_import(settings,network_dict)
            
    #bring in or create nodes and add reference ids to links
    links, nodes = creating_nodes(links,settings,network_dict)

    #create linkids to distinguish links when they have the same start/end node
    #prefer pre-established linkid but if linkid is not unique create new linkids
    if links.columns.isin(['linkid']).any() & (network_dict['linkid'] is None):
        print('Column named linkid detected but not set as linkid column in settings.')
        links.rename(columns={'linkid':'undefined_linkid'},inplace=True)
    
    if network_dict['linkid'] is not None:
        if links.columns.isin([network_dict['linkid']]).any() == False:
            print('Provided linkid column name is not in the columns.')
            network_dict['linkid'] = None
        if links[network_dict['linkid']].duplicated().any():
            print('Provided linkids are not unique.')
            network_dict['former_linkid'] = network_dict['linkid'] 
            network_dict['linkid'] = None
            
    if network_dict['linkid'] is not None:
        links.rename(columns={network_dict['linkid']:f'linkid'},inplace=True)
    else:
        print('Generating unique link ids.')
        # if network_dict.get('former_linkid',0) != 0:
        #     # encoded_linkids = [encode_linkid(*tuple(x)) for x in links[[f"{network_name}_A",f"{network_name}_B",network_dict['former_linkid']]].values]
        #     links[f'linkid'] = links.reset_index()
        # else:
        links.reset_index(inplace=True)
        links.rename(columns={'index':f'linkid'},inplace=True)

    return links, nodes

#TDDO there are way fewer than this many combinations
def encode_linkid(val1, val2, val3):
    """Encodes three 64-bit integers into a single large number."""
    
    val1 = int(val1)
    val2 = int(val2)
    val3 = int(val3)
    
    bit_width = 64
    max_value = (1 << bit_width) - 1

    # Ensure each number is an integer and fits within 64 bits
    if any(x < 0 or x >= (1 << bit_width) for x in (val1, val2, val3)):
        raise ValueError("One or more values exceed the 64-bit limit.")

    # Shift the bits of the first two integers to make space for others
    return (val1 << (2 * bit_width)) | (val2 << bit_width) | val3

def decode_linkid(n):
    """Decodes a large number back into three 64-bit integers."""
    bit_width = 64
    mask = (1 << bit_width) - 1
    
    # Extract the values by using bit masks and shifts
    val1 = (n >> (2 * bit_width)) & mask
    val2 = (n >> bit_width) & mask
    val3 = n & mask
    
    return (val1, val2, val3)

def import_nodes(nodes_fp,nodes_layer,settings):
    '''
    Nodes are only used for transfering ids
    '''

    if nodes_fp.suffix in ['.gpkg','.shp','.gdb','.gpkg']:        
        if nodes_layer is None:
            nodes = gpd.read_file(nodes_fp)
        else:
            nodes = gpd.read_file(nodes_fp,layer=nodes_layer)

        if nodes.crs != settings['project_crs']:
            nodes.to_crs(settings['project_crs'],inplace=True)

    elif nodes_fp.suffix in ['.pkl']:
        #pickle all attributes as is
        with nodes_fp.open('rb') as fh:
            nodes = pickle.load(fh)
            
    if nodes.crs != settings['project_crs']:
        nodes.to_crs(settings['project_crs'],inplace=True)

    return nodes

def creating_nodes(links:gpd.GeoDataFrame,settings:dict,network_dict:dict):
    '''
    This function creates a node layer for the links.
    If a nodes layer is already available, then use that.

    It will also assign reference ids to links if they aren't provided.
    '''
    
    #TODO have cleaner way of extracting variables from dict
    nodes_fp = network_dict['nodes_filepath']
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
        
        nodes = import_nodes(nodes_fp,nodes_layer,settings)

        #check if there is a node id column
        if nodeid_check is None:
            print('Setting index as the node IDs.')
            nodes['N'] = nodes.index
        else:
            nodes.rename(columns={nodes_id:'N'},inplace=True)

        if a_and_b:
            print("and links and nodes have reference ids.")
            links.rename(columns={A:'A',B:'B'},inplace=True)
        else:
            print('but no reference ids for links.')
            links = add_ref_ids(links,nodes,network_name)

    # no nodes layer
    else:
        print('no nodes layer')
        if a_and_b:
            print('but links have reference ids')
            links.rename(columns={A:'A',B:'B'},inplace=True)
            nodes = make_nodes_refid(links)
        else:
            print('and links dont have reference ids')
            nodes = make_nodes(links,network_name)
            #add node ids to links
            links = add_ref_ids(links,nodes,network_name)

    return links, nodes
    
def remove_directed_links(links, A_col, B_col, linkid_col=None):
    """
    Uses the provided A, B, and linkid column to turn a dataframe
    of directed edges into an undirected one. Assumes that the
    geometry of a pair of directed edges are identical. This can pose
    issues if this is not true as oneway links may not simplify to the
    correct link direction.

    Linkid column can be added to preserve multigraphs.

    #TODO add a geometry based version?
    """

    links = links.copy().reset_index(drop=True)
    
    #sort by A_col and B_col
    df_dup = pd.DataFrame(
        np.sort(links[[A_col,B_col]], axis=1),
            columns=[A_col,B_col]
            )

    #add linkid back in if provided (index should be the same)
    if (linkid_col is None) == False:
        df_dup[linkid_col] = links[linkid_col]

    #create oneway column?
    #preserve directionality in column called 'bothways' (not sure if this is needed)
    #df_dup['bothways'] = df_dup.duplicated(keep=False)
    df_dup.drop_duplicates(inplace=True)
    links = links.loc[df_dup.index]

    return links

# Extract Start and End Points as tuples and round to reduce precision
def start_node(row, geom):
   coords = np.array(row[geom].coords)
   return (round(coords[0][0],5), round(coords[0][1],5))
def end_node(row, geom):
   coords = np.array(row[geom].coords)
   return (round(coords[-1][0],5), round(coords[-1][1],5))

# Extract start and end points but turn them into shapely Points
def start_node_geo(row, geom):
   coords = np.array(row[geom].coords)
   return (Point(coords[0][0], coords[0][1])) 
def end_node_geo(row, geom):
   coords = np.array(row[geom].coords)
   return (Point(coords[-1][0], coords[-1][1]))

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
    links['A'] = general_utils.ckdnearest(for_matching,nodes,return_dist=False)['N']

    #repeat for end point
    for_matching = links.copy()
    #make the first point the active geometry
    for_matching['pt_geometry'] = for_matching.apply(end_node_geo, geom='geometry', axis=1)
    for_matching.set_geometry('pt_geometry',inplace=True)
    for_matching.drop(columns='geometry',inplace=True)
    #find nearest node from starting node and add to column
    links['B'] = general_utils.ckdnearest(for_matching,nodes,return_dist=False)['N']

    #check for missing reference ids
    if links['A'].isnull().any() | links['B'].isnull().any():
        print("There are missing reference ids")
    else:
        print("Reference IDs successfully added to links.")
    return links

def make_nodes_refid(links): 
    '''
    This function creates a nodes layer from links with reference ids labeled as 'A' and 'B'
    '''
    #starting point
    nodes_A = links.copy()
    nodes_A.geometry = nodes_A.apply(start_node_geo,geom='geometry',axis=1)
    nodes_A = nodes_A[['A','geometry']]
    nodes_A.rename(columns={'A':'N'},inplace=True)
    #ending point
    nodes_B = links.copy()
    nodes_B.geometry = nodes_B.apply(end_node_geo,geom='geometry',axis=1)
    nodes_B = nodes_B[['B','geometry']]
    nodes_B.rename(columns={'B':'N'},inplace=True)

    #append
    nodes = pd.concat([nodes_A,nodes_B],ignore_index=True)

    #drop duplicates
    nodes.drop_duplicates(subset=['N'],inplace=True)

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

    #turn into dataframe and assign node ids in sequential order
    nodes = pd.DataFrame({'N':range(0,len(nodes)),'geometry':nodes})
    
    #turn the WKT coordinates into points
    nodes['geometry'] = nodes.apply(lambda row: Point([row['geometry']]), axis=1)
        
    #convert to GeoDataFrame and set geo and CRS
    nodes = gpd.GeoDataFrame(nodes,geometry='geometry',crs="epsg:4326").to_crs(orig_crs)
    
    return nodes

def filter_nodes(links,nodes,network_name):
    #remove nodes that aren't in the filtered links
    nodes_in = set(pd.concat([links['A'],links['B']],ignore_index=True))
    nodes_filt = nodes[nodes['N'].isin(nodes_in)]
    return nodes_filt

def export(links,nodes,network_name,settings,network_dict,cols):
    '''
    - Export the links and nodes into network.gpkg as layers
    - Remove excess columns as this data is stored in a pkl
    - If there was an old link id, retain that value for attribute merge
    '''
    start = time.time()
    #remove excess columns?
    links_cols_to_keep = ['A','B',f'linkid','oneway','link_type','geometry'] + cols
    if network_dict.get("former_linkid",0) != 0:
        links_cols_to_keep.append(network_dict['former_linkid'])
    nodes_cols_to_keep = ['N','geometry']
    links = links[links_cols_to_keep]
    nodes = nodes[nodes_cols_to_keep]
    #export
    export_fp = settings['project_filepath'] / 'networks.gpkg'
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

    #remove node layers
    layers = [x for x in layers if 'node' not in x]

    #go through each network
    for network in layers:
        links = gpd.read_file(settings['project_filepath']/'filtered.gpkg',layer=network)
    
        #get network name
        network_name = network.split('_')[0]
    
        #how many links
        num_links = len(links)
    
        #how many nodes
        nodes = pd.concat([links['A'],links['B']],ignore_index=True)
        num_nodes = len(nodes.unique())

        #total length
        length_mi = links.geometry.length / 5280 # create a new distance column and calculate mileage of each link
        tot_link_length = round(length_mi.sum(),0)

        #average link length
        avg_link_length = round(links.geometry.length.mean(),1)

        #create data frame
        summary_table.loc[network,:] = [num_links,num_nodes,tot_link_length,avg_link_length]

    #export summary table
    summary_table.to_csv(settings['project_filepath']/ "network_summary.csv")
   
    print(summary_table)
