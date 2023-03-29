# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 10:12:50 2022

@author: tpassmore6
"""

import networkx as nx
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.ops import MultiLineString
from tqdm import tqdm

from itertools import chain
import time

from helper_functions import *

def snap_ods_to_network(od_pairs:pd.DataFrame,df_nodes:gpd.GeoDataFrame):
    """
    This function takes in a dataframe of OD pairs and geodataframe of network
    nodes and matches the OD pair coordinates to their nearest network node.
    Then network node ID and crow-flies (Euclidean) distance for the origin and
    destination is returned as an added column.

    The OD dataframe must have the following columns and the LAT/LON columns
    must be unprojected coordinates:
    'ori_id', 'ori_lat', 'ori_lon', 'dest_id', 'dest_lat', 'dest_lon'

    The nodes geodataframe must have an ID column called "N"

    The CRS of df_nodes is used to calculate the distance.

    """
    
    df_nodes_raw = df_nodes.copy()

    #put all ods into one to prevent duplicate matching
    origs = od_pairs[['ori_id','ori_lat','ori_lon']].rename(
        columns={'ori_id':'id','ori_lat':'lat','ori_lon':'lon'})
    dests = od_pairs[['dest_id','dest_lat','dest_lon']].rename(
        columns={'dest_id':'id','dest_lat':'lat','dest_lon':'lon'})
    comb = origs.append(dests).drop_duplicates()

    #turn into gdf and project to same crs as df_nodes
    comb['geometry'] = gpd.points_from_xy(comb['lon'], comb['lat'], crs='epsg:4326')
    comb = gpd.GeoDataFrame(comb).to_crs(df_nodes.crs)

    #add suffix to distinguish columns
    comb.columns = comb.columns + '_ods'
    df_nodes_raw.columns = df_nodes_raw.columns + '_nds'
    
    comb = comb.set_geometry('geometry_ods')
    df_nodes_raw = df_nodes_raw.set_geometry('geometry_nds')
    
    #find closest node
    closest_node = ckdnearest(comb, df_nodes_raw)

    # formatting for official bikewaysim code
    # o = origin
    # d = destination
    # o_d = distance between origin and nearest network node
    # o_t = walking time between origin and nearest network node
    # ox_sq = a rounded X coord for origin
    
    #rename columns to make dataframe
    closest_node = closest_node.drop(columns=['lat_ods','lon_ods','geometry_ods','geometry_nds','lon_nds','lat_nds'])
    origs = closest_node.rename(columns={'id_ods':'ori_id','N_nds':'o_node','X_nds':'ox','Y_nds':'oy','dist':'o_d'})
    dests = closest_node.rename(columns={'id_ods':'dest_id','N_nds':'d_node','X_nds':'dx','Y_nds':'dy','dist':'d_d'})
    
    #merge back to od_pairs
    od_pairs = pd.merge(od_pairs, origs, on='ori_id',suffixes=(None,None))
    od_pairs = pd.merge(od_pairs, dests, on='dest_id',suffixes=(None,None))
    
    return od_pairs
    
def create_graph(links,impedance_col):
    DGo = nx.DiGraph()  # create directed graph
    for ind, row2 in links.iterrows():
        DGo.add_weighted_edges_from([(str(row2['A']), str(row2['B']), float(row2[impedance_col]))],weight=impedance_col)   
    return DGo
        
def find_shortest(links:gpd.GeoDataFrame,nodes:gpd.GeoDataFrame,ods:pd.DataFrame,impedance_col:str):
    #record the starting time
    time_start = time.time()
    
    #create network graph
    DGo = create_graph(links,impedance_col)    
    
    #initialize empty dicts
    #all_impedances = {}
    all_nodes = {}
    all_paths = {}
    
    #listcheck
    ods['tup'] = list(zip(ods.o_node,ods.d_node))
    listcheck = set(ods['tup'].to_list())
    
    #from each origin find route to all other pairs
    #NOTE: routing is from snapped network node, not origin node
    for origin in tqdm(ods.o_node.unique()):
        #run dijkstra's algorithm (no limit to links considered)
        impedances, paths = nx.single_source_dijkstra(DGo,origin,weight='weight')    

        #iterate through dijkstra results to add them to ods dataframe
        for key in impedances.keys():
            #check if trip is in one of the ones we want
            if (origin,key) in listcheck:
                #for each trip id find impedance
                ods.at[ods['tup']==(origin,key),impedance_col] = impedances[key]
                #all_impedances[(origin,key)] = impedances[key]
              
                #convert from node list to edge list
                node_list = paths[key]
                edge_list = [ node_list[i]+'_'+node_list[i+1] for i in range(len(node_list)-1)]

                #store
                #ods.at[ods['tup']==(origin,key),'node_list'] = node_list
                #ods.at[ods['tup']==(origin,key),'edge_list'] = edge_list
                
                #for btw centrality and mapping
                all_nodes[(origin,key)] = node_list
                all_paths[(origin,key)] = edge_list

    #calculate betweeness centrality
    links, nodes = btw_centrality(all_nodes,all_paths,links,nodes,impedance_col)

    #add geometry
    all_geos = add_geo(all_paths,links)

    #add to impedance df and make gdf
    ods['geometry'] = ods['tup'].map(all_geos)

    #create gdf
    ods = gpd.GeoDataFrame(ods,geometry='geometry',crs=links.crs)

    #get the length of the route in the units of the crs
    ods['length'] = ods.length

    #drop tuple
    ods.drop(columns=['tup'],inplace=True)

    return ods, links, nodes

def btw_centrality(all_nodes:dict,all_paths:dict,links:gpd.GeoDataFrame,nodes:gpd.GeoDataFrame, impedance_col):

    #calculate betweenness centrality
    node_btw_centrality = pd.Series(list(chain(*[all_nodes[key] for key in all_nodes.keys()]))).value_counts()
    edge_btw_centrality = pd.Series(list(chain(*[all_paths[key] for key in all_paths.keys()]))).value_counts()
    
    #add betweenness centrality as network attribute
    nodes[f'{impedance_col}_btw_cntrlty'] = nodes['N'].map(node_btw_centrality)
    links[f'{impedance_col}_btw_cntrlty'] = links['A_B'].map(edge_btw_centrality)
    
    #fill all empty with zeros
    nodes[f'{impedance_col}_btw_cntrlty'].fillna(0,inplace=True)
    links[f'{impedance_col}_btw_cntrlty'].fillna(0,inplace=True)

    #what percent of trips used these links
    nodes[f'{impedance_col}_std_btw_cntrlty'] = nodes[f'{impedance_col}_btw_cntrlty'] / len(all_nodes)#nodes[f'{impedance_col}_btw_cntrlty'].sum()
    links[f'{impedance_col}_std_btw_cntrlty'] = links[f'{impedance_col}_btw_cntrlty'] / len(all_paths)#links[f'{impedance_col}_btw_cntrlty'].sum()

    return links, nodes

def add_geo(all_paths:dict,links:gpd.GeoDataFrame):

    geos_dict = dict(zip(links['A_B'],links['geometry']))
    
    all_geos = {}
    for key in all_paths.keys():
        id_list = all_paths[key]
        geos_list = [geos_dict.get(id,0) for id in id_list]
        if geos_list != []:
            all_geos[key] = MultiLineString(geos_list)

    return all_geos

def percent_detour(dist,imp,tazs):
    '''
    Finds the percent detour and returns it as a column on the ods_imp gdf
    '''
    #drop the geometry col for ods_dist
    imp = pd.merge(imp,dist[['trip_id','phys_dist']],on='trip_id',suffixes=('','_y'))

    #calculate pct detour
    imp['percent_detour'] = (imp['phys_dist'] - imp['phys_dist_y']) / imp['phys_dist_y'] * 100

    #get origin taz from trip id   
    imp['FID_1'] = imp['trip_id'].str.split('_',expand=True)[0]

    #groupby for viz
    by_taz = imp.groupby('FID_1')['percent_detour'].mean().reset_index()

    #add taz geo
    by_taz = pd.merge(by_taz,tazs,on='FID_1')

    return imp, by_taz

def make_bikeshed(links,nodes,taz,ods,radius,buffer_size,impedance_col):
    '''
    Get the bikeshed for select tazs.
    '''

    #get network node
    origin = taz
    #origin = ods.loc[ods['ori_id']==taz,'o_node'].unique().item()

    #turn links into directed graph
    DGo = create_graph(links,impedance_col)
    
    #create bikeshed
    #https://networkx.org/documentation/stable/reference/generated/networkx.generators.ego.ego_graph.html
    #https://geonetworkx.readthedocs.io/en/latest/5_Isochrones.html
    bikeshed = nx.ego_graph(DGo, radius=radius, n=origin, distance=impedance_col)
            
    #make column for cleaner merging
    links['A_B_tup'] = list(zip(links['A'],links['B']))

    #get all the links that contained in the egograph
    bikeshed = links.loc[links['A_B_tup'].isin(list(bikeshed.edges)),:]
    bikeshed_node = nodes.loc[nodes['N']==origin,:]
    
    #drop tuple column
    bikeshed.drop(columns=['A_B_tup'],inplace=True)
    
    #drop dual links to get accurate size
    #TODO fix this
    df_dup = drop_duplicate_links(bikeshed)
    
    print(f'---{taz}---')
    print(f'Bikeshed Network Miles: {df_dup.length.sum()/5280}')
    print(f'Bikeshed Size (square miles w/{radius} ft access distance): {df_dup.buffer(buffer_size).area.sum()/5280/5280}')    

    return bikeshed, bikeshed_node

def drop_duplicate_links(links):
    df_dup = pd.DataFrame(np.sort(links[["A","B"]], axis=1), columns=["A","B"])
    df_dup.drop_duplicates(inplace=True)
    merged = pd.merge(links,df_dup,how='inner', left_index = True, right_index = True, suffixes=(None,'_drop')).drop(columns={'A_drop','B_drop'})
    return merged