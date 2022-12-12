# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 10:12:50 2022

@author: tpassmore6
"""

import networkx as nx
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from itertools import chain
import time

import os
from pathlib import Path

user_directory = os.fspath(Path.home()) #get home directory and convert to path string
file_directory = r"/Documents/BikewaySimData" #directory of bikewaysim network processing code

#change this to where you stored this folder
os.chdir(user_directory+file_directory)

#%%

#take in od data find the neearest node in the network
def snap_ods_to_network(df_points,df_nodes_raw):

    #put all ods into one
    origs = df_points[['ori_id','ori_lat','ori_lon']].rename(
        columns={'ori_id':'id','ori_lat':'lat','ori_lon':'lon'})
    dests = df_points[['dest_id','dest_lat','dest_lon']].rename(
        columns={'dest_id':'id','dest_lat':'lat','dest_lon':'lon'})
    comb = origs.append(dests).drop_duplicates()

    comb['geometry'] = gpd.points_from_xy(comb['lon'], comb['lat'], crs='epsg:4326')

    #needs to be projected coordinate system
    comb = gpd.GeoDataFrame(comb).to_crs('epsg:2240')

    #take in two geometry columns and find nearest gdB point from each
    #point in gdA. Returns the matching distance too.
    #MUST BE PROJECTED COORDINATE SYSTEM
    def ckdnearest(gdA, gdB, return_dist=True):  
        
        nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
        nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
        btree = cKDTree(nB)
        dist, idx = btree.query(nA, k=1)
        gdB_nearest = gdB.iloc[idx].reset_index(drop=True)
        
        gdf = pd.concat(
            [
                gdA.reset_index(drop=True),
                gdB_nearest,
                pd.Series(dist, name='dist')
            ], 
            axis=1)
        
        if return_dist == False:
            gdf = gdf.drop(columns=['dist'])
        
        return gdf

    comb.columns = comb.columns + '_pts'
    df_nodes_raw.columns = df_nodes_raw.columns + '_nds'
    
    comb = comb.set_geometry('geometry_pts')
    df_nodes_raw = df_nodes_raw.set_geometry('geometry_nds')
    
    #find closest node
    closest_node = ckdnearest(comb, df_nodes_raw)

    # o = origin
    # d = destination
    # o_d = distance between origin and nearest network node
    # o_t = walking time between origin and nearest network node
    # ox_sq = a rounded X coord for origin
    
    #rename columns to make dataframe
    closest_node = closest_node.drop(columns=['lat_pts','lon_pts','geometry_pts','geometry_nds','lon_nds','lat_nds'])
    origs = closest_node.rename(columns={'id_pts':'ori_id','N_nds':'o_node','X_nds':'ox','Y_nds':'oy','dist':'o_d'})
    dests = closest_node.rename(columns={'id_pts':'dest_id','N_nds':'d_node','X_nds':'dx','Y_nds':'dy','dist':'d_d'})
    
    #merge back to df_points
    df_points = pd.merge(df_points, origs, on='ori_id', how='left')
    df_points = pd.merge(df_points, dests, on='dest_id', how='left')
    
    return df_points
    

def create_graph(links,impedance_col):
    DGo = nx.DiGraph()  # create directed graph
    for ind, row2 in links.iterrows():
        DGo.add_weighted_edges_from([(str(row2['A']), str(row2['B']), float(row2[impedance_col]))],weight='weight')   
    return DGo
        

def find_shortest(links,nodes,ods,impedance_col):
    #record the starting time
    time_start = time.time()
    
    #create network graph
    DGo = create_graph(links,impedance_col)    
    
    #initialize empty dicts
    all_impedances = {}
    all_paths = {}
    
    #listcheck
    listcheck = ods['tup'].to_list()
    
    #from each unique origin
    for origin in ods.o_node.unique():
        #run dijkstra's algorithm (no limit to links considered)
        impedances, paths = nx.single_source_dijkstra(DGo,origin,weight='weight')    
        
        #iterate through dijkstra results
        for key in impedances.keys():
            #check if trip is in one of the ones we want
            if (origin,key) in listcheck:
                #for each trip id find impedance
                all_impedances[(origin,key)] = impedances[key]
              
                #convert from node list to edge list
                node_list = paths[key]
                edge_list = [ node_list[i]+'_'+node_list[i+1] for i in range(len(node_list)-1)]
                all_paths[(origin,key)] = edge_list

    #add impedance column to ods dataframe
    ods[f'{impedance_col}'] = ods['tup'].map(all_impedances)
            
    links, all_paths_gdf = create_paths(links,all_paths,ods,impedance_col)
    
    #exporting
    export_time = time.time()
    #all_paths_gdf.to_file(r'trb2023\outputs.gpkg',layer=impedance_col,driver='GPKG')
    print(f'export took {round(((time.time() - export_time)/60), 2)} minutes')
    
    
    print(f'took {round(((time.time() - time_start)/60), 2)} minutes')
    
    return ods, links
    
def create_paths(links,all_paths,ods,impedance_col):
    
    #get the btw centrality by counting the number of times each link appears
    btw_centrality = pd.Series(list(chain(*[all_paths[key] for key in all_paths.keys()]))).value_counts()
    
    #add betweennes centrality to links
    links[f'{impedance_col}_btw_cntrlty'] = links['A_B'].map(btw_centrality)
    
    #add trip line geometry
    all_paths_geo = {}
    
    for key in all_paths.keys():
        #add item
        all_paths_geo[key] = links[links['A_B'].isin(all_paths[key])].dissolve().geometry.item()

    #add geo column
    ods['geometry'] = ods['tup'].map(all_paths_geo)

    #create gdf
    all_paths_gdf = gpd.GeoDataFrame(ods,geometry='geometry',crs='epsg:2240')
    
    #drop tuple
    all_paths_gdf.drop(columns=['tup'],inplace=True)
    
    return links, all_paths_gdf

#%%

#trb 2023 inputs
# ods = pd.read_csv(r'bikewaysim_outputs/samples_in/all_tazs.csv')
# links = gpd.read_file(r'trb2023/network.gpkg',layer='links')
# imp_links = gpd.read_file('trb2023/network.gpkg',layer='imp_links')
# nodes = gpd.read_file(r'trb2023/network.gpkg',layer='nodes')

# transitsim inputs



#get network nodes
ods = snap_ods_to_network(ods,nodes)

#filter ods
ods = ods[['trip_id','o_node','d_node']]
ods['tup'] = list(zip(ods.o_node,ods.d_node))

impedance_col = 'length'
ods, imp_links = find_shortest(imp_links, nodes, ods, impedance_col)

        

