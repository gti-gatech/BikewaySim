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

from scipy.spatial import cKDTree


#### R

"""
Re-structuring this to account for the new process



"""




###### Helper Functions ######

#take in two geometry columns and find nearest gdB point from each
#point in gdA. Returns the matching distance too.
#MUST BE A PROJECTED COORDINATE SYSTEM
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

###### OD Snapping ######

def snap_ods_to_network(od_gdf:pd.DataFrame,nodes:gpd.GeoDataFrame,od_list=False):
    """
    This function takes in a GeoDataFrame of points/polygons used to represent
    origins or destinations and finds the nearest network node using Euclidean
    (cross-flies) distance. The distance and network node ID is then appended
    to the original GeoDataFrame.

    Additional Notes:
    - If provided geometry is a Polygon, the centroid will
    be used for distance calculations.
    - The distance and units is determined by the nodes CRS

    """
    
    # project to same CRS as nodes
    od_gdf = od_gdf.copy()
    od_gdf.to_crs(nodes.crs,inplace=True)

    # find centroids
    od_gdf['centroid_geo'] = od_gdf.geometry.centroid
    od_gdf.set_geometry('centroid_geo',inplace=True)

    # find closest node
    closest_node = ckdnearest(od_gdf, nodes)

    return closest_node


##### Shortest Path Routing ######







    
def create_graph(links,impedance_col):
    '''
    Creates weighted directed network graph
    '''
    
    DGo = nx.DiGraph()  # create directed graph
    for ind, row2 in links.iterrows():
        DGo.add_weighted_edges_from([(int(row2['A']), int(row2['B']), float(row2[impedance_col]))],weight=impedance_col)   
    
    return DGo
        
def find_shortest(links:gpd.GeoDataFrame,nodes:gpd.GeoDataFrame,ods_:pd.DataFrame,impedance_col:str):
    #record the starting time
    #time_start = time.time()
    
    ods = ods_.copy()

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
        impedances, paths = nx.single_source_dijkstra(DGo,origin,weight=impedance_col)    

        #iterate through dijkstra results to add them to ods dataframe
        for key in impedances.keys():
            #check if trip is in one of the ones we want
            if (origin,key) in listcheck:
                #for each trip id find impedance
                ods.loc[ods['tup']==(origin,key),impedance_col] = impedances[key]
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

    #print number that can't be routed
    print(f"{ods[impedance_col].isna().sum()} trips couldnt be routed")

    #simplify
    ods = ods[['trip_id','ori_id','dest_id',impedance_col,'length','geometry']]

    return ods, links, nodes

def btw_centrality(all_nodes:dict,all_paths:dict,links:gpd.GeoDataFrame,nodes:gpd.GeoDataFrame, impedance_col):
    '''
    Calculates link betweenness centrality
    '''

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
    nodes[f'{impedance_col}_pct_btw_cntrlty'] = nodes[f'{impedance_col}_btw_cntrlty'] / len(all_paths)#nodes[f'{impedance_col}_btw_cntrlty'].sum()
    links[f'{impedance_col}_pct_btw_cntrlty'] = links[f'{impedance_col}_btw_cntrlty'] / len(all_paths)#links[f'{impedance_col}_btw_cntrlty'].sum()

    return links, nodes

def add_geo(all_paths:dict,links:gpd.GeoDataFrame):
    '''
    Takes an edge list and returns a multilinestring of the entire trip for GIS
    '''

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
    imp = pd.merge(imp,dist[['trip_id','length']],on='trip_id',suffixes=('','_y'))

    #calculate pct detour
    imp['percent_detour'] = (imp['length'] - imp['length_y']) / imp['length_y'] * 100

    #get origin taz from trip id   
    imp['OBJECTID'] = imp['trip_id'].str.split('_',expand=True)[0]

    #groupby for viz
    by_taz = imp.groupby('OBJECTID')['percent_detour'].mean().reset_index()

    #add taz geo
    by_taz = pd.merge(by_taz,tazs,on='OBJECTID')

    #round
    by_taz['percent_detour'] = by_taz['percent_detour'].round(1)

    return by_taz

def impedance_change(imp,improved,tazs,impedance_col):
    '''
    Finds the percent detour and returns it as a column on the ods_imp gdf
    '''
    #drop the geometry cols
    imp = pd.merge(imp[['trip_id',impedance_col]],improved[['trip_id',impedance_col]],on='trip_id',suffixes=('','_y'))

    #calculate impedance change
    imp['imp_change'] = (imp[impedance_col] - imp[impedance_col+'_y'])

    #get origin taz from trip id   
    imp['OBJECTID'] = imp['trip_id'].str.split('_',expand=True)[0]

    #groupby for viz
    by_taz = imp.groupby('OBJECTID')['imp_change'].mean().reset_index()

    #add taz geo
    by_taz = pd.merge(by_taz,tazs,on='OBJECTID')

    return by_taz


def make_bikeshed(links_c,nodes,origin,radius,buffer_size,impedance_col):
    '''
    Get the bikeshed for an origin
    '''

    links = links_c.copy()

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
    
    print(f'---{origin}---')
    print(f'Bikeshed Network Miles: {np.round(df_dup.length.sum()/5280,1)}')
    print(f'Bikeshed Size (square miles w/{radius} ft access distance): {np.round(df_dup.buffer(buffer_size).area.sum()/5280/5280,1)}')    

    return bikeshed, bikeshed_node

def drop_duplicate_links(links):
    #drops the additional two way link needed for network routing
    df_dup = pd.DataFrame(np.sort(links[["A","B"]], axis=1), columns=["A","B"], index = links.index).duplicated()
    links = links[df_dup]
    return links