# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 18:53:30 2023

@author: fizzyfan
"""

import os 
import time
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import osmnx as ox
import geopandas as gpd
from shapely.geometry import LineString
import geopy
from geopy.distance import geodesic


def obtain_bpoly(stops, node_stop_thres):
  # Format stops file
  stop_gis = stops.copy()
  stop_gis = gpd.GeoDataFrame(stop_gis, geometry = gpd.points_from_xy(stop_gis.stop_lon, stop_gis.stop_lat))
  # Create convex hull bounding for all stops
  stop_bound = pd.DataFrame({'geometry':[stop_gis.unary_union.convex_hull]})
  stop_bound = gpd.GeoDataFrame(stop_bound, geometry=stop_bound.geometry)
  stop_bound = stop_bound.set_crs('EPSG:4269')
  stop_bound = stop_bound.to_crs('EPSG:2163')
  stop_bound = gpd.GeoDataFrame(stop_bound, geometry=stop_bound.buffer(1609.344*node_stop_thres))
  stop_bound = stop_bound.to_crs('EPSG:4326')
  return(stop_bound)
  
  
  
def find_nearby_stops(df_stops, walk_thres):
  print(' - Looking for nearby stops...')
  ## First find stops that are within walkable distance (0.5 mile)
  # Make stations geospatial file
  df_stops = gpd.GeoDataFrame(df_stops, 
                              geometry=gpd.points_from_xy(df_stops.stop_lon, df_stops.stop_lat),
                              crs = 'EPSG:4269')
  df_stops = df_stops.to_crs('EPSG:2163')
  # Create 0.5 mile buffer for stops
  df_stops_buff = df_stops.buffer(1609.344*walk_thres)
  df_stops_buff = gpd.GeoDataFrame(df_stops, geometry = df_stops_buff)
  # Find stops that are walkable to each other 
  df_pairs = gpd.sjoin(df_stops, df_stops_buff, how='inner', predicate='within')
  # Filter out all those that are to itself
  df_pairs = df_pairs[df_pairs['stop_id_left'] != df_pairs['stop_id_right']]
  df_pairs = df_pairs.rename(columns={'stop_id_left':'stop1', 'stop_id_right':'stop2'})
  return(df_pairs)
  
  
  
def filter_same_route(df_trips, df_stop_times, df_pairs):
  print(' - Filtering out stops that are on the same routes...')
  ## Find stops that are on the same route (no transfer needed)
  # Find routes associated with each stop
  df_trips = df_trips[['route_id', 'trip_id']]
  df_trips = df_trips.merge(df_stop_times[['trip_id','stop_id']].drop_duplicates(), how='left', on='trip_id')
  df_trips = df_trips[['route_id', 'stop_id']].drop_duplicates()
  #
  route_ref = pd.DataFrame(df_trips.groupby('stop_id')['route_id'].unique())
  route_ref.reset_index(inplace=True)
  # Join this information to the stops pairs DF, and keep only those that are not on the same route in any circumstances
  route_ref['stop_id'] = route_ref['stop_id'].astype(type(df_pairs['stop1'].iloc[0]))
  df_pairs = (df_pairs
              .merge(route_ref
                      .rename(columns={'stop_id':'stop1','route_id':'route1'}), 
                      how='inner', on='stop1')
              .merge(route_ref
                      .rename(columns={'stop_id':'stop2','route_id':'route2'}), 
                      how='inner', on='stop2'))
  df_pairs['same_route'] = df_pairs.apply(lambda x: sum([y in x.route1 for y in x.route2]) > 0, axis=1)
  df_pairs = df_pairs[df_pairs['same_route'] == False]
  return(df_pairs)
  
  
  
def filter_nodes_by_stop(df_stops, nodes):
  #
  print(' - Filtering down nodes...')
  # Format stops
  df_stops = gpd.GeoDataFrame(df_stops, 
                              geometry=gpd.points_from_xy(df_stops.stop_lon, df_stops.stop_lat),
                              crs = 'EPSG:4269')
  df_stops = df_stops.to_crs('EPSG:2163')
  # Format nodes
  nodes = nodes[['osmid','y','x']].rename(columns={'y':'lat','x':'lon'})
  nodes = gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes.lon, nodes.lat),
                           crs = 'EPSG:4269')
  nodes = nodes.to_crs('EPSG:2163')
  # Stop outline
  stop_outline = df_stops.dissolve()
  stop_outline = gpd.GeoDataFrame(stop_outline, geometry=stop_outline.buffer(1609.344*node_stop_thres))
  # Filter node
  nodes = gpd.sjoin(nodes, stop_outline, how='left', predicate='within')
  nodes = nodes[~nodes['stop_id'].isnull()]
  #
  print(' - Generating stop-node pairs...')
  # Format nodes
  nodes = gpd.GeoDataFrame(nodes, geometry=nodes.buffer(1609.344*node_stop_thres))
  nodes = nodes[['osmid','lat','lon','geometry']]
  # Join nodes to stops
  stop_node = gpd.sjoin(df_stops, nodes, how='left', predicate='within')
  stop_node = stop_node[~stop_node['osmid'].isnull()]
  stop_node = stop_node[['stop_id','stop_lat','stop_lon','osmid','lat','lon']]
  #
  print(' - Looking for stop-node pair with the shortest distance...')
  # Looking for stop-node pair with the shortest distance
  stop_node['dist'] = stop_node.apply(lambda x: geopy.distance.geodesic((x.stop_lat, x.stop_lon),
                                                                        (x.lat, x.lon)).miles, 
                                      axis=1)
  #
  stop_node.reset_index(drop = True, inplace = True)
  stop_node = stop_node.loc[stop_node.groupby('stop_id')['dist'].idxmin()]
  stop_node = stop_node[['stop_id', 'osmid']]
  return(stop_node)
  
  
  
def shortest_path_finder(samp_in, links):
    if samp_in.shape[0] == 0:
        return(None)
    # Build graphs
    DG_dist = nx.DiGraph()
    for ind, row in links.iterrows():
        DG_dist.add_weighted_edges_from([(row['from'], row['to'], row['length'])])
    # Find shortest paths
    samp_in['dist'] = 0
    samp_in['path'] = 'nan'
    #
    for ind, row in tqdm(samp_in.iterrows(), total=samp_in.shape[0]):
        #
        source = row.node1
        target = row.node2
        #
        try:
            samp_in.loc[ind, 'dist'] = nx.shortest_path_length(DG_dist, source, target, 
                                                               weight='weight')
            samp_in.loc[ind, 'path'] = '__'.join([str(x) for x in 
                                                  nx.shortest_path(DG_dist, source, target, 
                                                                   weight='weight')])
        except:
            samp_in.loc[ind, 'dist'] = 99999
            samp_in.loc[ind, 'path'] = 'Error, node not in network'
    return(samp_in)
    
    
    
def get_transfer(FILE_PATH: str, 
                 walk_threshold: float = 0.21, 
                 network_prep: bool = False, 
                 breaker: str = '---', 
                 data: set = None) -> pd.DataFrame:
    """
    Create transfer pairs with transitive closure feature

    Args:
        FILE_PATH (str): path to all the files, both for both import (stops, trips, stop_times, nodes, edges) and export (transfer).
        walk_threshold (float): maximum allowable walk transfer distance in mile. Default is 0.21 mile, or 5 minutes walk at 2.5 mph. 
        network_prep (bool): whether the network has been prepared. If False, the function will prepare the network based on the stops location. Default is False.
        breaker (str): a string separating diffrent parts of the function. Default is '---'.
        data (set): input data, if provided, the data will be used directly in the function instead of import from FILE_PATH. Default is None.

    Returns:
        out (pd.DataFrame): pandas dataframe of transfer distance in miles, columns: stop1, stop2, dist.

    Examples:
        >>> transfers = get_transfer(FILE_PATH, 0.1)
        >>> transfers = get_transfer(FILE_PATH, 0.1, False, data = (stops, trips, stop_times))
        >>> transfers = get_transfer(FILE_PATH, 0.1, True, data = (stops, trips, stop_times, nodes, edges))
    """
    t0 = time.time()
    
    ## Import files
    t0_import = time.time()
    if data == None:
        stops = pd.read_csv(os.path.join(FILE_PATH, 'stops.txt'))
        trips = pd.read_csv(os.path.join(FILE_PATH, 'trips.txt'))
        stop_times = pd.read_csv(os.path.join(FILE_PATH, 'stop_times.txt'))
    else:
        stops = data[0]
        trips = data[1]
        stop_times = data[2]
    print('Time used for importing files: ', round((time.time()-t0_import)/60, 2), ' minutes.')
    print(breaker)

    ## Prepare the network, if it hasn't been done already
    t0_network = time.time()
    if not network_prep:
        print(' - preparing network...')
        # Find the appropriate bounding polygon for the stops
        stop_bound = obtain_bpoly(stops, node_stop_thres)
        # Grab network from OSM
        network = ox.graph_from_polygon(stop_bound['geometry'][0], network_type='walk')
        # Save
        ox.save_graph_shapefile(network, filepath=FILE_PATH)
        nodes = gpd.read_file(os.path.join(FILE_PATH, 'nodes.shp'))
        edges = gpd.read_file(os.path.join(FILE_PATH, 'edges.shp'))
    elif data == None:
        nodes = gpd.read_file(os.path.join(FILE_PATH, 'nodes.shp'))
        edges = gpd.read_file(os.path.join(FILE_PATH, 'edges.shp'))
    else:
        nodes = data[3]
        edges = data[4]
    print('Time used for preparing network: ', round((time.time()-t0_network)/60, 2), ' minutes.')
    print(breaker)
    
    ## Obtain possible transfer pairs
    t0_pairs = time.time()
    pairs = find_nearby_stops(stops, filter_thres)
    pairs = filter_same_route(trips, stop_times, pairs)
    print('Time used for finding possible transfer pairs: ', round((time.time()-t0_pairs)/60, 2), ' minutes.')
    print(breaker)
    
    ## Find corresponding stops and nodes
    t0_stop_node = time.time()
    stop_node = filter_nodes_by_stop(stops, nodes)
    print('Time used for stops and nodes correspondence: ', round((time.time()-t0_stop_node)/60, 2), ' minutes.')
    print(breaker)
    
    ## A bit of formatting
    t0_formatting = time.time()
    stop_node.reset_index(drop=True, inplace=True)
    #
    pairs['index'] = range(0,pairs.shape[0])
    pairs = pairs[['index','stop1','stop2']]
    pairs = pairs.merge(stop_node.rename(columns={'stop_id':'stop1', 'osmid':'node1'}), how='left', on='stop1')
    pairs = pairs.merge(stop_node.rename(columns={'stop_id':'stop2', 'osmid':'node2'}), how='left', on='stop2')
    #
    edges = edges[['from','to','length']] 
    # length is in meters
    # Since it's walk links, assuming everything reversable
    edges_rev = edges.copy()
    edges_rev['from'] = edges['to']
    edges_rev['to'] = edges['from']
    edges = pd.concat([edges, edges_rev])
    edges = edges.groupby(['from','to']).mean().reset_index()
    print('Time used for formatting: ', round((time.time()-t0_formatting)/60, 2), ' minutes.')
    print(breaker)
    
    ## Create the transfer paths
    t0_transfer = time.time()
    transfers = shortest_path_finder(pairs, edges)
    print('Time used for creating transfer: ', round((time.time()-t0_transfer)/60, 2), ' minutes.')
    print(breaker)
    
    ## create network graph
    t0_transclos = time.time()
    time_start = time.time()
    DGo = nx.Graph()  # create directed graph
    for ind, row in transfers_anls.iterrows():
        DGo.add_weighted_edges_from([(str(row['stop1']), str(row['stop2']), float(row['dist']))],weight='distance')   
    # get transitive closure of graph
    DGofinal = nx.transitive_closure(DGo,reflexive=None)
    # get edge list
    transfer_output = nx.to_pandas_edgelist(DGofinal)
    # rename columns
    transfer_output.columns = ['stop1','stop2','dist'] # dist in meters
    transfer_output['dist'] = transfer_output['dist']/1609.344 # convert to miles
    print('Time used for transitive closure: ', round((time.time()-t0_transclos)/60, 2), ' minutes.')
    print(breaker)
    
    print(f'TOTAL time used for {filter_thres} miles transfer: {round(((time.time() - t0)/60), 2)} minutes')
    return(transfer_output)
    
    


SAVE_PATH = r'fizzyfan/projects/transitAlg/data/gtfs'
breaker = '----------------------------------------'
transfers = get_transfer(SAVE_PATH, 0.1, True, breaker=breaker)



