# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:38:29 2022

@author: tpassmore6
"""
from pathlib import Path
import os
import partridge as ptg
import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
import time
from itertools import combinations, chain, permutations, product
from tqdm import tqdm
from datetime import datetime
import pickle
from datetime import date
from shapely.ops import MultiLineString

from helper_functions import *

pd.options.mode.chained_assignment = None  # default='warn'

#creates the study area
def create_study_area(settings):
    #read the transit stops file and create a polygon to show study area
    transit_stops = pd.read_csv(settings['gtfs_fp'] / 'stops.txt')   
    transit_stops.loc[:,'geometry'] = gpd.points_from_xy(transit_stops['stop_lon'],transit_stops['stop_lat'])
    transit_stops = gpd.GeoDataFrame(transit_stops,geometry='geometry',crs='epsg:4326')
    transit_stops.to_crs(crs=settings['crs'],inplace=True)
    
    #buffer all transit_stops by the biking threshold
    transit_stops.geometry = transit_stops.buffer(settings['thresh'])
    study_area = transit_stops.dissolve()
    study_area.to_file(settings['output_fp'] / 'base_layers.gpkg',layer='studyarea',driver='GPKG')
    
    # find TAZs within the specified study area
    tazs = gpd.read_file(settings['tazs_fp'])
    
    #make sure the key id is a string  
    tazs[settings['keyid']] = tazs[settings['keyid']].astype(str)

    #turn to the taz polygons to centroids
    centroids = tazs.copy()
    centroids.to_crs(settings['crs'],inplace=True)
    centroids['geometry'] = centroids.centroid
    centroids.set_geometry('geometry',inplace=True)
    centroids = centroids[[settings['keyid'],'geometry']]

    #intersect with marta study area
    centroids_cols = list(centroids.columns)
    centroids = gpd.overlay(centroids,study_area,how='intersection')[centroids_cols]

    #export for map making
    centroids.to_file(settings['output_fp'] / 'base_layers.gpkg',layer='centroids',driver='GPKG')

    #also export taz polygons
    tazs = tazs[tazs[settings['keyid']].isin(centroids[settings['keyid']])]
    tazs.to_crs(settings['crs'],inplace=True)
    tazs = tazs[[settings['keyid'],'geometry']]
    tazs.to_file(settings['output_fp'] / 'base_layers.gpkg',layer='tazs',driver='GPKG')

#create gtfs shapefile for examining and mapping raptor outputs
def add_new_routeid(settings):

    #get the new route ids from transit-routing and match to old
    route = pd.read_csv(settings['gtfs_fp'] / 'route.txt')[['new_route_id','route_type','route_short_name','route_long_name']].drop_duplicates()
    route.rename(columns={'new_route_id':'route_id'},inplace=True)
    route['route_id'] = route['route_id'].astype(str)
    
    #import stops for stop_id and stop geometry
    stops = pd.read_csv(settings['gtfs_fp'] / 'stops.txt')
    stops['stop_id'] = stops['stop_id'].astype(str)
    stops = gpd.GeoDataFrame(stops, geometry=gpd.points_from_xy(stops['stop_lon'], stops['stop_lat']), crs='epsg:4326')
    stops.to_crs(settings['crs'],inplace=True)
    stops = stops[['stop_id','geometry']]
    
    #import trips for trip_id and route_id
    trips = pd.read_csv(settings['gtfs_fp'] / 'trips.txt')[['trip_id','route_id']]
    trips['route_id'] = trips['route_id'].astype(str)
    #bring in stop times for trip_id and stop_id
    stop_times = pd.read_csv(settings['gtfs_fp'] / 'stop_times.txt')[['trip_id','stop_id']]
    stop_times['stop_id'] = stop_times['stop_id'].astype(str)
    #merge to get route_id and stop_id
    route_and_stop = pd.merge(trips,stop_times,on='trip_id')[['route_id','stop_id']].drop_duplicates()
    #merge to get route type (rail = 1, bus = 3)
    route_and_stop = route_and_stop.merge(route,on='route_id')
    #merge to add geometry
    route_and_stop = route_and_stop.merge(stops,on='stop_id')
    #drop street car service (0)
    route_and_stop = route_and_stop[route_and_stop['route_type']!=0]
    #make sure it's a geodataframe
    route_and_stop = gpd.GeoDataFrame(route_and_stop,geometry='geometry',crs=settings['crs'])    
    #export
    route_and_stop.to_file(settings['output_fp'] / 'base_layers.gpkg',layer='route_and_stop',driver='GPKG')

def gtfs_shapefiles(settings):

    # use partridge to create shapefiles for mapping
    _date = settings['service_date']
    service_ids = ptg.read_service_ids_by_date(settings['gtfs_zip'])
    service_ids = service_ids[_date]
    view = {'trips.txt': {'service_id': service_ids}}
    feed = ptg.load_geo_feed(settings['gtfs_zip'],view)
    
    #get the new route ids
    route = pd.read_csv(settings['gtfs_fp'] / 'route.txt')
    route.route_id = route.route_id.astype(str)
    
    trips = feed.trips[['route_id','shape_id']]
    shapes = feed.shapes
    
    #merge route and trips to get shapeid then merge to get shapes
    transit_network = route.merge(trips,on='route_id').merge(shapes,on='shape_id').drop_duplicates()
    
    #export transit shapes for examining purposes
    transit_network.drop(columns=['Unnamed: 0'],inplace=True)
    transit_network = gpd.GeoDataFrame(transit_network,geometry='geometry')
    transit_network.to_crs(settings['crs'],inplace=True)
    transit_network.to_file(settings['output_fp'] / 'base_layers.gpkg',layer='transit_shapes')

    #export the stops too
    stops = feed.stops
    stops.to_crs(settings['crs'],inplace=True)
    stops.to_file(settings['output_fp'] / 'base_layers.gpkg',layer='transit_stops')

def process_studyarea(settings):
    create_study_area(settings)
    add_new_routeid(settings)
    gtfs_shapefiles(settings)
    #write_network_to_base_layers(settings)


def get_candidate_stops(centroids,route_and_stop,thresh,settings):
    '''
    This function takes in all the origins/destinations and matches them to marta stops within the specified threshold. Only unique routes are matched.
    '''

    #initialize dict for candidate stops for each TAZ
    candidate_stops_by_taz = pd.DataFrame()    

    #for each taz, calculate the euclidean distance to transit stops
    for idx, row in centroids.iterrows():
        #make copy of stop_and_route dataframe
        candidate_stops = route_and_stop.copy()
        #calculate distance to all transit stops from centroid
        candidate_stops['distance'] = candidate_stops.distance(row.geometry)
        #knockout those beyond biking threshold
        candidate_stops = candidate_stops[candidate_stops['distance'] < thresh]
        #reset index
        candidate_stops.reset_index(drop=True,inplace=True)
        
        #only keep the closest route stop
        mask = candidate_stops.groupby('route_id')['distance'].idxmin().to_list()
        candidate_stops = candidate_stops.loc[mask]
        
        if len(candidate_stops) > 0:
            #add taz_id
            candidate_stops[settings['keyid']] = row[settings['keyid']]
            #append candidate stops
            candidate_stops_by_taz = pd.concat([candidate_stops_by_taz,candidate_stops],ignore_index=True).reset_index(drop=True)
    
    #drop distance and extra stops when when routes share the same stop
    candidate_stops_by_taz.drop(columns=['distance','route_id','route_type'],inplace=True)
    candidate_stops_by_taz.drop_duplicates(inplace=True)
    
    #average number of candidate stops
    avg_candidate  = round(candidate_stops_by_taz.groupby(settings['keyid'])['stop_id'].count().mean(),0)
    print(f'{avg_candidate} candidate stops with unique routes per TAZ on average')
    
    return candidate_stops_by_taz


def snap_to_network(to_snap,network_nodes_raw):
    '''
    Snap transit stops or origins/destinations to nearest network node
    '''
    
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
    snapped_nodes = ckdnearest(to_snap, network_nodes)

    #filter columns
    snapped_nodes = snapped_nodes[to_snap.columns.to_list()+['N','dist']]
        
    #drop geo column
    snapped_nodes.drop(columns=['original'],inplace=True)
    
    print(f'snapping took {round(((time.time() - time_start)/60), 2)} minutes')
    return snapped_nodes


def tazToTaz(links,nodes,selected_tazs,tazs,impedance,settings,allow_wrongway,thresh):
    '''
    This function performs shortest path routing between selected tazs to all other tazs.

    Used to get bike or walk only travel time to destination

    TODO Add feature for just getting travel time instead of doing the max threshold distance
    '''

    
    print(f'Finding shortest paths between TAZs using {impedance} impedance...')
    
    #record the starting time
    time_start = time.time()
    
    # if allow_wrongway:
    #     #make sure there are duplicate links for all
    #     #TODO fix this
    
    if isinstance(allow_wrongway,tuple):
        allow_wrongway = False
    
    #create weighted network graph
    if allow_wrongway:
        DGo = create_walk_graph(links,impedance)
    else:
        DGo = create_bike_graph(links,impedance)
    
    #initialize empty dicts
    all_impedances = {}
    all_nodes = {}
    all_paths = {}
    
    selected_tazs = tazs[tazs[settings['keyid']].isin(selected_tazs)]
    
    #if a tuple is provided use the higher threshold
    if isinstance(thresh,tuple):
        thresh1 = max(thresh)
    else:
        thresh1 = thresh
    
    #use 2x the threshold to get more tazs to compare against transit travel time
    #thresh1 = thresh1 * 2

    #from each unique origin
    for taz in tqdm(selected_tazs.tazN.unique()):
        #run dijkstra's algorithm
        #the cutoff gets a little weird for other impedances
        impedances, paths = nx.single_source_dijkstra(DGo,taz,weight=impedance,cutoff=thresh1)
        
        # #remove self
        # impedances.pop(taz)
        # paths.pop(taz)

        #filter dijkstra results
        for key in impedances.keys():

            #for each trip id find impedance
            all_impedances[(taz,key)] = impedances[key]
          
            #convert from node list to edge list
            node_list = paths[key]
            #edge_list = [node_list[i]+'_'+node_list[i+1] for i in range(len(node_list)-1)]
            edge_list = [(node_list[i],node_list[i+1]) for i in range(len(node_list)-1)]

            #store
            all_nodes[(taz,key)] = node_list
            all_paths[(taz,key)] = edge_list

    #print(f'Shortest path routing took {round(((time.time() - time_start)/60), 2)} minutes')
    return all_paths, all_impedances
    
def tazToFromTransit(links,nodes,candidate_stops_by_taz,impedance,allow_wrongway,thresh):
    '''
    Finds shortest path from taz to transit and transit to taz
    '''
    
    print(f'Finding shortest paths to and from TAZs/transit using {impedance} impedance...')
    
    #make copy
    stops_and_tazs = candidate_stops_by_taz.copy()
    
    #record the starting time
    time_start = time.time()
    
    #create weighted network graph
    if isinstance(allow_wrongway,tuple):
        if allow_wrongway[0]:
            DG1 = create_walk_graph(links,impedance)
        else:
            DG1 = create_bike_graph(links,impedance)
        if allow_wrongway[1]:
            DG2 = create_walk_graph(links,impedance)
        else:
            DG2 = create_bike_graph(links,impedance)
    elif allow_wrongway:
        DG1 = create_walk_graph(links,impedance)
        DG2 = create_walk_graph(links,impedance)
    else:
        DG1 = create_bike_graph(links,impedance)       
        DG2 = create_bike_graph(links,impedance)
    
    #initialize empty dicts
    all_impedances = {}
    all_nodes = {}
    all_paths = {}
    
    #listcheck
    stops_and_tazs['tup'] = list(zip(stops_and_tazs['tazN'],stops_and_tazs['stopsN']))
    listcheck = set(stops_and_tazs['tup'].to_list())
    
    # if two thresholds, then create two variables
    if isinstance(thresh,tuple):
        thresh1 = thresh[0]
        thresh2 = thresh[1]
    else:
        thresh1 = thresh
        thresh2 = thresh
    
    #from each unique origin
    print('Routing from TAZs to transit stops...')
    for taz in tqdm(stops_and_tazs['tazN'].unique()):
        #run dijkstra's algorithm
        #the cutoff gets a little weird for other impedances
        impedances, paths = nx.single_source_dijkstra(DG1,taz,weight=impedance,cutoff=thresh1)
        
        # #remove self
        # impedances.pop(taz)
        # paths.pop(taz)

        #filter dijkstra results
        for key in impedances.keys():
            #check if trip is in one of the ones we want
            if (taz,key) in listcheck:
                #for each trip id find impedance
                all_impedances[(taz,key)] = impedances[key]
              
                #convert from node list to edge list
                node_list = paths[key]
                edge_list = [(node_list[i],node_list[i+1]) for i in range(len(node_list)-1)]
                
                #store
                all_nodes[(taz,key)] = node_list
                all_paths[(taz,key)] = edge_list

    #add impedance column to ods dataframe
    stops_and_tazs[f'{impedance}_to'] = stops_and_tazs['tup'].map(all_impedances)

    #from each unique transit stop
    print('Routing from transit stops to TAZs...')
    for stop in tqdm(stops_and_tazs['stopsN'].unique()):
        #run dijkstra's algorithm
        impedances, paths = nx.single_source_dijkstra(DG2,stop,weight=impedance,cutoff=thresh2)
        
        # #remove self
        # impedances.pop(stop)
        # paths.pop(stop)

        #filter dijkstra results
        for key in impedances.keys():
            #check if trip is in one of the ones we want
            if (key,stop) in listcheck:
                #for each trip id find impedance
                all_impedances[(stop,key)] = impedances[key]
              
                #convert from node list to edge list
                node_list = paths[key]
                edge_list = [(node_list[i],node_list[i+1]) for i in range(len(node_list)-1)]
                
                #store
                all_nodes[(stop,key)] = node_list
                all_paths[(stop,key)] = edge_list

    #print(f'... took {round(((time.time() - time_start)/60), 2)} minutes')

    #add impedance column to ods dataframe
    stops_and_tazs[f'{impedance}_from'] = stops_and_tazs['tup'].map(all_impedances)
    
    #drop if both are na
    stops_and_tazs.dropna(subset=[f'{impedance}_from',f'{impedance}_to'],how='all',inplace=True)

    return stops_and_tazs, all_paths

def add_lengths(bikepaths,all_paths,links):
    #print('Getting path lengths')
    #make sure there is a length column
    links['length'] = links.length

    #turn into dict
    lengths_dict = links.set_index(['A','B']).length.to_dict()
    #lengths_dict = dict(zip(links['A_B'],links['length']))
    
    all_lengths = {}
    for key in all_paths.keys():
        id_list = all_paths[key]
        length_list = [lengths_dict.get(id,0) for id in id_list]
        all_lengths[key] = sum(length_list)

    #add length to bikepaths
    bikepaths['tup_to'] = list(zip(bikepaths['tazN'],bikepaths['stopsN']))
    bikepaths['tup_from'] = list(zip(bikepaths['stopsN'],bikepaths['tazN']))
    bikepaths['length_to'] = bikepaths['tup_to'].map(all_lengths)
    bikepaths['length_from'] = bikepaths['tup_from'].map(all_lengths)
    bikepaths.drop(columns=['tup_to','tup_from'])

    return bikepaths

#turns edge list to multilinestirng
def add_geo(all_paths,links):
    #print('Adding path geo')
    
    #create duplicate links to deal with wrong way
    links_dup = links.copy()
    links_dup.rename(columns={'A':'B','B':'A'},inplace=True)
    #links_dup['A_B'] = links_dup['A'] + '_' + links_dup['B']

    #add back
    links = pd.concat([links,links_dup],ignore_index=True).reset_index(drop=True)

    #turn into dict
    geos_dict = links.set_index(['A','B']).geometry.to_dict()
    #geos_dict = dict(zip(links['A_B'],links['geometry']))
    
    all_geos = {}
    for key in all_paths.keys():
        id_list = all_paths[key]
        geos_list = [geos_dict.get(id,0) for id in id_list]
        if geos_list != []:
            all_geos[key] = MultiLineString(geos_list)
    return all_geos

def create_raptor_inputs(select_tazs,bikepaths,impedance,exclude,mode,max_thresh,settings,ods=None,rail_start=False):
    '''
    Generate input files for raptor. If not specified, it will do all possible combinations of tazs.

    If using ods to filter the dataset, make sure the ods dataframe has an 'origin' and 'destination' column (it's hard-coded)

    Returns dataframe (from one select taz) with columns: source_taz, dest_taz, first_stop, last_stop, first_leg, last_leg
    '''

    for centroid in select_tazs: 
        #one origin
        origin = bikepaths[bikepaths[settings['keyid']] == centroid]
        #create tups
        origin = list(zip(
            origin[settings['keyid']],
            origin['stop_id'],
            origin[f'{impedance}_to'],
            origin['dist_to']
            ))
        
        if rail_start:
            #import route and stop
            route_and_stop = gpd.read_file(settings['output_fp'] / 'base_layers.gpkg', layer = 'route_and_stop')
            rail_stops = route_and_stop.loc[route_and_stop['route_type']==1,['stop_id','geometry']]
            rail_stops.geometry = rail_stops.buffer(500)
            route_and_stop = set(gpd.overlay(route_and_stop,rail_stops[['geometry']], how='intersection')['stop_id'].tolist())
            origin = [x for x in origin if x[1] in route_and_stop]     

        if ods is not None:       
            #drop duplicates
            ods = ods[['origin','destination']]
            
            #find ods with the right origin
            dest_tazs = ods.loc[ods['origin'] == centroid,'destination'].tolist()
            
            #remove ods that aren't present in trip file
            dest = bikepaths[bikepaths[settings['keyid']].isin(dest_tazs)]
        else:
            #do all possible destination tazs
            dest = bikepaths[bikepaths[settings['keyid']] != centroid]
        
        #creat tup
        dest = list(zip(
            dest[settings['keyid']],
            dest['stop_id'],
            dest[f'{impedance}_from'],
            dest['dist_from']
            ))
    
        #get all possible combinations that aren't in the exclude list or exceed 2x the biking threshold
        all_pairs = [(x[0][0],x[1][0],x[0][1],x[1][1],x[0][2],x[1][2],x[0][3],x[1][3]) for x in list(product(origin,dest)) if ((x[0][0],x[1][0]) not in exclude) & (x[0][2]+x[1][2]<max_thresh)]
        
        #this ignores the excluded pairs (so tazs that are close by are still calculated will take longer)
        #all_pairs = [(x[0][0],x[1][0],x[0][1],x[1][1],x[0][2],x[1][2],x[0][3],x[1][3]) for x in list(product(origin,dest)) if (x[0][2]+x[1][2]<max_thresh)]

        #turn to dataframe
        all_pairs = pd.DataFrame.from_records(all_pairs, columns=[
            'src_taz','dest_taz','src_stop','dest_stop',f'{impedance}_first_leg',f'{impedance}_last_leg','first_leg','last_leg'])
        
        #drop duplicates (and figure out why it's making duplicates)
        all_pairs.drop_duplicates(inplace=True)

        #create new folder
        if not (settings['output_fp'] / f'{mode}_{impedance}/trips').exists():
            (settings['output_fp'] / f'{mode}_{impedance}/trips').mkdir()
        
        #export
        all_pairs.to_parquet(settings['output_fp'] / f'{mode}_{impedance}/trips/{centroid}.parquet',engine='fastparquet')

#export walkable/bikeable centroids and polygons
def bikewalksheds(settings,centroids,all_impedances,mode,impedance):
    '''
    This function takes in a dict of shortest path routes where the keys are used to create
    points and polygons gpkg layer that shows all the bikeable/walkable tazs. Outputs a df
    used for excluding certain TAZ pairings from the RAPTOR input file generation
    '''

    #turn dict keys into dataframe with two columns
    o = []
    d = []
    for x in all_impedances.keys():
        o.append(x[0])
        d.append(x[1])
    df = pd.DataFrame({'origin':o,'dest':d})
    #group by taz origin to get all bikable/walkable tazs for origin tazN
    test = df.groupby('origin')['dest'].agg(set)
    
    #initialize list of tazs to exclude
    exclude = pd.DataFrame()
    
    #add taz geometry
    for idx,val in test.items():
        source = centroids.loc[centroids['tazN'] == idx,:]
        all_bikeable = centroids.loc[centroids['tazN'].isin(list(val)),:]
    
        #get taz name
        taz_name = source.iloc[0,0]
        
        #create new folders
        if not (settings['output_fp'] / f'{mode}_{impedance}').exists():
            (settings['output_fp'] / f'{mode}_{impedance}').mkdir() 

        if not (settings['output_fp'] / f'{mode}_{impedance}/visuals').exists():
            (settings['output_fp'] / f'{mode}_{impedance}/visuals').mkdir() 
        
        #export source centroid
        source.to_file(settings['output_fp'] / f'{mode}_{impedance}/visuals/{taz_name}.gpkg',layer='source_taz')
        
        #export bikable tazs as centroids
        all_bikeable.to_file(settings['output_fp'] / f'{mode}_{impedance}/visuals/{taz_name}.gpkg',layer=f'{mode}able_tazs_centroids')

        # #get bounds for dot plot map
        # bounds = all_bikeable.copy()
        # bounds_dissolved = bounds.dissolve()
        # bounds_dissolved.set_geometry(bounds_dissolved.convex_hull,inplace=True)
        # bounds_dissolved.to_file(settings['output_fp'] / f'{mode}_{impedance}/visuals/{taz_name}.gpkg',layer=f'{mode}_bounds')
        
        #use polygons instead
        tazs = gpd.read_file(settings['output_fp'] / 'base_layers.gpkg',layer='tazs')
        tazs[settings['keyid']] = tazs[settings['keyid']].astype(str)
            
        #get polygons
        all_bikeable.drop(columns=['geometry'],inplace=True)
        all_bikeable = pd.merge(all_bikeable,tazs[[settings['keyid'],'geometry']],on=settings['keyid'])
        
        #export
        all_bikeable.to_file(settings['output_fp'] / f'{mode}_{impedance}/visuals/{taz_name}.gpkg',layer=f'{mode}able_tazs_polygons')
    
        df = pd.DataFrame({'dest_id':all_bikeable.loc[:,settings['keyid']]})
        df['ori_id'] = taz_name
        exclude = pd.concat([exclude,df],ignore_index=True,axis=0)
    
    exclude = set(zip(exclude['ori_id'],exclude['dest_id']))
            
    return exclude

def write_network_to_base_layers(settings):
    links = gpd.read_file(settings['network_fp'],layer=settings['links_layer'])
    nodes = gpd.read_file(settings['network_fp'],layer=settings['nodes_layer'])

    links.to_file(settings['output_fp'] / 'base_layers.gpkg',layer='links')
    nodes.to_file(settings['output_fp'] / 'base_layers.gpkg',layer='nodes')

def raptor_preprocessing(settings,mode_settings,select_tazs,ods=None):
    
    thresh = mode_settings['thresh']
    max_thresh = mode_settings['max_thresh']
    mode = mode_settings['mode']
    impedance = mode_settings['impedance']
    allow_wrongway = mode_settings['allow_wrongway']
    
    #load files
    links = gpd.read_file(settings['output_fp'] / 'base_layers.gpkg',layer='links')
    nodes = gpd.read_file(settings['output_fp'] / 'base_layers.gpkg',layer='nodes')
    
    candidate_stops_by_taz = pd.read_pickle(settings['output_fp'] / 'candidate_stops_by_taz.pkl')
    centroids = pd.read_pickle(settings['output_fp'] / 'centroids.pkl')

    print(f'Preprocessing {mode} + transit with {impedance} impedance')
    
    #find ODs that are too close
    all_paths, all_impedances = tazToTaz(links,nodes,select_tazs,centroids,impedance,settings,allow_wrongway=allow_wrongway,thresh=thresh)
    exclude = bikewalksheds(settings,centroids,all_impedances,mode,impedance)
    
    #export exclude and test without it
    with (settings['output_fp'] / f'{mode}_{impedance}/exclude.pkl').open('wb') as fh:
        pickle.dump(exclude,fh)
    
    #get paths and impedances from and to all tazs and transit stops
    bikepaths, all_paths = tazToFromTransit(links, nodes, candidate_stops_by_taz, impedance=impedance, allow_wrongway=allow_wrongway,thresh=thresh)
    
    #TODO this can be consolidate
    #get geometric length of paths and retrieve geometries from street network
    bikepaths = add_lengths(bikepaths,all_paths,links)
    all_paths_geo = add_geo(all_paths,links)

    #export bikepaths geometry
    with (settings['output_fp'] / f'{mode}_{impedance}/all_paths.pkl').open('wb') as fh:
        pickle.dump(all_paths,fh)

    with (settings['output_fp'] / f'{mode}_{impedance}/all_paths_geo.pkl').open('wb') as fh:
        pickle.dump(all_paths_geo,fh)

    with (settings['output_fp'] / f'{mode}_{impedance}/bikepaths.pkl').open('wb') as fh:
        pickle.dump(bikepaths,fh)
    
    #create raptor inputs
    create_raptor_inputs(select_tazs,bikepaths,impedance,exclude,mode,max_thresh,settings,ods,mode_settings['rail_start'])


def candidate_stops(settings):
    
    #load files
    nodes = gpd.read_file(settings['output_fp'] / 'base_layers.gpkg',layer='nodes')
    centroids = gpd.read_file(settings['output_fp'] / 'base_layers.gpkg',layer='centroids',driver='GPKG')

    #bring in stops with route ids
    route_and_stop = gpd.read_file(settings['output_fp'] / 'base_layers.gpkg',layer='route_and_stop',driver='GPKG')

    #find candidate stops for each taz
    candidate_stops_by_taz = get_candidate_stops(centroids,route_and_stop,settings['thresh'],settings)
    
    #drop duplicates to simplify snapping
    route_and_stop = route_and_stop[['stop_id','geometry']].drop_duplicates()

    # snapping
    #TODO remove snapped values if they exceed threshold
    snapped_tazs = snap_to_network(centroids,nodes)
    snapped_stops = snap_to_network(route_and_stop,nodes)

    #rename columns  
    snapped_tazs.rename(columns={'N':'tazN','dist':'taz_snapdist'},inplace=True)
    snapped_stops.rename(columns={'N':'stopsN','dist':'stops_snapdist'},inplace=True)

    # turn into dict for mapping and export as pickle
    snapped_tazs_dict = dict(zip(snapped_tazs[settings['keyid']],snapped_tazs['tazN']))
    snapped_stops_dict = dict(zip(snapped_stops['stop_id'],snapped_stops['stopsN']))

    # export snapped tazs and transit stops
    with open(settings['output_fp'] / 'snapped_tazs.pkl','wb') as fh:
        pickle.dump(snapped_tazs_dict,fh)
    with open(settings['output_fp'] / 'snapped_stops.pkl','wb') as fh:
        pickle.dump(snapped_stops_dict,fh)

    #add snapped nodes back into dataframe
    candidate_stops_by_taz = candidate_stops_by_taz.merge(snapped_tazs[[settings['keyid'],'tazN','taz_snapdist']],on=settings['keyid']).merge(
        snapped_stops[['stop_id','stopsN','stops_snapdist']],on=['stop_id'])

    #bring back in
    centroids = gpd.read_file(settings['output_fp'] / 'base_layers.gpkg',layer='centroids',driver='GPKG')

    #set impedance column and select tazs
    centroids['tazN'] = centroids[settings['keyid']].map(snapped_tazs_dict)

    #export as pickle
    candidate_stops_by_taz.to_pickle(settings['output_fp'] / 'candidate_stops_by_taz.pkl')
    centroids.to_pickle(settings['output_fp'] / 'centroids.pkl')

    return candidate_stops_by_taz, centroids
