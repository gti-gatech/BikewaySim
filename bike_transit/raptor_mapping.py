# -*- coding: utf-8 -*-
"""
Updated on 3/8/23

@author: tpassmore6
"""
from pathlib import Path

#post processing specific
import geopandas as gpd
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from shapely.ops import LineString
from tqdm import tqdm
from datetime import datetime, timedelta, date

#suppress error message
# import warnings
# from shapely.errors import ShapelyDeprecationWarning
# warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 
# warnings.filterwarnings("ignore", category=FutureWarning)

from helper_functions import check_type, process_results, load_files

def map_routes(settings:dict,impedance:str,mode:str,select_tazs:list,time_limit:timedelta):
    '''
    Loops through each trip df outputed by RAPTOR
    and produces a GPKG file named after the origin
    TAZ and with a layer name corresponding to the 
    departure time. Each layer in a GPKG will be a
    collection of linestrings with corresponding to the
    bike/walk and transit portions of the journey.
    
    - Transit routes will have the route number and mode type.
    - Transfers will always be via walking
    - First and last leg will be by bike/walk unless
    TAZ/transit stop matched to the same network node
    '''

    for select_taz in select_tazs:
        #get filepaths for importing trips
        selected_trips = (settings['output_fp']).glob(f'{mode}_{impedance}/raptor_results/{select_taz}/*.pkl')

        #selected_trips = [Path.home() / 'Documents/TransitSimData/Data/bike_lts_dist/raptor_results/411/8_0.pkl']

        #bring in all_path_geo made in the find candidate stops code
        with (settings['output_fp'] / f'{mode}_{impedance}/all_paths_geo.pkl').open(mode='rb') as fh:
            all_paths = pickle.load(fh)

        #load files
        snapped_tazs, snapped_stops, shape_map, shapes, stops_file = load_files(settings)

        for selected_trip in selected_trips:
            
            #get the trip time
            trip_time = selected_trip.parts[-1].split('.pkl')[0]
            
            #load file
            trip_df = process_results(selected_trip)

            #remove trips longer than 1hr
            trip_df = trip_df[trip_df['travel_time'] <= time_limit]
                
            #add snapped network node to df
            trip_df['src_tazN'] = trip_df['src_taz'].map(snapped_tazs)
            trip_df['dest_tazN'] = trip_df['dest_taz'].map(snapped_tazs)
            trip_df['src_stopN'] = trip_df['src_stop'].map(snapped_stops)
            trip_df['dest_stopN'] = trip_df['dest_stop'].map(snapped_stops)
            
            #make sure it's str
            trip_df['src_tazN'] = trip_df['src_tazN'].astype(str)
            trip_df['dest_tazN'] = trip_df['dest_tazN'].astype(str)
            trip_df['src_stopN'] = trip_df['src_stopN'].astype(str)
            trip_df['dest_stopN'] = trip_df['dest_stopN'].astype(str)
            
            #create tups for first/last leg
            trip_df['first_leg'] = list(zip(trip_df['src_tazN'],trip_df['src_stopN']))
            trip_df['last_leg'] = list(zip(trip_df['dest_stopN'],trip_df['dest_tazN']))
            
            #add bike edge if available
            trip_df['bike_edge1'] = trip_df['first_leg'].map(all_paths)
            trip_df['bike_edge2'] = trip_df['last_leg'].map(all_paths)

            print(f"Mapping trips for mode {mode} and {impedance} impedance from origin taz {selected_trip.parts[-2]} at time {selected_trip.parts[-1].split('.pkl')[0]}")
            for idx, row in tqdm(trip_df.iterrows(),total=trip_df.shape[0]):
                process_transit_edges(row,trip_time,shapes,shape_map,stops_file,impedance,mode,settings)

        #return trip_df

def process_transit_edges(row,trip_time,shapes,shape_map,stops_file,impedance,mode,settings):

    #initialize empty lists for constructing dataframe                      
    transit_edges = []
    route_types = []
    route_ids = []
    names = []
    start_stop = []
    end_stop = []
    time_spent = []

    #get edge list
    edge_list = row['edge_list']

    #get start/end taz and transit stops
    src_taz = row['src_taz']
    dest_taz = row['dest_taz']
    src_stop = row['src_stop']
    dest_stop = row['dest_stop']

    #this loop creates the transit portions of the dataframe
    for edge in edge_list:        
        
        #get start station, end station, and tripid
        start = check_type(edge[0])
        end = check_type(edge[1])
        
        #append each to list
        start_stop.append(start)
        end_stop.append(end)
        time_spent.append(edge[2].total_seconds()/60)
        
        #get start and end stops points
        start_point = stops_file.loc[stops_file['stop_id'] == start,'geometry'].drop_duplicates().item()
        end_point = stops_file.loc[stops_file['stop_id'] == end,'geometry'].drop_duplicates().item()
        
        #get transit specific info
        if edge[-2] in ['rail','bus']:
            #get route name
            names.append(edge[-1])
            
            #extract trip_id
            trip_id = check_type(edge[3])

            #extract route_id
            route_id = str.split(trip_id,'_')[0]
            route_ids.append(route_id)
            
            #get transit type
            route_type = edge[-2]
            route_types.append(route_type)
            
            #get shape_id
            shape_id = shape_map[shape_map['new_route_id'] == route_id]
            
            #just go with first one (sometimes there are multiple)
            shape_id = shape_id.iloc[0,:]['shape_id']
            
            #filter shapes
            #TODO does not appear that shape id is lining up with shape id in shapes
            shape = shapes[shapes['shape_id']==shape_id].copy()
            
            #make the pt sequence the index
            shape.index = shape['shape_pt_sequence']
            
            #create geo column
            shape.loc[:,'geometry'] = gpd.points_from_xy(shape['shape_pt_lon'],shape['shape_pt_lat'])
            
            #turn to gdf
            shape = gpd.GeoDataFrame(shape,geometry='geometry',crs='epsg:4326')
            
            #project to same CRS as transit shapefile
            shape.to_crs(shape_map.crs,inplace=True)
            
            #get start pt sequence (snap start stop to geometry)
            start_point_snapped = shape.distance(start_point).idxmin()
            
            #get end pt sequence
            end_point_snapped = shape.distance(end_point).idxmin()
            
            #make sure they aren't the same points
            #only get shape points needed
            if start_point_snapped < end_point_snapped:
                line = shape.loc[start_point_snapped:end_point_snapped]
            else:
                line = shape.loc[end_point_snapped:start_point_snapped]
                
            #check length
            if len(line['geometry'].to_list()) > 1:
                #get linestring
                line = LineString(line['geometry'].to_list())
            else:
                line = None
            
        else:
            route_types.append('walking')
            route_ids.append('walking')
            names.append('walking')
            line = LineString([start_point,end_point])
            
        #replace the edge
        transit_edges.append(line)

    #contruct geodataframe
    gdf = gpd.GeoDataFrame(data={'start_stop':start_stop,'end_stop':end_stop,'mode':route_types,
                        'route_ids':route_ids,'name':names,'time':time_spent,'geometry':transit_edges}, geometry='geometry',crs=shape_map.crs)

    #add bike portions if available
    if row['bike_edge1'] is not np.nan:
        df = {
            'start_stop': src_taz,
            'end_stop': src_stop,
            'mode': 'bike',
            'route_ids': 'biking',
            'name': 'biking',
            'time': row['dist_first_leg'],
            'geometry': [row['bike_edge1']]
        }
        df = gpd.GeoDataFrame(df,geometry='geometry',crs=shape_map.crs)
        gdf = df.append(gdf,ignore_index=True)

    if row['bike_edge2'] is not np.nan:
        df = {
            'start_stop': dest_stop,
            'end_stop': dest_taz,
            'mode': 'bike',
            'route_ids': 'biking',
            'name': 'biking',
            'time': row['dist_last_leg'],
            'geometry': [row['bike_edge2']]
        }
        df = gpd.GeoDataFrame(df,geometry='geometry',crs=shape_map.crs)
        gdf = gdf.append(df,ignore_index=True)

    #create layer name
    trip_name = str(src_taz) + '_' + str(dest_taz)

    #create new folder
    if not (settings['output_fp'] / f'{mode}_{impedance}/mapped/{src_taz}').exists():
        (settings['output_fp'] / f'{mode}_{impedance}/mapped/{src_taz}').mkdir(parents=True) 

    #export
    gdf.to_file(settings['output_fp'] / f'{mode}_{impedance}/mapped/{src_taz}/{trip_name}.gpkg',layer=trip_time)