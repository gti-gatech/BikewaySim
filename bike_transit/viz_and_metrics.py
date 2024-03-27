# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 17:37:50 2022

@author: tpassmore6
"""
#imports
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import fiona
import numpy as np
from datetime import timedelta

#import custom functions
from helper_functions import process_results, load_files

'''
Notes:

take more of a sql approach update/merge?

have taz centroid/polygon layer instead seperating into many layers

when merging, it's duplicating the dest_taz column

'''

def viz_and_metrics(settings:dict,impedance:str,mode:str,list_tazs:list,time_limit:timedelta):

    for taz in list_tazs:
        print(f"Generating geopackages for {taz} taz.")        
        #import taz polygons
        tazs = gpd.read_file(settings['output_fp'] / 'base_layers.gpkg',layer='tazs')[[settings['keyid'],'geometry']]

        #import taz centroids
        centroids = gpd.read_file(settings['output_fp'] / 'base_layers.gpkg',layer='centroids')[[settings['keyid'],'geometry']] 

        #import walkable/bikable tazs
        #bike_tazs = gpd.read_file(settings['output_fp'] / f'{mode}_{impedance}/{taz}.gpkg',layer=f'{mode}able_tazs_polygons',ignore_geometry=True)

        #get filepaths for all departure times
        dep_times = settings['output_fp'].glob(f'{mode}_{impedance}/raptor_results/{taz}/*.pkl')

        #initialize empty dataframe for storing all the trip data per departure time
        all_trips = pd.DataFrame()

        #loop through each departure time
        for dep_time in dep_times:

            #get departure time text
            #dep = dep_time.parts[-1].split('.pkl')[0]

            #read in results file and find the least time route per departure time
            trip_df = process_results(dep_time)

            #convert time cols from datetime to total minutes (round to 1 decimal)
            time_cols = trip_df.columns[trip_df.dtypes == 'timedelta64[ns]']
            for time_col in time_cols:
                trip_df[time_col] = trip_df[time_col].apply(lambda x: round(x.total_seconds() / 60,1))

            #append to all_trips dataframe
            all_trips = pd.concat([all_trips,trip_df],ignore_index=True)

        #get average transit time, travel time, and wait time
        avg_transit_time = all_trips.groupby('dest_taz')['transit_time'].mean()
        avg_travel_time = all_trips.groupby('dest_taz')['travel_time'].mean()
        avg_wait_time = all_trips.groupby('dest_taz')['wait_time'].mean()

        #get minimum number of transfers
        #min_transfers = all_trips.groupby('dest_taz')['num_transfers'].min()
        #find tazs with the minimum number of tranfers that are within 2 mins of shortest time
        within_five = all_trips[(all_trips['travel_time'] - all_trips.groupby('dest_taz')['travel_time'].transform(min)) <= 2]
        min_transfers = within_five.groupby('dest_taz')['num_transfers'].min()

        # go through the edge list and concatanate all the transit modes together  
        for idx, row in all_trips.iterrows():
            
            #list comp to get list of modes for each trip
            modes = [edge[-2] for edge in row['edge_list'] if (edge[-2] == 'bus') or (edge[-2] == 'rail')]
            
            #turn to series
            modes = pd.Series(modes)

            #types
            if modes.nunique() > 1:
                types = 'Two Modes'
            elif modes[0] == 'rail':
                types = 'Rail'
            elif modes[0] == 'bus':
                types = 'Bus'
     
            
            #add to trip_df
            all_trips.at[idx,'types'] = types

        #get mode of transit type (bus, rail, mixed)
        mode_type = all_trips.groupby('dest_taz')['types'].agg(pd.Series.mode)

        #if more than one mode, say mixed
        mode_type = mode_type.apply(lambda x: replace_list_with_string(x))

        #join data to tazs
        tazs['avg_transit_time'] = tazs[settings['keyid']].map(avg_transit_time)
        tazs['avg_travel_time'] = tazs[settings['keyid']].map(avg_travel_time)
        tazs['avg_wait_time'] = tazs[settings['keyid']].map(avg_wait_time)
        tazs['min_transfers'] = tazs[settings['keyid']].map(min_transfers)
        tazs['mode_type'] = tazs[settings['keyid']].map(mode_type)
        #prolly add min/max for the times and count how many time periods taz was inaccessible
        
        #drop null rows (see what happens when i don't)
        #tazs = tazs[-tazs.isna().any(axis=1)]

        #drop if above 60 minutes
        tazs = tazs[tazs['avg_travel_time'] <= (time_limit.total_seconds() / 60)]

        #map all transit routes used
        under60 = all_trips[all_trips['dest_taz'].isin(tazs[settings['keyid']])]
        under60 = (under60['src_taz'] + '_' + under60['dest_taz']).tolist()
        transit_shed(settings,impedance,mode,taz,under60)

        #bin for easier symbology
        bins = list(range(0,61,15))
        labels = []
        while len(bins) > 1:
            label = f'{bins[0]+1}-{bins[1]}'
            labels.append(label)
            bins.pop(0)

        #bin data
        tazs['time_bin'] = pd.cut(x=tazs['avg_travel_time'],bins=range(0,61,15),labels=labels).astype(str)
        #https://colorbrewer2.org/?type=sequential&scheme=OrRd&n=4
        #color dict
        colors = {labels[0]:'#fef0d9',labels[1]:'#fdcc8a',labels[2]:'#fc8d59',labels[3]:'#d7301f'}
        tazs['time_color'] = tazs['time_bin'].map(colors)

        #bin wait time too
        labels = ['0-5','5-10','10-15','15-20+']
        tazs['wait_bin'] = pd.cut(x=tazs['avg_wait_time'],bins=[0,5,10,15,20],labels=labels,right=False).astype(str)
        colors = {labels[0]:'#fef0d9',labels[1]:'#fdcc8a',labels[2]:'#fc8d59',labels[3]:'#d7301f'}
        tazs['wait_color'] = tazs['wait_bin'].map(colors)

        #export
        tazs.to_file(settings['output_fp'] / f'{mode}_{impedance}/visuals/{taz}.gpkg',layer='tazs_viz')
        
        #dissolve modes
        tazs.dissolve('mode_type').to_file(settings['output_fp'] / f'{mode}_{impedance}/visuals/{taz}.gpkg',layer='mode_type')
        tazs.dissolve('min_transfers').to_file(settings['output_fp'] / f'{mode}_{impedance}/visuals/{taz}.gpkg',layer='min_transfers')

        #drop the geometry column
        tazs.drop(columns=['geometry'],inplace=True)

        #join data to centroids by merging with tazs
        centroids = pd.merge(centroids,tazs,on=settings['keyid'])
        
        #drop null rows (shouldn't be any)
        #centroids = centroids[-centroids.isna().any(axis=1)]

        #export
        centroids.to_file(settings['output_fp'] / f'{mode}_{impedance}/visuals/{taz}.gpkg',layer='centroids_viz')


def replace_list_with_string(lst):
    if isinstance(lst, np.ndarray):
        return 'Mixed'
    elif isinstance(lst,list):
        return 'Mixed'
    else:
        return lst

def transit_shed(settings:dict,impedance:str,mode:str,list_taz:str,shortest_trips:list):
    '''
    Creates polygon showing all the transit lines utilized using the RAPTOR outputs

    would like to also figure out a betweenness centrality like metric but that will wait

    Edited to only inlude trips that are the shortest
    '''

    #get filepaths for trips
    fps = settings['output_fp'].glob(f'{mode}_{impedance}/mapped/{list_taz}/*.gpkg')

    #only keep ones that are the shortest
    fps = [fp for fp in fps if fp.parts[-1].split('.gpkg')[0] in set(shortest_trips)]

    big_df = gpd.GeoDataFrame()
    
    for fp in fps:
        #load each time
        for start_time in fiona.listlayers(fp):
        
            trip = gpd.read_file(fp,layer=start_time)
                
            #only keep transit
            trip = trip.loc[trip['mode'].isin(['rail','bus']),['mode','start_stop','end_stop','geometry']]
            
            big_df = pd.concat([big_df,trip],ignore_index=True)
        
        #drop duplicates
        big_df.drop_duplicates(['mode','start_stop','end_stop'],inplace=True)
    
    #set activee geo column
    big_df.set_geometry('geometry',inplace=True)

    #buffer because polygons faster to dissolve than linestrings
    big_df.geometry = big_df.buffer(400)
    
    #dissolve
    big_df = big_df.dissolve('mode')
    
    #export
    big_df.to_file(settings['output_fp'] / f'{mode}_{impedance}/visuals/{list_taz}.gpkg',layer='transitshed')




# def access_map(settings:dict,impedance:str,mode:str,list_taz:str):
        
#     '''
#     import routing results for each taz and mark all tazs that can be reached   
#     '''

#     #read tazs
#     tazs = gpd.read_file(settings['output_fp'] / 'base_layers.gpkg',layer='tazs')
#     #tazs.loc[:,settings['keyid']] = tazs[settings['keyid']].astype(str)

#     #get the walk/bikeable tazs too
#     bike_tazs = gpd.read_file(settings['output_fp'] / f'{mode}_{impedance}/{list_taz}.gpkg',layer=f'{mode}able_tazs_polygons',ignore_geometry=True)
    
#     #get departure times filepaths
#     dep_times = settings['output_fp'].glob(f'{mode}_{impedance}/raptor_results/{list_taz}/*.pkl')
    
#     #initialize empty set that will be filled with accessible tazs
#     dest_tazs = set()

#     for dep_time in dep_times:
#         #read in trip_df
#         trip_df = process_results(dep_time)
#         #add accessible tazs to set
#         dest_tazs.update(trip_df['dest_taz'])
        
#     #add bikeable/walkable tazs
#     dest_tazs.update(bike_tazs[settings['keyid']])   
      
#     #filter to reachable tazs
#     reachable_tazs = tazs[tazs[settings['keyid']].isin(dest_tazs)]
    
#     #export    
#     reachable_tazs.to_file(settings['output_fp'] / f'{mode}_{impedance}/visuals/{list_taz}.gpkg',layer='access_map')


# def dot_map(settings:dict,impedance:str,mode:str,list_taz:str):
    
#     #import tazs
#     #tazs = gpd.read_file(settings['output_fp'] / 'base_layers.gpkg',layer='tazs')
#     #tazs.loc[:,settings['keyid']] = tazs[settings['keyid']].astype(str)  

#     #read taz centroids
#     tazs_centroid = gpd.read_file(settings['output_fp'] / 'base_layers.gpkg',layer='centroids')
#     #tazs_centroid[settings['keyid']] = tazs_centroid[settings['keyid']].astype(str)
    
#     #get filepaths
#     dep_times = settings['output_fp'].glob(f'{mode}_{impedance}/raptor_results/{list_taz}/*.pkl')
    
#     for dep_time in dep_times:
#         #read in trip df
#         trip_df = process_results(dep_time)
        
#         #get dep time
#         dep = dep_time.parts[-1].split('.pkl')[0]
        
#         #match to taz centroids and convert form datetime to total minutes
#         tazs_centroid[dep] = pd.merge(tazs_centroid,trip_df[['dest_taz','travel_time']],left_on=settings['keyid'],right_on=['dest_taz'],how='left')['travel_time'].apply(lambda x: x.total_seconds() / 60)

#     #get the average times and clean up
#     tazs_centroid['mean_travel_time'] = tazs_centroid.iloc[:,2:].mean(axis=1)

#     #filter columns
#     tazs_centroid = tazs_centroid[[settings['keyid'],'mean_travel_time','geometry']]
    
#     #drop na
#     tazs_centroid.dropna(inplace=True)
    
#     #bin the data for easy symbology
#     bins = list(range(0,61,15))
#     labels = []
#     while len(bins) > 1:
#         label = f'{bins[0]+1}-{bins[1]}'
#         labels.append(label)
#         bins.pop(0)

#     #bin data
#     #TODO: consider not having a bin for the less than 25 mins if there are none
#     tazs_centroid['time_bin'] = pd.cut(x=tazs_centroid['mean_travel_time'],bins=range(0,61,15),labels=labels).astype(str)

#     #https://colorbrewer2.org/?type=sequential&scheme=OrRd&n=4
#     #color dict
#     colors = {labels[0]:'#fef0d9',labels[1]:'#fdcc8a',labels[2]:'#fc8d59',labels[3]:'#d7301f'}

#     #map
#     tazs_centroid['color'] = tazs_centroid['time_bin'].map(colors)

#     #merge data to centroid one
#     tazs_centroid.to_file(settings['output_fp'] / f'{mode}_{impedance}/visuals/{list_taz}.gpkg',layer='dot_map')
              


# depricated, run time shown in notebook
# def get_runtime(mode_imp):
#     '''
#     Gets the runtime statistics using the metadata outputted from RAPTOR
#     '''
    
#     arr = []
#     for x in Path('.').glob(f'Outputs/{mode_imp}/metadata/*/*.pkl'):
#         with open(x,'rb') as fh:
#             arr.append(pickle.load(fh)['run_time'])
    
#     #get runtime stats
#     #round these
#     print(f'Mean: {np.mean(arr)} mins.')
#     print(f'Median: {np.median(arr)} mins.')
#     print(f'Max: {np.max(arr)} mins.')
#     print(f'Min: {np.min(arr)} mins.')
