# -*- coding: utf-8 -*-
"""
Updated on 2/14/23

@author: tpassmore6
"""

# Import all necessary modules
from datetime import datetime, timedelta, date  # for handling dates and times
import geopandas as gpd  # for handling spatial data
import pandas as pd  # for handling data
import pickle  # for working with Python objects
from tqdm import tqdm  # for displaying progress bars
import time  # for measuring time
import os  # for interacting with the file system
import sys  # for interacting with the Python interpreter
from pathlib import Path  # for working with file paths

#suppress error messages
# import warnings
# from shapely.errors import ShapelyDeprecationWarning
# warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 
# warnings.filterwarnings("ignore", category=FutureWarning)

from RAPTOR.std_raptor import raptor
from miscellaneous_func import *

#used for creating departure times
def get_times(first_time,end_time,timestep):
    
    times = [first_time]
    
    reps = int(round((end_time - first_time) / timestep,0))
    
    for x in range(0,reps,1):
        times.append(times[-1] + timestep)
        
    return times

#makes sure thing is str (nessessary becuase taz ids will get converted to numeric)
def check_type(item):
    if type(item) == float:
        item = str(int(item))
    elif type(item) == int:
        item = str(item)      
    return item

#input format of parquet files
#'src_taz' = starting taz
#'dest_taz' = ending taz
#'src_stop' = potential starting transit stop
#'dest_stop' = potential ending transit stop
#f'{impedance}_to' = walking/biking distance to transit stop (used for determining arrival time)

def run_raptor(select_tazs:list,raptor_settings:dict,mode_specific:dict,ods=None):
    """
    run_raptor uses the one-to-one standard RAPTOR algorithm in the transit-routing Python Package to
    find the best transit routes between all pairs of potential starting transit stops and ending transit stops.

    Provide the run_raptor function a list of filepaths for parquet files, each containing a pandas dataframe with the following columns:
        'src_taz': Starting TAZ.
        'dest_taz': Ending TAZ.
        'src_stop': Potential starting transit stop.
        'dest_stop': Potential ending transit stop.
        '{}_first_leg': impedance to travel from src_taz to src_stop
        '{}_last_leg': impedance to travel form dest_stop to dest_taz
        f'{impedance}_to': Walking/biking distance to first transit stop (used for determining arrival time).
        f'{impedance}_from': Walking/biking distance from last transit stop

    Also provide run_raptor with dictionary raptor_settings with keys specifying the following:

        'NETWORK_NAME': The name of the network to use.
        'first_time': The first departure time for RAPTOR.
        'end_time': The last departure time for RAPTOR.
        'timestep': The time interval between departure times for RAPTOR.
        'MAX_TRANSFER': The maximum number of transfers allowed in a trip.
        'WALKING_FROM_SOURCE': whether you start at the given transit stop or allow walking to others (DISABLED)
        'CHANGE_TIME_SEC': The time it takes to change between two transit routes (in seconds).
        'PRINT_ITINERARY': Whether to print the itinerary of each trip or not.
        'thresh': The maximum biking distance (in feet) to transit stops.
        'spd': The biking speed (in miles per hour).
        'mode': The transportation mode ('transit' or 'bike').
        'overwrite_existing': Whether to overwrite the results if they already exist or not.
        'output_fp': The directory path for output files.

    Also provide a dict of mode specific settings:

        'spd'

    """
    
    print('Running raptor algorithm')

    #import routes for naming
    route = pd.read_csv(raptor_settings['gtfs_fp'] / 'route.txt')

    #retrieve info from dicts
    mode = mode_specific['mode']
    impedance = mode_specific['impedance']
    
    #deal with travel speeds if there are two
    if isinstance(mode_specific['spd'],tuple):
        spd1 = mode_specific['spd'][0]
        spd2 = mode_specific['spd'][1]
    else:
        spd1 = mode_specific['spd']
        spd2 = mode_specific['spd']

    #TODO change to make units chang
    print(f'First leg speed is {spd1} mph and last leg speed is {spd2} mph')
    
    #get filepaths for select tazs
    #set to where trips are stored
    trips_dir = raptor_settings['output_fp'] / f'{mode}_{impedance}/trips'
    all_trips = [trips_dir / f'{taz}.parquet' for taz in select_tazs]

    #import network files
    stops_file, trips_file, stop_times_file, transfers_file, stops_dict, stoptimes_dict, footpath_dict, routes_by_stop_dict, idx_by_route_stop_dict, routesindx_by_stop_dict = read_testcase(
        raptor_settings['NETWORK_NAME'])

    #print details
    print_network_details(transfers_file, trips_file, stops_file)
    
    #get list of times
    start_times = get_times(raptor_settings['first_time'],raptor_settings['end_time'],raptor_settings['timestep'])

    #do for each start time and start/end transit stop in the trip file
    for start_time in start_times:
        #print(f"Caclulating shortest transit route at time: {start_time}")
        for trips_fp in all_trips:
    
    #for x in tqdm([(start_time_lst, trip_lst) for start_time_lst in start_times for trip_lst in all_trips]):
            #pull values out of tuple
            #start_time = x[0]
            #trips_fp = x[1]
            
            #get start taz name
            taz_name = trips_fp.parts[-1].split('.')[0]

            #make file name
            time_name = f'{start_time.hour}_{start_time.minute}'

            #record starting time
            time_start = time.time()
            
            #check to see if trip file exists
            if not trips_fp.exists():
                print(f'No trip file exists for {trips_fp}')
                continue
            elif (raptor_settings['output_fp'] / f'{mode}_{impedance}/raptor_results/{taz_name}/{time_name}.pkl').exists() & (mode_specific['overwrite_existing'] == False):
                print(f"Already calculated {taz_name} at time {time_name}")
                continue

            #import trips file
            trips = pd.read_parquet(trips_fp)

            #check mode restrictions
            # if mode_specific['allow_bus_to_bus'] == False:
            #     route_and_stop = gpd.read_file(raptor_settings['output_fp']/'base_layers.gpkg',layer='route_and_stop')[['stop_id','route_type']]
            #     trips = pd.merge(trips,route_and_stop,left_on='src_stop',right_on='stop_id',how='left')
            #     trips = pd.merge(trips,route_and_stop,left_on='dest_stop',right_on='stop_id',how='left')
            #     check = (trips['route_type_y'] == 3) & (trips['route_type_x'] == 3)
            #     trips = trips[-check]
            #     print(f"{check.sum()} trips were bus to bus")
            #     trips.drop(columns=['stop_id_x','stop_id_y','route_type_y','route_type_x'],inplace=True)

            #remove trips if start_time isn't within 30 mins of the departure period in ABM (if depart period is 8:30am then don't run routing if current start_time isn't 8:30am or 8:45am)
            if ods is not None:
                subset = ods.copy()
                subset = subset[subset['origin']==taz_name]
                #time for trip in abm data must be at or past the start time
                equal = (start_time - subset['adjusted']) >= timedelta(0)
                #time for trip in abm data must be 15 minutes or less than the start time (8:15-8:00)
                less_than_15 = (start_time - subset['adjusted'] ) <= timedelta(minutes=15)
                subset = subset[equal & less_than_15]
                trips = trips[trips['dest_taz'].isin(set(subset['destination']))]

            #check to see if there are any trips in the parquet file
            if trips.shape[0] == 0:
                #print(f'No trips possible from {taz_name}')
                continue

            #get actual first leg time (using feet and miles per hour)
            trips[f'{impedance}_to_time'] = pd.to_timedelta(trips[f'{impedance}_first_leg'] / 5280 / spd1 * 60 * 60, unit='s')#.dt.round(datetime.timedelta(minutes=1))
        
            #find actual final leg time
            trips[f'{impedance}_from_time'] = pd.to_timedelta(trips[f'{impedance}_last_leg'] / 5280 / spd2 * 60 * 60, unit='s')#.dt.round(datetime.timedelta(minutes=1))
        
            #get arrival time (to nearest minute) at first transit stop (needs to be datetime)
            trips['arrival_time'] = (start_time + trips[f'{impedance}_to_time']).dt.round(timedelta(minutes=1))
            
            #make empty cols to store raptor outputs
            trips['status'] = pd.Series(dtype=str)
            trips['transit_time'] = pd.Series(dtype='timedelta64[ns]')
            trips['travel_time'] = pd.Series(dtype='timedelta64[ns]')
            trips['num_transfers'] = pd.Series(dtype=int)
            trips['edge_list'] = pd.Series(dtype=object)
            
            #create a new raptor df for calucating the all unique src/dest transit stops with same arrival time at first stop
            raptor_df = trips[['src_stop','dest_stop','arrival_time']].drop_duplicates().reset_index(drop=True)

            #import solved raptor trips to see if it's already been calculated 
            if ((raptor_settings['output_fp'] / f"raptor_dict_{raptor_settings['MAX_TRANSFER']}_transfers.pkl").exists()):
                with (raptor_settings['output_fp'] / f"raptor_dict_{raptor_settings['MAX_TRANSFER']}_transfers.pkl").open(mode='rb') as fh:
                    pareto_dict = pickle.load(fh)
                #remove rows from raptor_df if already solved
                raptor_df['tup'] = list(zip(raptor_df['src_stop'].astype(int),raptor_df['dest_stop'].astype(int),raptor_df['arrival_time']))
                previously_calculated = raptor_df['tup'].isin(list(pareto_dict.keys()))
                raptor_df = raptor_df[-previously_calculated]
                print(f"RAPTOR routing for {previously_calculated.sum()} trips have already been solved")
            else:
                #initialize an empty dict to store pareto results
                pareto_dict = {}

            if raptor_df.shape[0] != 0:
                print(f'Performing transit routing from {taz_name} at {start_time}:')
                #should loop through all unique times instead of every row (round arrival time to nearest minute)
                for row in tqdm(raptor_df.itertuples(),total=raptor_df.shape[0]):
                #for row in raptor_df.itertuples():
                    
                    #pull out the three inputs needed for raptor
                    SOURCE = int(row[1])
                    DESTINATION = int(row[2])
                    D_TIME = row[3]

                    #run standard raptor algorithm
                    output, pareto_set = raptor(SOURCE, DESTINATION, D_TIME, raptor_settings['MAX_TRANSFER'], 
                                    raptor_settings['WALKING_FROM_SOURCE'], raptor_settings['CHANGE_TIME_SEC'], raptor_settings['PRINT_ITINERARY'],
                                    routes_by_stop_dict, stops_dict, stoptimes_dict, footpath_dict, idx_by_route_stop_dict)
                    
                    #store pareto set in dict for next step
                    pareto_dict[(SOURCE,DESTINATION,D_TIME)] = pareto_set
            else:
                print(f"RAPTOR solution already found from {taz_name} at {start_time}")

            #export the pareto dict (can import later to speed up processing and go across modes)
            with (raptor_settings['output_fp'] / f"raptor_dict_{raptor_settings['MAX_TRANSFER']}_transfers.pkl").open(mode='wb') as fh:
                pickle.dump(pareto_dict,fh)

            #need to change this from iterrows to concat instead
            print(f'Solving shortest {mode} + transit routing from {taz_name} at {start_time}:')
            for row in tqdm(trips.itertuples(),total=trips.shape[0]):
            #for row in trips.itertuples():
                
                #pull out the three inputs needed for raptor
                SOURCE = int(row[3])
                DESTINATION = int(row[4])
                D_TIME = row[11]
                
                #get pareto set from the pareto dict
                pareto_set = pareto_dict.get((SOURCE,DESTINATION,D_TIME),'Error')

                #skip to next if no solutions
                #come back to this
                if pareto_set == None:
                    trips.at[row[0],'status'] = 'not possible'
                    continue
                elif pareto_set == 'Error':
                    print('Error')
                    break

                #create empty list for storing pareto results
                shortest_time = []
                
                #go through each pareto optimal for the given number of transfers
                for pareto_optimal in pareto_set:
                
                    #pull out the number of transfers and the edge list
                    num_transfers = pareto_optimal[0]
                    edges = pareto_optimal[1]
                    
                    #get rid of final walking leg (will be replaced with network results)
                    while edges[-1][0] == 'walking':
                        edges = edges[:-1]

                    #get total travel time from start (time at final egress + last bike/walk leg - departure time)
                    travel_time = edges[-1][3] + row[10] - start_time

                    #get total transit travel time (final egress - departure time from first transit stop) 
                    transit_time = edges[-1][3] - D_TIME

                    #store as list
                    candidate = [num_transfers,travel_time,transit_time,edges]

                    #only retain shortest travel time
                    if len(shortest_time) == 0:
                        shortest_time = candidate
                    elif candidate[1] < shortest_time[1]:
                        shortest_time = candidate
                
                #update trips dataframe with results from raptor 
                trips.at[row[0],'num_transfers'] = shortest_time[0]
                trips.at[row[0],'travel_time'] = shortest_time[1]
                trips.at[row[0],'transit_time'] = shortest_time[2]
                
                #initialize edge list for storing legs of transit trip
                edge_list = []
                
                #track the wait time
                wait_time = timedelta(minutes=0)

                #store the first arrival time
                arrive = D_TIME

                #track transfer time
                transfer_time = timedelta(minutes=0)

                #edge list structure
                #leg[0] : ""walking" (str) or the boarding time (datetime) if transit
                #leg[1] : starting transit stop
                #leg[2] : ending transit stop
                #leg[3] : if walking = travel time and if transit = time at egress
                #leg[4] : only for transit = route and trip number

                #transit tup to output for edge list
                #(start stop, end stop, egress time - boarding time (travel time), route/trip, transit mode)
            
                #go through each leg in edge list and format
                for leg in shortest_time[3]:
                    
                    #format transit legs
                    if leg[0] != 'walking':
                        
                        #calculate wait time for segment (time at boarding - time arrived)
                        segment_wait_time = leg[0] - arrive

                        #add to total weight time
                        wait_time += segment_wait_time
                        
                        #replace with the next arrival time at the next stop
                        arrive = leg[3]

                        #extract route_id
                        route_id = str.split(leg[4],'_')[0]
                        
                        #get transit mode from the transit routes csv
                        route_type = route[route['new_route_id'].astype(str)==route_id]['route_type'].item()

                        #get transit line name
                        name = route[route['new_route_id'].astype(str)==route_id]['route_long_name'].item()      
                        
                        #check if bus trip is at least 5 mins (FUTURE)

                        #format transit tuple to add to the edge list
                        #TODO work with prateek on ensuring that the newly labeled routes and stops are labeled in all GTFS files
                        if (route_type == 1) | (route_type == '1'):
                            tup = (leg[1],leg[2],leg[3]-leg[0],leg[4],'rail',name)
                        elif (route_type == 3) | (route_type == '3'):
                            tup = (leg[1],leg[2],leg[3]-leg[0],leg[4],'bus',name) 

                    #format walk legs
                    else:
                        #format the walking tuple and add to edge list
                        tup = (leg[1],leg[2],leg[3],'walking')
                        
                        #track transfer time (so it can be subtracted from wait time)
                        transfer_time += leg[3]

                    #add tuple to edge list
                    edge_list.append(tup)
                
                #get total wait time
                trips.at[row[0],'wait_time'] = wait_time - transfer_time
                
                #get total transfer time
                trips.at[row[0],'transfer_time'] = transfer_time

                #add edge list to trips
                trips.at[row[0],'edge_list'] = edge_list
                
                #set initial success message
                trips.at[row[0],'status'] = 'success'
                
                # removed these two for now
                # #check travel time
                # if travel_time >= raptor_settings['timelimit']:
                #     trips.at[row[0],'status'] = 'time limit exceeded'
                
                #check mode restrictions
                if not mode_specific['allow_bus_to_bus']:
                    if len([x for x in edge_list if x[-2] == 'bus']) > 1:
                        trips.at[row[0],'status'] = 'two buses'

            #embed start time
            trips['start_time'] = start_time

            #create filepath
            if not (raptor_settings['output_fp'] / f'{mode}_{impedance}/raptor_results/{taz_name}').exists():
                (raptor_settings['output_fp'] / f'{mode}_{impedance}/raptor_results/{taz_name}').mkdir(parents=True)
                
            #export dataframe
            with (raptor_settings['output_fp'] / f'{mode}_{impedance}/raptor_results/{taz_name}/{time_name}.pkl').open(mode='wb') as fh:
                pickle.dump(trips,fh)
            
            #record total run time
            run_time_transit = round(((time.time() - time_start)/60), 2)  
            
            if not (raptor_settings['output_fp'] / f'{mode}_{impedance}/metadata.pkl').exists():
                metadata = {}
            else:
                with (raptor_settings['output_fp'] / f'{mode}_{impedance}/metadata.pkl').open(mode='rb') as fh:
                    metadata = pickle.load(fh)
            
            #add runtime info
            metadata[taz_name+' '+time_name] = run_time_transit
            
            #export
            with (raptor_settings['output_fp'] / f'{mode}_{impedance}/metadata.pkl').open(mode='wb') as fh:
                pickle.dump(metadata,fh)
            

