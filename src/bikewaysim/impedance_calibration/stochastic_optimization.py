#https://keurfonluu.github.io/stochopy/api/optimize.html

from pathlib import Path
from stochopy.optimize import minimize
from shapely.ops import nearest_points, Point, MultiLineString, LineString
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from tqdm import tqdm
import time
from datetime import timedelta
import ast
import shapely
import sys
import inspect
import random
import similaritymeasures

from bikewaysim.paths import config
from bikewaysim.network import modeling_turns
from bikewaysim.routing import rustworkx_routing_funcs, route_utils
from bikewaysim.impedance_calibration import export_utils, loss_functions, post_calibration

'''
This module is for deriving link costs from matched GPS traces
through the stochastic optimization method used in Schweizer et al. 2020

Required Files:
Network links for shortest path routing
Matched traces

Pre-Algo:
- Create network graph with lengths as original weights
- Create dict with link lengths
- set parameter search range

Algo Steps:


'''

########################################################################################

# Pre-Calibration Functions

########################################################################################

# rematch the data in case certain links were removed

def extract_bounds(betas):
    '''
    For data wrangling
    '''
    return [x.get('range') for x in betas]

#NOTE still not sure if this would be better to with the coordinates instead incase something gets removed
def match_results_to_ods_w_year(match_results):
    #for the shortest path routing ste, except this returns the year
    ods = [(item['origin_node'],item['destination_node'],item['trip_start_time']) for key, item in match_results.items()]
    #ensure that we don't repeat duplicate shortest path searches
    ods = list(set(ods))
    #sort it by year
    ods = sorted(ods,key=lambda x:x[-1])[::-1]
    return ods

########################################################################################

# Impedance Calibration

########################################################################################

import datetime
#TODO rename variables like full_set to match_results or matched_traces, just develop some type of convention

def impedance_calibration(betas:np.array,
                          past_vals:list,
                          betas_tup:tuple,
                          set_to_zero:list,
                          set_to_inf:list,
                          ods:list,
                          match_results:dict,
                          link_impedance_function,
                          base_impedance_col:str,
                          turn_impedance_function,
                          links:pd.DataFrame,turns:pd.DataFrame,turn_G,
                          loss_function,
                          loss_function_kwargs,
                          print_results=True,
                          track_calibration_results=True,
                          ):

    
    # start_time = time.time()

    # runs the shortest path routing part
    results_dict = impedance_routing(
        betas,
        betas_tup,
        link_impedance_function,
        base_impedance_col,
        turn_impedance_function,
        links,
        turns,
        turn_G,
        ods,
        set_to_zero,
        set_to_inf
        )

    if results_dict is None:
        if print_results:
            print('negative edge weights, skipping guess')
        if track_calibration_results == True:
            past_vals.append(list(betas)+[np.nan])
    
    #calculate the objective function
    #trim kwargs to only what's needed for the funtion 
    signature = inspect.signature(loss_function)
    arg_names = [param.name for param in signature.parameters.values()]
    loss_function_kwargs = {key:item for key,item in loss_function_kwargs.items() if key in arg_names}
    val_to_minimize = loss_function(results_dict,match_results,**loss_function_kwargs)
    
    if print_results:
        print(list(betas.round(2))+[np.round(val_to_minimize,2)])

    #keep track of all the past betas used for visualization purposes
    if track_calibration_results == True:
        past_vals.append(list(betas)+[val_to_minimize])
    
    # print('took',(time.time()-start_time),'s')
    return val_to_minimize

def impedance_routing(
        betas,
        betas_tup,
        link_impedance_function,
        base_impedance_col,
        turn_impedance_function,
        links,
        turns,
        turn_G,
        ods,
        set_to_zero,
        set_to_inf
    ):
    '''
    Function that performs the shortest path routing aspect of the impedance calibration
    '''

    # create a copy of links so the original isn't modified when the next 
    # round of shortest paths is calculated
    links = links.copy()
    
    # if infra is off street (i.e., the link should no longer be traversable)
    links['link_cost_override'] = False

    #set link costs accordingly
    #TODO think about how to make this work with the trip dates, thinking that it would loop through and update for each year
    # takes about 2 seconds to update the impedance factors
    updated_edge_costs = rustworkx_routing_funcs.impedance_update(betas,betas_tup,
                          link_impedance_function,
                          base_impedance_col,
                          turn_impedance_function,
                          links,turns,turn_G)

    # make sure there are no negative edge weights
    if isinstance(updated_edge_costs,bool):
        return None

    #TODO clean this up while calibrations are running in the background
    starts = [x[0] for x in ods]
    ends = [x[1] for x in ods]
    years = sorted(list(set([x[2] for x in ods])))[::-1]

    year_networks = rustworkx_routing_funcs.create_year_networks(
        betas,betas_tup,starts,ends,turn_G,links,turns,set_to_zero,set_to_inf,years,link_impedance_function,turn_impedance_function,base_impedance_col
    )
    
    # second negative impedance check
    if year_networks is None:
        return None

    # add virtual edges to the main graph
    added_nodes = rustworkx_routing_funcs.add_virtual_edges(starts,ends,links,turns,turn_G)
    
    # calculate the shortest paths
    path_lengths, shortest_paths = rustworkx_routing_funcs.rx_shortest_paths_year(ods,turn_G,year_networks)
    
    # format the output
    results_dict = {(od[0],od[1]):{'length':path_length,'edge_list':shortest_path} for path_length, shortest_path, od in zip(path_lengths,shortest_paths,ods)}
    
    # remove the virtual links for the next run
    rustworkx_routing_funcs.remove_virtual_links(added_nodes,turn_G)

    return results_dict

########################################################################################

# Impedance Functions

########################################################################################

'''
Currently works with binary and numeric variables. Categorical data will have to be
cast into a different format for now.

Link impedance is weighted by the length of the link, turns are just the impedance associated
'''

def link_impedance_function(betas:np.array,betas_tup:tuple,links:pd.DataFrame,base_impedance_col:str,trip_specific=None):
    '''
    Default link impedance function. Assumes that link impedance factors are additive
    and increase/decrease link impedance proportional to the link's distance/travel time.

    Rounded to eight decimals because expected units are in travel time

    Modifies the links dataframe inplace, so this function doesn't return anything.

    #TODO change travel time to seconds instead?
    '''
    
    # set up a multiplier of zeros that gets addded/subtracted to for visualization purposes
    multiplier = np.zeros(links.shape[0])

    # retrieve col names of the links and the positions in the betas array
    betas_links = [(idx,x['col']) for idx, x in enumerate(betas_tup) if x['type']=='link']

    # check to see if there are any link impedances at all
    if len(betas_links) > 0:
        #assumes that these effects are additive
        for idx, col in betas_links:
            multiplier = multiplier + (betas[idx] * links[col].values)
        #scale the multiplier
        if trip_specific is not None:
            multiplier = multiplier * trip_specific
        #stores the multiplier
        links['multiplier'] = multiplier
        links['link_cost'] = links[base_impedance_col] * (1 + multiplier)
    else:
        print('No link impedance factors assigned')
        links['link_cost'] = links[base_impedance_col]

    # round link cost
    links['link_cost'] = links['link_cost'].round(8)

def turn_impedance_function(betas:np.array,betas_tup:tuple,turns:pd.DataFrame):
    '''
    Default turn impedance function. Assumes that turns have zero initial cost.
    Event impedance so it's in the same units as the impedance (time or distance)
    '''
    #initialize a zero turn cost column
    turns['turn_cost'] = 0

    # retrieve col names of the links and the positions in the betas array
    betas_turns = [(idx,x['col']) for idx, x in enumerate(betas_tup) if x['type']=='turn']

    if len(betas_turns) > 0:
        #instance impedance
        for idx, col in betas_turns:
            turns['turn_cost'] = turns['turn_cost'] + (betas[idx] * turns[col])
    # TODO also have the print thing appear here but use the print results setting

    # round turn cost
    turns['turn_cost'] = turns['turn_cost'].round(8)

########################################################################################

# Scripting Calibration Runs

########################################################################################

def full_impedance_calibration(
        calibration_name:str,
        betas_tup:tuple,
        objective_function=loss_functions.jaccard_buffer_mean,
        set_to_zero=[],
        set_to_inf=[],
        stochastic_optimization_settings={'method':'pso','options':{'maxiter':100,'popsize':25}},
        print_results = False, # default is false
        base_impedance_col='travel_time_min',
        subset = None, # tuple with (userid/rider_type/etc,list_of_trips) to subset data (OPTIONAL)
        ):
    
    '''
    Use this to run the impedance calibration. Parameters are subject to change but it's meant to represent the
    parameters you actually want to change and not ones that are going to be repeated across calibration runs such
    as the network. This is subject to change if it seems some new variables should be added.

    All results save as pickle files into one of two sets of directories depending on if using user by user calibration
    or all trips calibration.

    '''
        
    # import network
    links, turns, length_dict, geo_dict, turn_G = rustworkx_routing_funcs.import_calibration_network(config)

    # import matched traces (all of them at first)
    with (config['calibration_fp']/'ready_for_calibration.pkl').open('rb') as fh:
        full_set = pickle.load(fh)

    # subset to a user's trips if user argument is provided
    if subset is not None:
        full_set = {tripid:item for tripid, item in full_set.items() if tripid in subset[1]}

    # format arguments for impedance calibration
    args = (
        [], # empty list for storing past calibration results
        betas_tup, # tuple containing the impedance spec
        set_to_zero, # if the trip year exceeds the link year set these attributes to zero
        set_to_inf, # if the trip year exceeds the link year set the cost to 9e9
        match_results_to_ods_w_year(full_set), # list of OD network node pairs needed for shortest path routing 
        full_set, # dict containing the origin/dest node and map matched edges
        link_impedance_function, # link impedance function to use
        base_impedance_col, # column with the base the base impedance in travel time or distance
        turn_impedance_function, # turn impedance function to use
        links,turns,turn_G, # network parts
        objective_function, # loss function to use
        {'length_dict':length_dict,'geo_dict':geo_dict}, # keyword arguments for loss function
        print_results, #whether to print the results of each iteration (useful when testing the calibration on its own)
        True, #whether to store calibration results
    )

    # run the calibration
    start = time.time()
    if print_results:
        print([x['col'] for x in betas_tup]+['objective_function'])
    x = minimize(impedance_calibration,
                extract_bounds(betas_tup),
                args=args,
                **stochastic_optimization_settings)
    end = time.time()
    
    print('Duration:',str(pd.Timedelta(seconds=end-start).round('s')))
    print(f"{objective_function.__name__}: {x.fun}")
    print(x)
    
    # assemble dictionary of the calibration results
    calibration_result = {
        'betas_tup': tuple({**item,'beta':x.x[idx].round(4)} for idx,item in enumerate(betas_tup)), # contains the betas
        'set_to_zero': set_to_zero,
        'set_to_inf': set_to_inf,
        # TODO add base impedance column, name of the turn and link impedance functions here later on
        'settings': stochastic_optimization_settings, # contains the optimization settings
        'objective_function': objective_function.__name__, # objective function used
        'results': x, # stochastic optimization outputs
        'trips_calibrated': set(full_set.keys()), # saves which trips were calibrated
        'past_vals': args[0], # all of the past values/guesses
        'runtime': datetime.timedelta(seconds= end - start),
        'time': datetime.datetime.now()
    }

    #NOTE consider keeping these in the same directory
    result_fp, routing_fp, loss_fp = export_utils.handle_directories(subset,calibration_name)

    # export the calibration results
    with export_utils.uniquify(result_fp).open('wb') as fh:
        pickle.dump(calibration_result,fh)

# Helper function to unpack the dictionary and call example_function
def run_calibration(task):
    task_dict, run_num, NUM_RUNS = task
    task_name = f"{task_dict['calibration_name']} ({run_num+1}/{NUM_RUNS})"
    print('Starting:',task_name)
    success = full_impedance_calibration(**task_dict)
    return task_name, success
