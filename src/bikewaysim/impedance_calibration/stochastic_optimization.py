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
import branca
import random

import similaritymeasures
from bikewaysim.network import modeling_turns

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

Todo:
- see if args can be a named tuple so it's easier to make changes to it

'''

########################################################################################

# Import Functions

########################################################################################

def import_calibration_network(config):
    '''
    Backend function for loading calibration network files
    '''
    # import the calibration network
    with (config['calibration_fp']/"calibration_network.pkl").open('rb') as fh:
        links,turns = pickle.load(fh)
    # make the length and geo dict
    length_dict = dict(zip(links['linkid'],links.length))
    geo_dict = dict(zip(links['linkid'],links.geometry))
    # form turn graph
    turn_G = modeling_turns.make_turn_graph(turns)
    return links, turns, length_dict, geo_dict, turn_G

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
def match_results_to_ods(match_results):
    #for the shortest path routing step, takes match_results and gets all unique od pairs
    ods = [(item['origin_node'],item['destination_node']) for key, item in match_results.items()]
    ods = np.unique(np.array(ods),axis=0)
    ods = [tuple(row) for row in ods]
    return ods

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

def full_impedance_calibration(betas_tup,args,stochastic_optimization_settings,full_set,calibration_name):
    '''
    Use this to run the impedance calibration with less code
    '''
    
    start = time.time()
    if args[-3]:
        print([x['col'] for x in betas_tup]+['objective_function'])
    
    x = minimize(impedance_calibration,
                extract_bounds(betas_tup),
                args=args,
                **stochastic_optimization_settings)
    end = time.time()
    print(calibration_name,'complete, took',str(pd.Timedelta(seconds=end-start).round('s')),'hours')
 
    if args[-2]:
        print(args[12])
        print(f"{args[12].__name__}: {x.fun}")
        print(x)

    calibration_result = {
        'betas_tup': tuple({**item,'beta':x.x[idx].round(4)} for idx,item in enumerate(betas_tup)), # contains the betas
        'settings': stochastic_optimization_settings, # contains the optimization settings
        'objective_function': args[12].__name__, # objective function used
        'results': x, # stochastic optimization outputs
        'trips_calibrated': set(full_set.keys()), # saves which trips were calibrated
        'past_vals': args[0], # all of the past values/guesses
        'runtime': pd.Timedelta(end-start),
        'time': datetime.datetime.now()
    }
    return calibration_result

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
                          links:pd.DataFrame,turns:pd.DataFrame,turn_G:nx.digraph,
                          loss_function,
                          loss_function_kwargs,
                          print_results=True,
                          track_calibration_results=True,
                          batching=False
                          ):

    # create a copy of links so the original isn't modified when it's re-added
    links = links.copy()
    
    # if infra is off street (i.e., the link should no longer be traversable)
    links['link_cost_override'] = False

    #set link costs accordingly
    #TODO think about how to make this work with the trip dates, thinking that it would loop through and update for each year
    # takes about 2 seconds to update the impedance factors
    updated_edge_costs = impedance_update(betas,betas_tup,
                          link_impedance_function,
                          base_impedance_col,
                          turn_impedance_function,
                          links,turns,turn_G)

    # make sure there are no negative edge weights
    if isinstance(updated_edge_costs,bool):
        if print_results:
            print('negative edge weights, skipping guess')
        return np.nan

    # TODO print the iteration number and pop member

    if batching == True:
        batch_ods = random.sample(ods,k=5)
        batch_ods = sorted(batch_ods,key=lambda x: x[-1])[::-1]
        batch_match_results = {tripid:item for tripid, item in match_results.items() if (item['origin_node'],item['destination_node'],item['trip_start_time']) in set(batch_ods)}
    else:
        batch_ods = ods
        batch_match_results = match_results

    #find leastdance path
    #NOTE right now I use start/end node to reduce the number of repeated shortest path searches
    trips_years = set()
    results_dict = {}
    for start_node, end_node, year in batch_ods:
        if year not in trips_years:
            trips_years.add(year) # add it to the years already looked at
            if (links['year'] > year).any(): # only then should we run this
                # print('Re-making network to year',year)
                
                # if infra is on street (i.e., the link is still traversable but the impedance doesn't apply)
                links.loc[links['year']>year,set_to_zero] = 0 
                links.loc[(links['year']>year) & (links.loc[:,set_to_inf]==1).any(axis=1),'link_cost_override'] = True
                
                # re-run network update
                updated_edge_costs = impedance_update(betas,betas_tup,
                        link_impedance_function,
                        base_impedance_col,
                        turn_impedance_function,
                        links,turns,turn_G)
            
                # # find the links with the new facilities and then find where they appear in the turns df
                # links_w_new_facilities = set(links.loc[(links['year']>year) & (links[set_to_inf]==1).any(axis=1),'linkid'].tolist())
                # turns_w_new_facilities = turns.loc[turns['source_linkid'].isin(links_w_new_facilities) | turns['target_linkid'].isin(links_w_new_facilities),['source_linkid','source_reverse_link','target_linkid','target_reverse_link']]
                # turns_w_new_facilities = {tuple(x):1e9 for x in turns_w_new_facilities.values} # set an absurdly high cost
                # updated_edge_costs.update(turns_w_new_facilities) # update the costs of those particular edges
                # #NOTE, this won't adress the case in which the first link is the bike path
                # nx.set_edge_attributes(turn_G,values=updated_edge_costs,name='weight')

        results_dict[(start_node,end_node)] = impedance_path(turns,turn_G,links,start_node,end_node)

    # results_dict = {(start_node,end_node):impedance_path(turns,turn_G,links,start_node,end_node) for start_node, end_node in batch_ods}

    #calculate the objective function
    #trim kwargs to only what's needed for the funtion 
    signature = inspect.signature(loss_function)
    arg_names = [param.name for param in signature.parameters.values()]
    loss_function_kwargs = {key:item for key,item in loss_function_kwargs.items() if key in arg_names}
    val_to_minimize = loss_function(results_dict,batch_match_results,**loss_function_kwargs)
    
    if print_results:
        print(list(betas.round(2))+[np.round(val_to_minimize,2)])

    #keep track of all the past betas used for visualization purposes
    #NOTE remember that the rendering needs to consider the population part of pso (so show pop size routes and then the converge process)
    #TODO actually implement this for disseration and because it would be cool in ppt to convey what is happening
    if track_calibration_results == True:
        past_vals.append(list(betas)+[val_to_minimize])
        # TODO: create a progress bar using this
        # if print_results:
        #     worker = len(past_vals) % 3 + 1
        #     iteration = len(past_vals)
        #     print('Current Worker',worker,'Current Iteration',iteration)

    return val_to_minimize

def save_calibration_result(betas_links,betas_turns,coefs,val):
    calibration_result = {}
    combined_betas = {**betas_links, **betas_turns}
    for key, item in combined_betas.items():
        calibration_result[item] = coefs[key]
    calibration_result['loss'] = np.array(val)
    calibration_result['beta_links'] = betas_links
    calibration_result['beta_turns'] = betas_turns
    calibration_result

########################################################################################

# Impedance Routing

########################################################################################

def impedance_path(turns,turn_G,links,o,d):
    #NOTE: without these it'll throw a 'the result is ambiguous error', in a future version we should probably
    # start storing linkids as str instead of int to prevent the consant floating back and forth
    o = int(o)
    d = int(d)
    #TODO, time this step too would it be better to do all at once vs one at a time?
    turn_G, virtual_starts, virtual_ends = modeling_turns.add_virtual_links_new(turns,turn_G,links,[o],[d])

    # NOTE replaced with astar but since there is no heurstic it behaves like dijkstra
    # however it terminates as soon as the target is reached
    # it ends up being faster to do it this way, but it'd be nice to have it set up
    # so that wasn't repeating path searches from the same origin
    edge_list = nx.astar_path(turn_G,source=o,target=d,weight='weight')
    actual_edge_list = list(zip(edge_list,edge_list[1:]))
    length = np.sum([turn_G.edges.get(edge)['weight'] for edge in actual_edge_list])

    edge_list = edge_list[1:-1] #chop off the virtual nodes added
    turn_G = modeling_turns.remove_virtual_links_new(turn_G,virtual_starts,virtual_ends)
    return {'length':np.round(length,1), 'edge_list':edge_list}


########################################################################################

# Network Impedance Update

########################################################################################

def impedance_update(betas:np.array,betas_tup:tuple,
                     link_impedance_function,
                     base_impedance_col:str,
                     turn_impedance_function,
                     links:pd.DataFrame,turns:pd.DataFrame,turn_G:nx.digraph):
    '''
    This function takes in the betas, impedance functions, and network objects
    and updates the network graph accordingly.

    Need to think about how to incorporate infrastructure availability into this
    '''
    #update link costs
    link_impedance_function(betas, betas_tup, links, base_impedance_col)
    
    # override the cost with 9e9 if future off-street facility
    # this effectively prevents routing w/o messing around with the network structure
    links.loc[links['link_cost_override']==True,'link_cost'] = 9e9

    #create cost dict (i think this is the fastest python way to do this?)
    tuple_index = tuple(zip(links['linkid'],links['reverse_link']))
    cost_dict = dict(zip(tuple_index,links['link_cost']))
    
    #costs are stored in the turn graph (only target matters, initial link cost is added during routing)
    turns['target_link_cost'] = turns[['target_linkid','target_reverse_link']].apply(lambda x: cost_dict.get(tuple(x.values),False),axis=1)

    #update turn costs
    turn_impedance_function(betas, betas_tup, turns)

    #cacluate new total cost
    turns['total_cost'] = (turns['target_link_cost'] + turns['turn_cost'])

    #check for negative link impedance
    if (links['link_cost'] < 0).any() | (turns['total_cost'] < 0).any():
        return False

    #update turn network graph with final cost
    cols = ['source_linkid','source_reverse_link','target_linkid','target_reverse_link','total_cost']
    updated_edge_costs = {((row[0],row[1]),(row[2],row[3])):row[4] for row in turns[cols].itertuples(index=False)}
    nx.set_edge_attributes(turn_G,values=updated_edge_costs,name='weight')

    return updated_edge_costs

def back_to_base_impedance(link_impedance_col,links,turns,turn_G):
    '''
    This function reverts the network graph back to base impedance (distance or travel time)
    with all turns as 0 cost
    '''
    #update link costs
    links['link_cost'] = links[link_impedance_col]
    cost_dict = dict(zip(links['linkid'],links['link_cost']))
    turns['target_link_cost'] = turns['target_linkid'].map(cost_dict)

    #cacluate new total cost
    turns['total_cost'] =  turns['target_link_cost'] #(turns['source_link_cost'] +

    #update turn network graph with final cost
    cols = ['source_linkid','source_reverse_link','target_linkid','target_reverse_link','total_cost']
    updated_edge_costs = {((row[0],row[1]),(row[2],row[3])):row[4] for row in turns[cols].itertuples(index=False)}
    nx.set_edge_attributes(turn_G,values=updated_edge_costs,name='weight')

########################################################################################

# Objective Functions

# TODO clean this up because these funcitons should be easily generalizable
# try using classes for this later

########################################################################################
import inspect

def jaccard_exact_total(results_dict,match_results,length_dict):
    '''
    Takes the sum of all the intersection lengths and
    divides it by all the union length.
    
    Intersections and union lengths from the jaccard exact
    loss function.
    '''

    #NOTE this would be the same across functions
    loss_values = []
    for tripid, item in match_results.items():
        #retrievs linkids in (linkid:int,reverse_link:boolean) format
        chosen = [tuple(row) for row in match_results[tripid]['matched_edges'].values]
        od = (item['origin_node'],item['destination_node'])
        modeled = results_dict[od]['edge_list']
        loss_value = jaccard_exact(chosen,modeled,length_dict)
        loss_values.append(loss_value)
    loss_values = np.array(loss_values)

    #but this part would be different
    # the higher the value, the better, so multiply by -1 because we're minimizing
    val_to_minimize = -1 * loss_values[:,0].sum() / loss_values[:,1].sum()

    return val_to_minimize

def jaccard_exact_mean(results_dict,match_results,length_dict):
    '''
    Takes the mean of the jaccard values.
    
    Intersections and union lengths from the jaccard exact
    loss function .
    '''

    #NOTE this would be the same across functions
    loss_values = []
    for tripid, item in match_results.items():
        #retrievs linkids in (linkid:int,reverse_link:boolean) format
        chosen = [tuple(row) for row in match_results[tripid]['matched_edges'].values]
        od = (item['origin_node'],item['destination_node'])
        modeled = results_dict[od]['edge_list']
        loss_value = jaccard_exact(chosen,modeled,length_dict)
        loss_values.append(loss_value)
    loss_values = np.array(loss_values)

    #but this part would be different
    # the higher the value, the better, so multiply by -1 because we're minimizing
    val_to_minimize = -1 * np.sum(loss_values[:,0] / loss_values[:,1]) / loss_values.shape[0]

    return val_to_minimize

def jaccard_buffer_total(results_dict,match_results,geo_dict):
    '''
    Takes the sum of all the intersection lengths and
    divides it by all the union length.
    
    Intersections and union lengths from the jaccard buffer
    loss function.
    '''
    
    #NOTE this would be the same across functions
    loss_values = []
    for tripid, item in match_results.items():
        #retrievs linkids in (linkid:int,reverse_link:boolean) format
        chosen = [tuple(row) for row in match_results[tripid]['matched_edges'].values]
        od = (item['origin_node'],item['destination_node'])
        modeled = results_dict[od]['edge_list']
        loss_value = jaccard_buffer(chosen,modeled,geo_dict,buffer_ft=50)
        loss_values.append(loss_value)
    loss_values = np.array(loss_values)

    #but this part would be different
    # the higher the value, the better, so multiply by -1 because we're minimizing
    val_to_minimize = -1 * loss_values[:,0].sum() / loss_values[:,1].sum()

    return val_to_minimize

def jaccard_buffer_mean(results_dict,match_results,geo_dict):
    '''
    Takes the mean of the jaccard values.
    
    Intersections and union lengths from the jaccard buffer
    loss function .
    '''
    
    #NOTE this would be the same across functions
    loss_values = []
    
    for tripid, item in match_results.items():
        #retrievs linkids in (linkid:int,reverse_link:boolean) format
        chosen = [tuple(row) for row in match_results[tripid]['matched_edges'].values]
        od = (item['origin_node'],item['destination_node'])
        modeled = results_dict[od]['edge_list']
        loss_value = jaccard_buffer(chosen,modeled,geo_dict,buffer_ft=50)
        loss_values.append(loss_value)
    loss_values = np.array(loss_values)

    #but this part would be different
    # the higher the value, the better, so multiply by -1 because we're minimizing
    val_to_minimize = -1 * np.sum(loss_values[:,0] / loss_values[:,1]) / loss_values.shape[0]

    return val_to_minimize

# def general_objective_function(loss_function,
#                                loss_function_kwargs:dict,
#                                match_results:dict,
#                                results_dict:dict):
#     '''
#     General form of objective function
#     '''

#     result = []

#     #trim kwargs to only what's needed for the funtion 
#     signature = inspect.signature(loss_function)
#     arg_names = [param.name for param in signature.parameters.values()]
#     loss_function_kwargs = {key:item for key,item in loss_function_kwargs.items() if key in arg_names}
    
#     for tripid, item in match_results.items():
#         #retrieve linkids in (linkid:int,reverse_link:boolean) format
#         chosen = [tuple(row) for row in match_results[tripid]['matched_edges'].values]
#         od = (item['origin_node'],item['destination_node'])
#         modeled_edges = results_dict[od]['edge_list']
#         #grab the loss value
#         loss_value = loss_function(chosen,modeled_edges,**loss_function_kwargs)
#         # append
#         result.append([tripid,loss_value])
    
#     result = np.array(result)

#     return result


# def jaccard_exact(chosen,other,length_dict):
#     '''
#     Returns the quotient of the intersection length divided by the union length
#     of the chosen and modeled results. No weighting by route length, just the average.
#     '''
    
#     #convert to sets
#     chosen = set([tuple(x) for x in chosen])
#     other = set([tuple(x) for x in other])

#     # Jaccard index (intersection over union) weighted by edge length
#     intersection =  list(set.intersection(chosen,other))
#     union = list(set.union(chosen,other))
    
#     # get lengths
#     intersection_length = [length_dict.get(linkid[0],False) for linkid in intersection]
#     union_length = [length_dict.get(linkid[0],False) for linkid in union]

#     intersection_length = np.array(intersection_length).sum()
#     union_length = np.array(union_length).sum()

#     jaccard_index = intersection_length / union_length

#     return jaccard_index

########################################################################################

# Loss Functions

########################################################################################


def jaccard_exact(chosen, other, length_dict):
    '''
    Returns intersection and union lengths between the
    chosen route and the other route using linkids as the
    basis for determining intersection and union.
    '''

    #convert to sets
    chosen = set([tuple(x) for x in chosen])
    other = set([tuple(x) for x in other])

    # Jaccard index (intersection over union) weighted by edge length
    intersection =  list(set.intersection(chosen,other))
    union = list(set.union(chosen,other))
    
    # get lengths
    intersection_length = [length_dict.get(linkid[0],False) for linkid in intersection]
    union_length = [length_dict.get(linkid[0],False) for linkid in union]

    intersection_length = np.array(intersection_length).sum()
    union_length = np.array(union_length).sum()

    # jaccard_index = intersection_length / union_length

    return (intersection_length,union_length)

def jaccard_buffer(chosen,other,geo_dict,buffer_ft=50):
    '''
    Returns the inersection and union area between the
    chosen route and the other route using the buffered geometries
    of each feature as the basis for determining intersection and union.
    
    Useful for trips where there are parallel links (think multi use path alongside road)
    where we may not be confident that map matching chose the precise route.

    Buffer amount should be small so as to only consider things directly adjacent to the 
    chosen route.

    Initially, this method would buffer the modeled route and then intersect it with the chosen
    route to see how much of the chosen route was covered, but this doesn't really follow jaccard index
    as it should penalize modeled routes that detour too much (unlikely but possible).
    '''
    # get the route geometry
    # NOTE geo dict MUST be projected coordinates
    chosen = get_route_line(chosen,geo_dict)
    other = get_route_line(other,geo_dict)
    # turn to linestring
    chosen = LineString(chosen)
    other = LineString(other)
    # buffer both by the specified distance
    chosen_buffer = chosen.buffer(buffer_ft)
    other_buffer = other.buffer(buffer_ft)
    # find the intersection and union area
    intersection_area = chosen_buffer.intersection(other_buffer).area
    union_area = chosen_buffer.union(other_buffer).area
    return (intersection_area,union_area)

def detour_factor(chosen,shortest,length_dict):

    chosen_length = [length_dict.get(linkid[0],False) for linkid in chosen]
    shortest_length = [length_dict.get(linkid[0],False) for linkid in shortest]

    chosen_length = np.array(chosen_length).sum()
    shortest_length = np.array(shortest_length).sum()

    diff = (chosen_length - shortest_length) / shortest_length

    return diff

# IN DEVELOPMENT PAST HERE FOR THIS SECTION

# These are helper functions for calculating frechet distance
def get_correct_link_direction(link_coord_seq,reverse_link):
    if reverse_link:
        return link_coord_seq[::-1]
    else:
        return link_coord_seq

def get_route_line(route,geo_dict): 
    #get all the links
    route = [get_correct_link_direction(geo_dict.get(linkid[0],False).coords,linkid[1]) for linkid in route]
    #remove the last point of each link except for the last one
    route = [x[0:-1] if idx != len(route) - 1 else x for idx, x in enumerate(route)]
    #flatten to produce one linestring
    route = [x for xs in route for x in xs]
    return route

def frechet_distance(chosen,modeled,geo_dict):
    '''
    Returns the frechet distance between the chosen and the modeled route. No weighting by route length, just the average.
    '''
    modeled = get_route_line(modeled,geo_dict)
    chosen = get_route_line(chosen,geo_dict)
    result = similaritymeasures.frechet_dist(modeled,chosen)
    return -result

def trace_difference(chosen,modeled,geo_dict,tripid,trace_dict):
    '''
    Returns the frechet distance between the chosen and the modeled route. No weighting by route length, just the average.
    '''
    modeled = get_route_line(modeled,geo_dict)
    # chosen = get_route_line(chosen,geo_dict)
    trace = trace_dict[tripid]
    result = similaritymeasures.frechet_dist(modeled,trace)
    return -result

def first_preference_recovery(match_results,results_dict,**kwargs):
    '''
    Seen in Meister et al. 2024: https://doi.org/10.1016/j.jcmr.2024.100018

    "FPR is the percentage of correct predictions assuming that the predicted choice
    is the one with the highest choice probability"

    This has been modified to look at the similarity between the modeled route
    and the chosen route instead of choice probabilities. A correct modeled route will
    contain all or most of the links included in the map matched trip. An overlap threshold
    controls what percentage of intersection between the chosen and modeled route is
    needed to be considered a match. A 100% overlap threshold means that the modeled
    route contains all of the links included in the chosen route. A 0% overlap threshold
    means that the modeled route doesn't need to contain any of links in the chosen route
    to count. Length is used to weight the overlap appropriately (i.e., missing short links
    isn't as big of a deal as long ones).

    This function returns a 1 or a 0 depending on the overlap threshold set. The average is
    taken across all the trips, hence it will be between 0 and 1.

    '''

    result = []
    
    for tripid, item in match_results.items():

        start_node = item['origin_node']
        end_node = item['destination_node']

        #retrieve linkids in (linkid:int,reverse_link:boolean) format
        chosen = [tuple(row) for row in match_results[tripid]['matched_edges'].to_numpy()]
        modeled_edges = results_dict[(start_node,end_node)]['edge_list']

        #get lengths (non-directional)
        chosen_length = np.sum([kwargs['length_dict'][linkid[0]] for linkid in chosen])
        #modeled_length = np.sum([kwargs['length_dict'][linkid[0]] for linkid in modeled_edges])

        #convert to sets
        chosen = set(chosen)
        modeled_edges = set(modeled_edges)

        #find intersection of sets
        shared = list(set.intersection(chosen,modeled_edges))

        #find intersection length
        intersection_length = np.sum([kwargs['length_dict'][linkid[0]] for linkid in shared])

        # result.append((intersection_length,chosen_length))
        #result.append((intersection_length,chosen_length))

        overlap_calc = intersection_length / chosen_length

        if overlap_calc >= kwargs['overlap_threshold']:
            result.append(tripid)
        # else:
        #     result.append(tripid,False)
    
    #TODO another interpretation could be percentage of all currect?
    result = np.array(result)
    #result = result.sum() / len(result)

    # if kwargs['standardize']:
    #     #average intersect over chosen length
    #     result = np.mean(result[:,0] / result[:,1])
    # else:
    #     #total intersect over total chosen length
    #     result = np.sum(result[:,0]) / np.sum(result)
    
    #return negative result because we want to minimize
    return result


#TODO use frechet area instead https://towardsdatascience.com/gps-trajectory-clustering-with-python-9b0d35660156
# def frechet(match_results,results_dict,**kwargs):
#     '''
#     IN DEVELOPMENT
#     '''

#     result = []
    
#     for tripid, item in match_results.items():

#         start_node = item['origin_node']
#         end_node = item['destination_node']

#         #retrieve tuples of the format (linkid:int,reverse_link:boolean)
#         chosen = [tuple(row) for row in match_results[tripid]['matched_edges'].to_numpy()]
#         #shortest = [tuple(row) for row in match_results[linkid]['shortest_edges'].to_numpy()]
#         modeled = results_dict[(start_node,end_node)]['edge_list']

#         chosen_geo = [retrieve_coordinates(link,kwargs['geo_dict']) for link in chosen]
#         modeled_geo = [retrieve_coordinates(link,kwargs['geo_dict']) for link in modeled]

#         #turn to a single line
#         chosen_geo = LineString(np.vstack(chosen_geo))
#         modeled_geo = LineString(np.vstack(modeled_geo))

#         #simplify with rdp
#         chosen_coords = np.array(chosen_geo.simplify(kwargs['rdp_ft']).coords)
#         modeled_coords = np.array(modeled_geo.simplify(kwargs['rdp_ft']).coords)

#         #find frechet distance
#         # see https://github.com/topics/trajectory-similarity for documentation
#         frechet_distance = similaritymeasures.frechet_dist(chosen_coords,modeled_coords)

#         result.append(frechet_distance)

#     result = np.array(result).mean()
    
#     #can minimize total frechet distance or an average value
#     #don't take negative because we're already taking minimum
#     return result

#retrieve coordinates, revesing coordinate sequence if neccessary
def retrieve_coordinates(link,geo_dict):
    line = np.array(geo_dict[link[0]].coords)
    if link[1] == True:
        line = line[::-1]
    return line

########################################################################################

# Impedance Functions

########################################################################################

'''
Currently works with binary and numeric variables. Categorical data will have to be
cast into a different format for now.

Link impedance is weighted by the length of the link, turns are just the impedance associated
'''

# def link_impedance_function_1(betas,beta_links,links,base_impedance_col):
#     #prevent mutating the original links gdf
#     links = links.copy()
    
#     #multiplier = np.zeros(links.shape[0])
#     links["multiplier"] = 0
    
#     links.loc[links['link_type']=='road','multiplier'] = \
#     (beta_links['lanes'] * links['lanes']) + \
#     (beta_links['speed'] * links['speed']) + \
#     (beta_links['bike_lane'] * links['bike_lane'])
    
#     links.loc[links['link_type']=='bike','multiplier'] = \
#     (beta_links[])

#     #grade
#     links.loc[:,'multiplier'] = links.loc[:,'multiplier'] + (beta_links[''])


#     if len(beta_links) > 0:
#         #assumes that these effects are additive
#         #TODO i think this can be done as a matrix product
#         for key, item in beta_links.items():
#             multiplier = multiplier + (betas[key] * links[item].values)
    
#         links['link_cost'] = links[base_impedance_col] * (1 + multiplier) #removeing the + 1 for now

#     else:
#         links['link_cost'] = links[base_impedance_col]

#     return links


def link_impedance_function(betas:np.array,betas_tup:tuple,links:pd.DataFrame,base_impedance_col:str):
    '''
    Default link impedance function. Assumes that link impedance factors are additive
    and increase/decrease link impedance proportional to the link's distance/travel time
    '''
    
    #set up a multiplier a zero that gets addded/subtracted to
    multiplier = np.zeros(links.shape[0])

    # retrieve col names of the links and the positions in the betas array
    betas_links = [(idx,x['col']) for idx, x in enumerate(betas_tup) if x['type']=='link']

    # check to see if there are any link impedances at all
    if len(betas_links) > 0:
        #assumes that these effects are additive
        for idx, col in betas_links:
            multiplier = multiplier + (betas[idx] * links[col].values)
        #stores the multiplier
        links['multiplier'] = multiplier
        links['link_cost'] = links[base_impedance_col] * (1 + multiplier) #removing the + 1 for now
    else:
        print('No link impedance factors assigned')
        links['link_cost'] = links[base_impedance_col]

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

    # #not sure if needed
    # turns['turn_cost'] = turns['turn_cost'].astype(float)

########################################################################################

# Post Calibration Functions

########################################################################################

def post_calibration_routing(
        links,
        turns,
        turn_G,
        base_impedance_col,
        betas,
        betas_links,
        betas_turns,
        ods
        #results_dict
    ):

    #base_impedance_col = "travel_time_min"
    back_to_base_impedance(base_impedance_col,links,turns,turn_G)
    impedance_update(betas,betas_links,betas_turns,
                            link_impedance_function,
                            base_impedance_col,
                            turn_impedance_function,
                            links,turns,turn_G)
    
    #find shortest path
    routing_results = {(start_node,end_node):impedance_path(turns,turn_G,links,start_node,end_node) for start_node, end_node in ods}
    return routing_results
    #results_dict.update(routing_results)


########################################################################################

# Visualization and QAQC Tools

########################################################################################

import folium
import geopandas as gpd
from folium.plugins import MarkerCluster, PolyLineTextPath
from folium.map import FeatureGroup
from shapely.ops import Point, MultiLineString

def construct_line_dict(keys,result_dict,geo_dict):
    """
    Function for creating line dictionary for visualization functions that assumes
    a dictionary with keys corresponding to a dataframe of the link sequence
    """
    line_dict = {}
    for key in keys:
        if key == 'matched_edges':
            new_key = 'Chosen'
        elif key == 'shortest_edges':
            new_key = 'Shortest'
        elif key == 'modeled_edges':
            new_key = 'Modeled'
        else:
            new_key = key
        
        line_dict[new_key] = {
            'links': result_dict[key].values,
            'coords': get_route_line(result_dict[key].values,geo_dict),
        }
    return line_dict

def add_metrics_to_tooltip(line_dict,length_dict,geo_dict):
    '''
    Function used to add various overlap metrics to the line dictionary
    '''

    chosen = line_dict['Chosen']['links']
    shortest = line_dict['Shortest']['links']

    line_dict['Chosen']['detour_pct'] = detour_factor(chosen,shortest,length_dict)

    for key, item in line_dict.items():
        if key == 'Chosen':
            continue
        line = item['links']
        line_dict[key]['jaccard index'] = round(jaccard(chosen,line,length_dict),3)
        line_dict[key]['frechet_dist'] = round(frechet_distance(chosen,line,geo_dict),3)
        line_dict[key]['buffer_dist'] = round(buffer_overlap(chosen,line,geo_dict),3)
        line_dict[key]['detour_pct'] = round(detour_factor(line,shortest,length_dict),3)

    return line_dict

def basic_three_viz(tripid,results_dict,crs,length_dict,geo_dict,tile_info_dict):
    line_dict = construct_line_dict(['matched_edges','shortest_edges','modeled_edges'],results_dict[tripid],geo_dict)
    line_dict = add_metrics_to_tooltip(line_dict,length_dict,geo_dict)
    mymap = visualize_three(tripid,line_dict,results_dict[tripid]['coords'],crs,tile_info_dict)
    return mymap

def retrieve_geos(x,y,results_dict,links,latlon=False):
    '''
    Pulls out the chosen, shortest, and modeled geometry from results dict that intersect
    with the inputed coordinate
    '''
    links = links.copy()
    links.set_index(['linkid','reverse_link'],inplace=True)
    feature = Point(x,y).buffer(100)
    trips_intersecting_geo = []
    for tripid, item in results_dict.items():
        #get line geo
        shortest_geo = links.loc[[tuple(x) for x in results_dict[tripid]['matched_edges'].values],'geometry'].tolist()
        #test if intersecting
        if MultiLineString(shortest_geo).intersects(feature):
            trips_intersecting_geo.append(tripid)
    return trips_intersecting_geo

# def visualize_three_no_legend(tripid,results_dict,links,tile_info_dict,shortest_overlap,modeled_overlap):
#     '''
#     Standard visualization of the chosen/shortest/modeled lines.
#     Provide GeoSeries for each.
#     '''
    
#     #retrieve link geo from results dict
#     chosen_geo = links.loc[[tuple(x) for x in results_dict[tripid]['matched_edges'].values],'geometry']
#     shortest_geo = links.loc[[tuple(x) for x in results_dict[tripid]['shortest_edges'].values],'geometry']
#     modeled_geo = links.loc[[tuple(x) for x in results_dict[tripid]['modeled_edges'].values],'geometry']

#     #turn into geoseries and project to project coordinates
#     chosen_line = gpd.GeoSeries(chosen_geo,crs=links.crs)
#     shortest_line = gpd.GeoSeries(shortest_geo,crs=links.crs)
#     modeled_line = gpd.GeoSeries(modeled_geo,crs=links.crs)
    
#     #reproj
#     chosen_line = chosen_line.to_crs(epsg='4326')
#     shortest_line = shortest_line.to_crs(epsg='4326')
#     modeled_line = modeled_line.to_crs(epsg='4326')

#     #start_pt
#     start_pt = list(chosen_line.iloc[0].coords)[0]
#     end_pt = list(chosen_line.iloc[-1].coords)[-1]

#     # reproject
#     x_mean = chosen_line.unary_union.centroid.x
#     y_mean = chosen_line.unary_union.centroid.y

#     # Create a Folium map centered around the mean of the GPS points
#     center = [y_mean,x_mean]
#     mymap = folium.Map(location=center,
#                        zoom_start=12,
#                        control_scale=True,
#                        tiles=None)
#     # add tiles
#     folium.TileLayer(**tile_info_dict).add_to(mymap)

#     # Convert GeoDataFrames to GeoJSON
#     chosen_line_geojson = chosen_line.to_json()
#     shortest_line_geojson = shortest_line.to_json()
#     modeled_line_geojson = modeled_line.to_json()

#     # Add GeoJSON data to FeatureGroups
#     folium.GeoJson(chosen_line_geojson, name='Chosen Path',
#                 style_function=lambda x: {'color': '#fc8d62', 'weight': 12, 'opacity':0.5}).add_to(mymap)
#     folium.GeoJson(shortest_line_geojson, name='Shortest Path',
#                 #tooltip=f"Overlap: {shortest_overlap:.5f}",
#                 style_function=lambda x: {'color': '#66c2a5', 'weight': 8, 'opacity':0.5}).add_to(mymap)
#     folium.GeoJson(modeled_line_geojson, name='Modeled Path',
#                 #tooltip=f"Overlap: {modeled_overlap:.5f}",
#                 style_function=lambda x: {'color': '#8da0cb','weight': 8, 'opacity':0.5}).add_to(mymap)

#     # Add start and end points with play and stop buttons
#     start_icon = folium.Icon(color='green',icon='play',prefix='fa')
#     end_icon = folium.Icon(color='red',icon='stop',prefix='fa')
#     folium.Marker(location=[start_pt[1], start_pt[0]],icon=start_icon).add_to(mymap)
#     folium.Marker(location=[end_pt[1], end_pt[0]],icon=end_icon).add_to(mymap)

#     # Add layer control to toggle layers on/off
#     folium.LayerControl(collapsed=False).add_to(mymap)

#     legend_html = f'''    
#     {{% macro html(this, kwargs) %}}               
#     <div style="
#         position: fixed; 
#         bottom: 50px; left: 10px; width: 160px; height: 130px; 
#         z-index:9999; font-size:14px; background-color: white; 
#         border:2px solid grey; padding: 10px; opacity: 0.9;">
#         <p>Trip ID: {tripid}</p>
#         <p><span style="display:inline-block; background-color:#fc8d62; width:50px; height:10px; vertical-align:middle;"></span>&emsp;Chosen</p>
#         <p><span style="display:inline-block; background-color:#66c2a5; width:50px; height:10px; vertical-align:middle;"></span>&emsp;Shortest</p>
#         <p><span style="display:inline-block; background-color:#8da0cb; width:50px; height:10px; vertical-align:middle;"></span>&emsp;Modeled</p>
#     {{% endmacro %}}
#     </div>
#     '''

#     legend = branca.element.MacroElement()
#     legend._template = branca.element.Template(legend_html) 
#     mymap.get_root().add_child(legend)

#     return mymap


import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Extract colors from a ColorBrewer scheme (e.g., 'Set2')
# # Convert them to HEX format if needed
colorbrewer_hex = [colors.to_hex(c) for c in plt.get_cmap('Set2').colors]

def visualize_three(tripid,match_dict,modeled_dicts,geo_dict,coords_dict,crs,tile_info_dict):
    '''
    Takes in one or more line strings, an origin, and a destination and plots them
    '''

    # handle the chosen and the shortest (these should always by provided)
    chosen = match_dict[tripid]['matched_edges'].values
    shortest = match_dict[tripid]['shortest_edges'].values

    chosen = get_route_line(chosen,geo_dict)
    shortest = get_route_line(shortest,geo_dict)
    
    #get start/end and center from the chosen geometry
    line_geo = LineString(chosen)
    line_geo = gpd.GeoSeries(line_geo,crs=crs)
    line_geo = line_geo.to_crs(epsg='4326')
        
    start_pt = list(line_geo.iloc[0].coords)[0]
    end_pt = list(line_geo.iloc[-1].coords)[-1]
    x_mean = line_geo.unary_union.centroid.x
    y_mean = line_geo.unary_union.centroid.y

    # Create a Folium map centered around the mean of the GPS points
    center = [y_mean,x_mean]
    mymap = folium.Map(location=center,
                       zoom_start=15,
                       control_scale=True,
                       tiles=None)
    # add tiles
    folium.TileLayer(**tile_info_dict).add_to(mymap)
    
    # init legend
    legend_lines = ""
    idx = 0 # for the color selection

    # add chosen to folium
    chosen = gpd.GeoDataFrame(
        {'length':match_dict[tripid]['chosen_length'],'detour':match_dict[tripid]['chosen_detour'],'geometry':LineString(chosen)},
        index = [0],
        crs = crs
    ).to_crs('epsg:4326').to_json()
    tooltip = folium.GeoJsonTooltip(fields= ['length','detour'])
    folium.GeoJson(chosen,name='Chosen',
                   style_function=lambda x,
                   color=colorbrewer_hex[idx]: {'color': color, 'weight': 12, 'opacity':0.5},
                   tooltip=tooltip
                   ).add_to(mymap)
    label = 'Chosen'
    legend_lines += f'''
    <p><span style="display:inline-block; background-color:{colorbrewer_hex[idx]}; width:50px; height:10px; vertical-align:middle;"></span>&emsp;{label}</p>
    '''
    idx += 1

    # add shortest to folium
    shortest = gpd.GeoDataFrame(
        {'length':match_dict[tripid]['shortest_length'],'jaccard':match_dict[tripid]['shortest_jaccard'],'buffer':match_dict[tripid]['shortest_buffer'],'geometry':LineString(shortest)},
        index = [0],
        crs = crs
    ).to_crs('epsg:4326').to_json()
    tooltip = folium.GeoJsonTooltip(fields= ['length','jaccard','buffer'])
    folium.GeoJson(shortest,name='Shortest',
                   style_function=lambda x,
                   color=colorbrewer_hex[idx]: {'color': color, 'weight': 12, 'opacity':0.5},
                   tooltip=tooltip
                   ).add_to(mymap)
    label = 'Shortest'
    legend_lines += f'''
    <p><span style="display:inline-block; background-color:{colorbrewer_hex[idx]}; width:50px; height:10px; vertical-align:middle;"></span>&emsp;{label}</p>
    '''
    idx += 1
    
    # add modeled to folium
    # TODO make it so it accepts many different modeled results
    # TODO also make it optional incase we just want to examine shortest vs chosen
    for model_name, modeled_dict in modeled_dicts:
        if idx > len(colorbrewer_hex) - 1:
            colore = 'grey'
        else:
            color = colorbrewer_hex[idx]
        
        modeled = modeled_dict[tripid]['modeled_edges'].values
        modeled = get_route_line(modeled,geo_dict)
        modeled = gpd.GeoDataFrame(
            {'name':model_name,'length':modeled_dict[tripid]['modeled_length'],'detour':modeled_dict[tripid]['modeled_detour'],'jaccard':modeled_dict[tripid]['modeled_jaccard'],'buffer':modeled_dict[tripid]['modeled_buffer'],'geometry':LineString(modeled)},
            index = [0],
            crs = crs
        ).to_crs('epsg:4326').to_json()
        tooltip = folium.GeoJsonTooltip(fields= ['name','length','detour','jaccard','buffer'])
        folium.GeoJson(modeled,name=model_name,
                    style_function=lambda x,
                    color=color: {'color': color, 'weight': 12, 'opacity':0.5},
                    tooltip=tooltip
                    ).add_to(mymap)
        legend_lines += f'''
        <p><span style="display:inline-block; background-color:{color}; width:50px; height:10px; vertical-align:middle;"></span>&emsp;{model_name}</p>
        '''
        idx += 1

    # for idx, (label, line) in enumerate(line_dict.items()):
    #     line['geometry'] = LineString(line['coords'])
    #     line_gdf = {key:item for key,item in line.items() if key not in ['coords','links']}
    #     line_gdf = gpd.GeoDataFrame(line_gdf,index=[0],crs=crs)
    #     line_gdf = line_gdf.to_crs(epsg='4326')
        
    #     if label == 'Chosen':
    #         # start_pt = list(line['coords'])[0]
    #         # end_pt = list(line['coords'])[-1]
    #         x_mean = line_gdf.geometry.unary_union.centroid.x
    #         y_mean = line_gdf.geometry.unary_union.centroid.y
    #         line_gdf = line_gdf.to_json()
    #         line_gdf = folium.GeoJson(line_gdf, name=label,
    #             style_function=lambda x, color=colorbrewer_hex[idx]: {'color': color, 'weight': 12, 'opacity':0.5},
    #             tooltip=f"Chosen Route").add_to(mymap)
            
    #     else:
    #         line_gdf = line_gdf.to_json()
    #         tooltip = folium.GeoJsonTooltip(
    #             fields= ['jaccard index','frechet_dist','buffer_dist','detour_pct']
    #         )
            
    #         line_gdf = folium.GeoJson(line_gdf, name=label,
    #             style_function=lambda x, color=colorbrewer_hex[idx]: {'color': color, 'weight': 8, 'opacity':0.5},
    #             tooltip=tooltip).add_to(mymap)
        
    #     legend_lines += f'''
    #     <p><span style="display:inline-block; background-color:{colorbrewer_hex[idx]}; width:50px; height:10px; vertical-align:middle;"></span>&emsp;{label}</p>
    #     '''
    
    # add the trace coordinates so we can see if there is a map matching error
    coords = coords_dict[tripid]
    coords = [Point(x) for x in coords]
    coords = gpd.GeoDataFrame({'geometry':coords},crs=crs)
    coords.to_crs('epsg:4326',inplace=True)
    coords = coords.to_json()
    coords = folium.GeoJson(coords,
                            name='coords',
                            marker=folium.Circle(radius=10, fill_color="grey", fill_opacity=1, color="black", weight=0),
                            show=False).add_to(mymap)

    # Add start and end points with play and stop buttons
    start_icon = folium.Icon(color='green',icon='play',prefix='fa')
    end_icon = folium.Icon(color='red',icon='stop',prefix='fa')
    folium.Marker(location=[start_pt[1], start_pt[0]],icon=start_icon).add_to(mymap)
    folium.Marker(location=[end_pt[1], end_pt[0]],icon=end_icon).add_to(mymap)
    
    # Add layer control to toggle layers on/off
    folium.LayerControl(collapsed=False).add_to(mymap)
    legend_html = f'''    
    {{% macro html(this, kwargs) %}}               
    <div style="
        position: fixed; 
        bottom: 50px; left: 10px; width: auto; height: auto; 
        z-index:9999; font-size:14px; background-color: white; 
        border:2px solid grey; padding: 10px; opacity: 0.9;">
        <p>Trip ID: {tripid}</p>
        {legend_lines}
    {{% endmacro %}}
    </div>
    '''
    legend = branca.element.MacroElement()
    legend._template = branca.element.Template(legend_html) 
    mymap.get_root().add_child(legend)

    return mymap

def uniquify(path):
    counter = 1
    original_stem = path.stem
    extension = path.suffix

    while path.exists():
        path = path.parent / (original_stem + f" ({str(counter)})" + extension)
        counter += 1

    return path


# def visualize_route_attributes(
#         tripid,
#         results_dict, # contains the edge list etc
#         matched_gdf,
#         modeled_gdf,
#         links_df,
#         turns_df,
#         nodes_df,
#         route_attribute_cols,
#         ):

#     '''
#     This function displays the matched vs shortest/modeled route for a particular trip
#     It also displays the trip characteristics side be side and plots the any signalized
#     intersections and stressful turns passed through.
#     '''

#     # Create copies to prevent alteration
#     matched_gdf = matched_gdf.copy()
#     modeled_gdf = modeled_gdf.copy()

#     # Subset data to relevant trip
#     matched_gdf = matched_gdf[matched_gdf['tripid']==tripid]
#     modeled_gdf = modeled_gdf[modeled_gdf['tripid']==tripid]
    
#     # Create a Folium map centered around the mean of the matched route
#     minx, miny, maxx, maxy = matched_gdf.to_crs(epsg='4326').total_bounds
#     x_mean = (maxx - minx) / 2 + minx
#     y_mean = (maxy - miny) / 2 + miny
#     center = [y_mean,x_mean]
#     m = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")
    
#     # Add GeoJSON data to FeatureGroups
#     folium.GeoJson(matched_gdf.to_crs(epsg='4326').to_json(),
#                    name='Matched',
#                    tooltip=folium.GeoJsonTooltip(fields=route_attribute_cols),
#                    style_function=lambda x: {'color': 'red'}).add_to(m)
    
#     folium.GeoJson(modeled_gdf.to_crs(epsg='4326').to_json(),
#                    name='Modeled',
#                    tooltip=folium.GeoJsonTooltip(fields=route_attribute_cols),
#                    style_function=lambda x: {'color': 'blue'}).add_to(m)

#     # Get the start and end points
#     start_node = results_dict[tripid]['origin_node']
#     start_node = nodes_df.to_crs('epsg:4326').loc[nodes_df['N']==start_node,'geometry'].item()

#     end_node = results_dict[tripid]['destination_node']
#     end_node = nodes_df.to_crs('epsg:4326').loc[nodes_df['N']==end_node,'geometry'].item()

#     # Add start and end points with play and stop buttons to map
#     #start_icon = folium.Icon(color='green',icon='play',prefix='fa')
#     #end_icon = folium.Icon(color='red',icon='stop',prefix='fa')
#     folium.Marker(location=[start_pt.y, start_pt.x],color='green').add_to(m)
#     folium.Marker(location=[end_pt.y, end_pt.x],color='red').add_to(m)

#     # Add signals and turns for matched route

#     edges = match_dict[tripid]['edges']
#     list_of_edges = list(zip(edges['linkid'],edges['reverse_link']))
#     list_of_turns = [(list_of_edges[i][0],list_of_edges[i][1],list_of_edges[i+1][0],list_of_edges[i+1][1]) for i in range(0,len(list_of_edges)-1)]


#     #from these we want to get the locations and number of singalized intersections and stressful crossing passed through
    
#     df_of_turns = pd.DataFrame(list_of_turns,columns=['source_linkid','source_reverse_link','target_linkid','target_reverse_link'])
#     subset = pseudo_df.merge(df_of_turns,on=['source_linkid','source_reverse_link','target_linkid','target_reverse_link'])

#     # from this subset we can get the right node ids
#     #TODO turns should be by edges probably?
#     #turns = subset[['source_B','turn_type']]
#     signals = subset.loc[subset['signalized']==True,'source_B'].value_counts()
#     two_way_stops = subset.loc[subset['unsignalized']==True,'source_B'].value_counts()

#     #and then get the correct rows of the gdf
#     #turns = nodes.merge(signals,left_on='N',right_on='')
#     signals = nodes.merge(signals,left_on='N',right_index=True)
#     signals.columns = ['N','geometry','num_times']
#     two_way_stops = nodes.merge(two_way_stops,left_on='N',right_index=True)
#     two_way_stops.columns = ['N','geometry','num_times']

#     # get the start and end point for plotting
#     start_N = gdf.loc[gdf['tripid']==tripid,'start'].item()
#     start_pt = nodes.to_crs('epsg:4326').loc[nodes['N']==start_N,'geometry'].item()
#     end_N = gdf.loc[gdf['tripid']==tripid,'end'].item()
#     end_pt = nodes.to_crs('epsg:4326').loc[nodes['N']==end_N,'geometry'].item()





 
#    # Add FeatureGroups to the map


#    if signals.shape[0] > 0:
#       signals_geojson = signals.to_crs(epsg='4326').to_json()
#       signals_fg = FeatureGroup(name='Signals')

#       folium.GeoJson(
#       signals_geojson,
#       name="Traffic Signal Turn Movement",
#       marker=folium.Circle(radius=20, fill_color="red", fill_opacity=.5, color="black", weight=1),
#       tooltip=folium.GeoJsonTooltip(fields=['N','num_times']),
#       popup=folium.GeoJsonPopup(fields=['N','num_times']),
#       #    style_function= lambda feature: {
#       #        'fillColor': colormap(feature['properties']['speed_mph']),
#       #    },
#       highlight_function=lambda feature: {"color":"yellow","weight":3}
#       ).add_to(signals_fg)
#       signals_fg.add_to(mymap)

#    if two_way_stops.shape[0] > 0:
#       two_way_stops_geojson = two_way_stops.to_crs(epsg='4326').to_json()
#       two_way_stops_fg = FeatureGroup(name='Two Way Stop (chosen)')

#       folium.GeoJson(
#       two_way_stops_geojson,
#       name="Two Way Stop with High Stress Cross Street",
#       marker=folium.Circle(radius=20, fill_color="yellow", fill_opacity=.5, color="black", weight=1),
#       tooltip=folium.GeoJsonTooltip(fields=['N','num_times']),
#       popup=folium.GeoJsonPopup(fields=['N','num_times']),
#       #    style_function= lambda feature: {
#       #        'fillColor': colormap(feature['properties']['speed_mph']),
#       #    },
#       highlight_function=lambda feature: {"color":"yellow","weight":3}
#       ).add_to(two_way_stops_fg)

#       two_way_stops_fg.add_to(mymap)




#    #autofit content not in this version?
#    #folium.FitOverlays().add_to(mymap)

#    # Add layer control to toggle layers on/off
#    folium.LayerControl().add_to(mymap)

#    #retrive overlap
#    exact_overlap = gdf.loc[gdf['tripid']==tripid,'shortest_exact_overlap_prop'].item()
#    buffer_overlap = gdf.loc[gdf['tripid']==tripid,'shortest_buffer_overlap'].item()

#    attr = gdf.loc[gdf['tripid']==tripid].squeeze()

#    # Add legend with statistics
#    legend_html = f'''
#    <div style="position: fixed; 
#             bottom: 5px; left: 5px; width: 300px; height: 500px; 
#             border:2px solid grey; z-index:9999; font-size:14px;
#             background-color: white;
#             opacity: 0.9;">
#    &nbsp; <b>Tripid: {tripid}</b> <br>
#    &nbsp; Start Point &nbsp; <i class="fa fa-play" style="color:green"></i><br>
#    &nbsp; End Point &nbsp; <i class="fa fa-stop" style="color:red"></i><br>
#    &nbsp; Exact Overlap: {exact_overlap*100:.2f}% <br>
#    &nbsp; Buffer Overlap: {buffer_overlap*100:.2f}% <br>

#    &nbsp; Trip Type: {attr['trip_type']} <br>
#    &nbsp; Length (mi): {attr['length_ft']/5280:.0f} <br>
#    &nbsp; Age: {attr['age']} <br>
#    &nbsp; Gender: {attr['gender']} <br>
#    &nbsp; Income: {attr['income']} <br>
#    &nbsp; Ethnicity: {attr['ethnicity']} <br>
#    &nbsp; Cycling Frequency: {attr['cyclingfreq']} <br>
#    &nbsp; Rider History: {attr['rider_history']} <br>
#    &nbsp; Rider Type: {attr['rider_type']} <br><br>

#    &nbsp; Residential %: {attr['highway.residential']*100:.2f}% <br>
#    &nbsp; Secondary %: {attr['highway.secondary']*100:.2f}% <br>
#    &nbsp; Tertiary %: {attr['highway.tertiary']*100:.2f}% <br>

#    &nbsp; # of bridges: {int(attr['bridge'])} <br>
#    &nbsp; # of left turns: {int(attr['left'])} <br>
#    &nbsp; # of straight turns: {int(attr['straight'])} <br>
#    &nbsp; # of right turns: {int(attr['right'])} <br>
#    &nbsp; # of stressful turns: {int(attr['unsignalized'])} <br>
#    &nbsp; # of signalized turns: {int(attr['signalized'])} <br>

#    </div>
#    '''

#    mymap.get_root().html.add_child(folium.Element(legend_html))

#    # Save the map to an HTML file or display it in a Jupyter notebook
#    #mymap.save('map.html')
#    # mymap.save('/path/to/save/map.html')  # Use an absolute path if needed
#    return mymap  # Uncomment if you are using Jupyter notebook

   #TODO add in the legend with trip info and then we're golden




# DEPRECATED
# def loss_function(betas,betas_links,betas_turns,links,
#                        pseudo_links,pseudo_G,
#                        matched_traces,link_impedance_function,
#                        turn_impedance_function,exact,follow_up):

#     #use initial/updated betas to calculate link costs
#     print('setting link costs')
#     links = link_impedance_function(betas, betas_links, links)
#     cost_dict = dict(zip(links['linkid'],links['link_cost']))
    
#     #add link costs to pseudo_links
#     pseudo_links['source_link_cost'] = pseudo_links['source_linkid'].map(cost_dict)
#     pseudo_links['target_link_cost'] = pseudo_links['target_linkid'].map(cost_dict)

#     #use initial/updated betas to calculate turn costs
#     print('setting turn costs')
#     pseudo_links = turn_impedance_function(betas, betas_turns, pseudo_links)

#     #add the source edge, target edge, and turn costs
#     #TODO experiment with multiplying the turn cost
#     pseudo_links['total_cost'] = pseudo_links['source_link_cost'] + pseudo_links['target_link_cost'] + pseudo_links['turn_cost']

#     #only keep link with the lowest cost
#     print('finding lowest cost')
#     costs = pseudo_links.set_index(['source','target'])['total_cost']

#     #update edge weights
#     print('updating edge weights')
#     nx.set_edge_attributes(pseudo_G,values=costs,name='weight')

#     #update edge ids (what was this for?)
    
#     #do shortest path routing
#     shortest_paths = {}
#     print(f'Shortest path routing with coefficients: {betas}')    
#     for source, targets in matched_traces.groupby('start')['end'].unique().items():

#         #add virtual links to pseudo_G
#         pseudo_G, virtual_edges = modeling_turns.add_virtual_links(pseudo_links,pseudo_G,source,targets)
        
#         #perform shortest path routing for all target nodes from source node
#         #(from one to all until target node has been visited)
#         for target in targets:  
#             #cant be numpy int64 or throws an error
#             target = int(target)
            
#             try:
#                 #TODO every result is only a start node, middle node, then end node
#                 length, node_list = nx.single_source_dijkstra(pseudo_G,source,target,weight='weight')
#             except:
#                 print(source,target)
#                 length, node_list = nx.single_source_dijkstra(pseudo_G,source,target,weight='weight')

#             #get edge list
#             edge_list = node_list[1:-1]

#             #get geometry from edges
#             modeled_edges = links.set_index(['source','target']).loc[edge_list]

#             # modeled_edges = links.merge(linkids.loc[edge_list],on=['linkid','reverse_link'],how='inner')
#             # modeled_edges = gpd.GeoDataFrame(modeled_edges,geometry='geometry')

#             shortest_paths[(source,target)] = {
#                 'edges': set(modeled_edges['linkid'].tolist()),
#                 'geometry':MultiLineString(modeled_edges['geometry'].tolist()),#modeled_edges.dissolve()['geometry'].item(),
#                 'length':MultiLineString(modeled_edges['geometry'].tolist()).length
#                 }

#         #remove virtual links
#         pseudo_G = modeling_turns.remove_virtual_edges(pseudo_G,virtual_edges)
    
#     print('calculating objective function')

#     #turn shortest paths dict to dataframe
#     shortest_paths = pd.DataFrame.from_dict(shortest_paths,orient='index')
#     shortest_paths.reset_index(inplace=True)
#     shortest_paths.columns = ['start','end','linkids','geometry','length']
#     #shortest_paths[['start','end']] = shortest_paths['index'].apply(lambda x: pd.Series(x))
#     #shortest_paths.drop(columns=['index'],inplace=True)

#     #add modeled paths to matched_traces dataframe
#     merged = matched_traces.merge(shortest_paths,on=['start','end'],suffixes=(None,'_modeled'))

#     if exact:
#         sum_all = merged['length'].sum() * 5280
#         all_overlap = 0

#         for idx, row in merged.iterrows():
#             #find shared edges
#             chosen_and_shortest = row['linkids_modeled'] & row['linkids']
#             #get the lengths of those links
#             overlap_length = links.set_index('linkid').loc[list(chosen_and_shortest)]['length_ft'].sum()
#             #overlap_length = np.sum([link_lengths.get(link_tup,'error') for link_tup in chosen_and_shortest])
#             all_overlap += overlap_length

#         #calculate objective function value
#         val = all_overlap / sum_all
#         print('Exact overlap percent is:',np.round(val*100,1),'%')
    
#     #calculate approximate overlap (new approach)
#     else:
#         #buffer and dissolve generated route and matched route
#         buffer_ft = 500

#         merged.set_geometry('geometry',inplace=True)
#         merged['buffered_geometry'] = merged.buffer(buffer_ft)
#         merged.set_geometry('buffered_geometry',inplace=True)
#         merged['area'] = merged.area

#         merged.set_geometry('geometry_modeled',inplace=True)
#         merged['buffered_geometry_modeled'] = merged.buffer(buffer_ft)
#         merged.set_geometry('buffered_geometry_modeled',inplace=True)
#         merged['area_modeled'] = merged.area

#         #for each row find intersection between buffered features
#         merged['intersection'] = merged.apply(lambda row: row['buffered_geometry'].intersection(row['buffered_geometry_modeled']), axis=1)

#         # merged['intersection'] = merged.apply(
#         #     lambda row: shapely.intersection(row['buffered_geometry'],row['buffered_geometry_modeled']))
#         merged.set_geometry('intersection',inplace=True)
#         merged['intersection_area'] = merged.area

#         #find the overlap with the total area (not including intersections)
#         #if the modeled/chosen links are different, then overlap decreases
#         #punishes cirquitious modeled routes that utilize every link in the chosen one but include extraneous ones
#         merged['overlap'] = merged['intersection_area'] / (merged['area_modeled'] + merged['area'] - merged['intersection_area'])

#         #find average overlap (using median to reduce impact of outliers?)
#         val = merged['overlap'].median()
#         print('Median overlap percent is:',np.round(val*100,1),'%')
    
#     if follow_up:
#         return merged

#     return -val#, merged

# def follow_up(betas,links,pseudo_links,pseudo_G,matched_traces,exact=False):

#     #
#     modeled_trips = {}
    
#     #use initial/updated betas to calculate link costs
#     links = link_impedance_function(betas, links)
#     cost_dict = dict(zip(links['linkid'],links['link_cost']))
    
#     #TODO is this the best way to accomplish this?
#     #get_geo = links.loc[links.groupby(['source','target'])['link_cost'].idxmin(),['source','target','geometry']]

#     #add link costs to pseudo_links
#     pseudo_links['source_link_cost'] = pseudo_links['source_linkid'].map(cost_dict)
#     pseudo_links['target_link_cost'] = pseudo_links['target_linkid'].map(cost_dict)

#     #use initial/updated betas to calculate turn costs
#     pseudo_links = turn_impedance_function(betas, pseudo_links)

#     #assign na turns a cost of 0
#     pseudo_links.loc[pseudo_links['turn_cost'].isna(),'turn_cost'] = 0

#     #add links and multiply by turn cost
#     pseudo_links['total_cost'] = pseudo_links['source_link_cost'] + pseudo_links['target_link_cost'] + pseudo_links['turn_cost']

#     #only keep link with the lowest cost
#     costs = pseudo_links.groupby(['source','target'])['total_cost'].min()

#     #get linkids used
#     source_cols = ['source','source_linkid','source_reverse_link']
#     target_cols = ['target','target_linkid','target_reverse_link']
#     min_links = pseudo_links.loc[pseudo_links.groupby(['source','target'])['total_cost'].idxmin()]
#     source_links = min_links[source_cols]
#     target_links = min_links[target_cols]
#     source_links.columns = ['A_B','linkid','reverse_link']
#     target_links.columns = ['A_B','linkid','reverse_link']
#     linkids = pd.concat([source_links,target_links],ignore_index=True).drop_duplicates().set_index('A_B')

#     #update edge weights
#     nx.set_edge_attributes(pseudo_G,values=costs,name='weight')
    
#     #do shortest path routing
#     shortest_paths = {}
#     print(f'Shortest path routing with coefficients: {betas}')    
#     for source, targets in matched_traces.groupby('start')['end'].unique().items():

#         #add virtual links to pseudo_G
#         pseudo_G, virtual_edges = modeling_turns.add_virtual_links(pseudo_links,pseudo_G,source,targets)
        
#         #perform shortest path routing for all target nodes from source node
#         #(from one to all until target node has been visited)
#         for target in targets:  
#             #cant be numpy int64 or throws an error
#             target = int(target)
            
#             try:
#                 #TODO every result is only a start node, middle node, then end node
#                 length, node_list = nx.single_source_dijkstra(pseudo_G,source,target,weight='weight')
#             except:
#                 print(source,target)
#                 length, node_list = nx.single_source_dijkstra(pseudo_G,source,target,weight='weight')

#             #get edge list
#             edge_list = node_list[1:-1]

#             #get geometry from edges
#             modeled_edges = links.merge(linkids.loc[edge_list],on=['linkid','reverse_link'],how='inner')
#             modeled_edges = gpd.GeoDataFrame(modeled_edges,geometry='geometry')
            
#             #
#             shortest_paths[(source,target)] = {
#                 'edges': set(modeled_edges['linkid'].tolist()),
#                 'geometry':MultiLineString(modeled_edges['geometry'].tolist()),
#                 'length':modeled_edges.length.sum()
#                 }

#         #remove virtual links
#         pseudo_G = modeling_turns.remove_virtual_edges(pseudo_G,virtual_edges)
    
#     #turn shortest paths dict to dataframe
#     shortest_paths = pd.DataFrame.from_dict(shortest_paths,orient='index')
#     shortest_paths.reset_index(inplace=True)
#     shortest_paths.columns = ['start','end','linkids','geometry','length']
#     #shortest_paths[['start','end']] = shortest_paths['index'].apply(lambda x: pd.Series(x))
#     #shortest_paths.drop(columns=['index'],inplace=True)

#     #add modeled paths to matched_traces dataframe
#     merged = matched_traces.merge(shortest_paths,on=['start','end'],suffixes=(None,'_modeled'))

#     if exact:
#         sum_all = merged['length'].sum() * 5280
#         all_overlap = 0

#         for idx, row in merged.iterrows():
#             #find shared edges
#             chosen_and_shortest = row['linkids_modeled'] & row['linkids']
#             #get the lengths of those links
#             overlap_length = links.set_index('linkid').loc[list(chosen_and_shortest)]['length_ft'].sum()
#             #overlap_length = np.sum([link_lengths.get(link_tup,'error') for link_tup in chosen_and_shortest])
#             all_overlap += overlap_length

#         #calculate objective function value
#         val = all_overlap / sum_all
#         print('Exact overlap percent is:',np.round(val*100,1),'%')
    
#     #calculate approximate overlap (new approach)
#     else:
#         #buffer and dissolve generated route and matched route
#         buffer_ft = 500

#         merged.set_geometry('geometry',inplace=True)
#         merged['buffered_geometry'] = merged.buffer(buffer_ft)
#         merged.set_geometry('buffered_geometry',inplace=True)
#         merged['area'] = merged.area

#         merged.set_geometry('geometry_modeled',inplace=True)
#         merged['buffered_geometry_modeled'] = merged.buffer(buffer_ft)
#         merged.set_geometry('buffered_geometry_modeled',inplace=True)
#         merged['area_modeled'] = merged.area

#         #for each row find intersection between buffered features
#         merged['intersection'] = merged.apply(lambda row: row['buffered_geometry'].intersection(row['buffered_geometry_modeled']), axis=1)

#         # merged['intersection'] = merged.apply(
#         #     lambda row: shapely.intersection(row['buffered_geometry'],row['buffered_geometry_modeled']))
#         merged.set_geometry('intersection',inplace=True)
#         merged['intersection_area'] = merged.area

#         #find the overlap with the total area (not including intersections)
#         #if the modeled/chosen links are different, then overlap decreases
#         #punishes cirquitious modeled routes that utilize every link in the chosen one but include extraneous ones
#         merged['overlap'] = merged['intersection_area'] / (merged['area_modeled'] + merged['area'] - merged['intersection_area'])

#         #find average overlap (using median to reduce impact of outliers?)
#         val = merged['overlap'].median()
#         print('Median overlap percent is:',np.round(val*100,1),'%')
    
#     return -val#, merged         



     #TODO add in the legend with trip info and then we're golden

        # #calculate approximate overlap
        # all_overlap = 0
        # for idx, row in trips_df.iterrows():

        #     #buffer and dissolve generated route and matched route
        #     modeled_edges = shortest_paths[row['od']]['edge_list']
        #     chosen_edges = matched_traces[row['tripid']]['matched_trip']
            
        #     #grab links
        #     links.index = list(zip(links['A'],links['B']))
        #     modeled_edges = links.loc[modeled_edges]

        #     #caluclate absolute difference
        #     difference_ft = (modeled_edges.length.sum() - chosen_edges.legnth.sum()).abs()

        #     #buffer edges and dissolve
        #     buffer_ft = 500
        #     modeled_edges_dissolved = modeled_edges.buffer(buffer_ft).dissolve()
        #     chosen_edges_dissovled = chosen_edges.buffer(buffer_ft).dissolve()

        #     #intersect
        #     intersect = gpd.overlay(modeled_edges_dissolved,chosen_edges_dissovled,how='intersection')
        #     overlap = intersect.area / chosen_edges.area
            
        #     #exponent for difference in lengths between generated route and matched route
        #     #as absolute difference in length increases, overlap gets smaller
        #     all_overlap += overlap ** difference_ft

        # duration = timedelta(seconds=time.perf_counter()-start_time)
        # durations.append(duration)
        # start_time = time.perf_counter()



    # print('Overlap =', val)




# #import segment to use
# segment_filepaths = list((fp/'segments').glob('*'))
# results = {}

# def replace_missing_node(row,trips_df,pseudo_G,nodes):                
#     source = row['od'][0]
#     target = row['od'][1]
    
#     #source column
#     if ~pseudo_G.has_node(source):
#         start_coord = row[['start_lat','start_lon']]
#         start_coord['geometry'] = Point(start_coord.iloc[0,1],start_coord.iloc[0,0])
#         start_coord = gpd.GeoDataFrame(start_coord,geometry='geometry',crs='epsg:4326')
#         start_coord.to_crs(nodes.crs)
#         source = gpd.sjoin_nearest(start_start_coord, nodes)['N'].item()
        
#     if ~pseudo_G.has_node(target):
#         end_coord = trips_df.loc[trips_df['od']==(source,target),['end_lat','end_lon']].iloc[0]
#         end_coord['geometry'] = Point(end_coord.iloc[0,1],end_coord.iloc[0,0])
#         end_coord = gpd.GeoDataFrame(end_coord,geometry='geometry',crs='epsg:4326')
#         end_coord.to_crs(nodes.crs)
#         target = gpd.sjoin_nearest(end_coord, nodes)['N'].item()
        
#     return (source,target)

# for segment_filepath in segment_filepaths:
#     trips_df = pd.read_csv(segment_filepath)
#     #turn to tuple
#     trips_df['od'] = trips_df['od'].apply(lambda row: ast.literal_eval(row))
#     trips_df['od'] = trips_df.apply(lambda row: replace_missing_node(row, trips_df, pseudo_G, nodes))
    
#     #inputs
#     sum_all = trips_df['chosen_length'].sum() * 5280
#     links['tup'] = list(zip(links['A'],links['B']))
#     link_lengths = dict(zip(links['tup'],links['length_ft']))
#     durations = []
#     ods = list(set(trips_df['od'].tolist()))
    
#     start = time.time()
#     bounds = [[-5,5],[-5,5],[-5,5]]
#     x = minimize(loss_function, bounds, args=(links,G,ods,trips_df,matched_traces,durations), method='pso')
#     end = time.time()
#     print(f'Took {(end-start)/60/60} hours')
#     results[segment_filepath] = (x.x,x.fun)

# #%%
# timestr = time.strftime("%Y-%m-%d-%H-%M")
# with (fp/f"calibration_results_{timestr}.pkl").open('wb') as fh:
#     pickle.dump(results,fh)


# new_results = {key.parts[-1].split('.csv')[0]:items for key, items in results.items()}
# new_results = pd.DataFrame.from_dict(new_results,orient='index',columns=['coefs','overlap'])
# new_results[['not_beltline','not_infra','2lanes','30mph']] = new_results['coefs'].apply(pd.Series)