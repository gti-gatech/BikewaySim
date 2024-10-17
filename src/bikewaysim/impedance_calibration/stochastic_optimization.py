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

import random
import similaritymeasures

from bikewaysim.paths import config
from bikewaysim.network import modeling_turns
from bikewaysim.routing import rustworkx_routing_funcs

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

TODO:
Start seperating out functions into different .py files as it makes sense

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
                          links:pd.DataFrame,turns:pd.DataFrame,turn_G:nx.digraph,
                          loss_function,
                          loss_function_kwargs,
                          print_results=True,
                          track_calibration_results=True,
                          batching=False
                          ):

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
        if print_results:
            print('negative edge weights, skipping guess')
        if track_calibration_results == True:
            past_vals.append(list(betas)+[np.nan])
        return np.nan

    if batching == True:
        batch_ods = random.sample(ods,k=5)
        batch_ods = sorted(batch_ods,key=lambda x: x[-1])[::-1]
        batch_match_results = {tripid:item for tripid, item in match_results.items() if (item['origin_node'],item['destination_node'],item['trip_start_time']) in set(batch_ods)}
    else:
        batch_ods = ods
        batch_match_results = match_results

    starts = [x[0] for x in batch_ods]
    ends = [x[1] for x in batch_ods]
    years = sorted(list(set([x[2] for x in batch_ods])))[::-1]

    # create the networks by year
    year_networks = {}
    for year in years:
        # create a copy of the network to modify
        turn_G_copy = turn_G.copy()
        
        # if infra is on street (i.e., the link is still traversable but the impedance doesn't apply)
        links.loc[links['year'] > year,set_to_zero] = 0 
        # if it's off-street then assign it a very high cost
        links.loc[(links['year'] > year) & (links.loc[:,set_to_inf]==1).any(axis=1),'link_cost_override'] = True

        # run network update
        rustworkx_routing_funcs.impedance_update(betas,betas_tup,
                link_impedance_function,
                base_impedance_col,
                turn_impedance_function,
                links,turns,turn_G_copy)
        
        # re-add virtual links
        rustworkx_routing_funcs.add_virtual_edges(starts,ends,links,turns,turn_G_copy)
        
        # add the network to the dict
        year_networks[year] = turn_G_copy

    added_nodes = rustworkx_routing_funcs.add_virtual_edges(starts,ends,links,turns,turn_G)
    
    path_lengths, shortest_paths = rustworkx_routing_funcs.rx_shortest_paths_year(batch_ods,turn_G,year_networks)
    
    results_dict = {(od[0],od[1]):{'length':path_length,'edge_list':shortest_path} for path_length, shortest_path, od in zip(path_lengths,shortest_paths,batch_ods)}

    rustworkx_routing_funcs.remove_virtual_links(added_nodes,turn_G)

    #TODO continue here

    # #find least impedance path
    # trips_years = set()
    # results_dict = {}
    # for start_node, end_node, year in tqdm(batch_ods):
    #     if year not in trips_years:
    #         trips_years.add(year) #  it to the years already looked at
    #         if (links['year'] > year).any(): 
    #             rustworkx_routing_funcs.remove_virtual_links(added_nodes,turn_G)
                
    #             # if infra is on street (i.e., the link is still traversable but the impedance doesn't apply)
    #             links.loc[links['year'] > year,set_to_zero] = 0 
    #             # if it's off-street then assign it a very high cost
    #             links.loc[(links['year'] > year) & (links.loc[:,set_to_inf]==1).any(axis=1),'link_cost_override'] = True
    #             # set these years to nan
    #             links.loc[(links['year'] > year),'year'] = np.nan

    #             # re-run network update
    #             rustworkx_routing_funcs.impedance_update(betas,betas_tup,
    #                     link_impedance_function,
    #                     base_impedance_col,
    #                     turn_impedance_function,
    #                     links,turns,turn_G)
                
    #             # re-add virtual links
    #             added_nodes = rustworkx_routing_funcs.add_virtual_edges(starts,ends,links,turns,turn_G)
            
    #     path_length, shortest_path = rustworkx_routing_funcs.rx_shortest_paths([(start_node,end_node)],turn_G)
    #     results_dict[(start_node,end_node)] = {'length':path_length[0],'edge_list':shortest_path[0]}
        # results_dict[(start_node,end_node)] = impedance_path(turns,turn_G,links,start_node,end_node) old version

    # if no consideration for date
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
    if track_calibration_results == True:
        past_vals.append(list(betas)+[val_to_minimize])

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
    # TODO round these values because they don't need this many decimal points (usually in travel time)
    turns['total_cost'] = (turns['target_link_cost'] + turns['turn_cost'])

    if turns['total_cost'].isna().any():
        raise Exception("There are nan edge costs, exiting")

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
    turns['total_cost'] =  turns['target_link_cost']

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

def link_impedance_function(betas:np.array,betas_tup:tuple,links:pd.DataFrame,base_impedance_col:str):
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

# Post Calibration Functions

########################################################################################

def post_calibration_routing(links,turns,turn_G,full_ods,base_impedance_col,set_to_zero,set_to_inf,calibration_result):
    
    '''
    Used after a calibration run to return the least impedance paths using the calibrated coefficients
    '''

    # base_impedance_col = "travel_time_min" # set the base impedance (default is travel time)
    betas = [x['beta'] for x in calibration_result['betas_tup']] # get betas

    # create a copy of links so we don't mutate it
    links = links.copy()
    
    # reset the link costs
    rustworkx_routing_funcs.back_to_base_impedance(base_impedance_col,links,turns,turn_G)

    # do initial impedance update
    # if infra is off street (i.e., the link should no longer be traversable)
    links['link_cost_override'] = False
    rustworkx_routing_funcs.impedance_update(betas,calibration_result['betas_tup'],
                        link_impedance_function,
                        base_impedance_col,
                        turn_impedance_function,
                        links,turns,turn_G)

    # trips_years = set()
    # results_dict = {}
    # for start_node, end_node, year in full_ods:
    #     if year not in trips_years:
    #         # print('Remaking network for',year)
    #         trips_years.add(year) # add it to the years already looked at
    #         if (links['year'] > year).any(): # only then should we run this
    #             # print('Re-making network to year',year)
                
    #             # if infra is on street (i.e., the link is still traversable but the impedance doesn't apply)
    #             links.loc[links['year']>year,set_to_zero] = 0 
    #             links.loc[(links['year']>year) & (links.loc[:,set_to_inf]==1).any(axis=1),'link_cost_override'] = True
                
    #             # re-run network update
    #             rustworkx_routing_funcs.impedance_update(betas,calibration_result['betas_tup'],
    #                     link_impedance_function,
    #                     base_impedance_col,
    #                     turn_impedance_function,
    #                     links,turns,turn_G)
    #     path_length, shortest_path = rustworkx_routing_funcs.rx_shortest_paths([(start_node,end_node)],turn_G)
    #     results_dict[(start_node,end_node)] = {'length':path_length[0],'edge_list':shortest_path[0]}
    #     # results_dict[(start_node,end_node)] = impedance_path(turns,turn_G,links,start_node,end_node)

    # create the networks by year
    
    starts = [x[0] for x in full_ods]
    ends = [x[1] for x in full_ods]
    years = sorted(list(set([x[2] for x in full_ods])))[::-1]

    year_networks = {}
    for year in years:
        # create a copy of the network to modify
        turn_G_copy = turn_G.copy()
        
        # if infra is on street (i.e., the link is still traversable but the impedance doesn't apply)
        links.loc[links['year'] > year,set_to_zero] = 0 
        # if it's off-street then assign it a very high cost
        links.loc[(links['year'] > year) & (links.loc[:,set_to_inf]==1).any(axis=1),'link_cost_override'] = True

        # run network update
        rustworkx_routing_funcs.impedance_update(betas,calibration_result['betas_tup'],
                link_impedance_function,
                base_impedance_col,
                turn_impedance_function,
                links,turns,turn_G_copy)
        
        # re-add virtual links
        rustworkx_routing_funcs.add_virtual_edges(starts,ends,links,turns,turn_G_copy)
        
        # add the network to the dict
        year_networks[year] = turn_G_copy

    added_nodes = rustworkx_routing_funcs.add_virtual_edges(starts,ends,links,turns,turn_G)
    
    path_lengths, shortest_paths = rustworkx_routing_funcs.rx_shortest_paths_year(full_ods,turn_G,year_networks)
    
    results_dict = {(od[0],od[1]):{'length':path_length,'edge_list':shortest_path} for path_length, shortest_path, od in zip(path_lengths,shortest_paths,full_ods)}

    rustworkx_routing_funcs.remove_virtual_links(added_nodes,turn_G)

    return results_dict

def shortest_loss():
    '''
    Calculate loss values and metrics for the shortest path
    '''

    #TODO put users in this too

    # Retreive shortest and chosen stats first (already calculated the shortest paths)
    links, turns, length_dict, geo_dict, turn_G = import_calibration_network(config)
    with (config['calibration_fp']/'ready_for_calibration.pkl').open('rb') as fh:
        full_set = pickle.load(fh)
    
    for key, item in full_set.items():
        # extract chosen and shortest routes
        chosen = item['matched_edges'].values
        shortest = item['shortest_edges'].values

        #compute the loss values (store the intermediates too so total vs mean methods can be compared)
        shortest_jaccard_exact_intersection, shortest_jaccard_exact_union =  jaccard_exact(chosen,shortest,length_dict)
        shortest_jaccard_exact = shortest_jaccard_exact_intersection / shortest_jaccard_exact_union
        shortest_jaccard_buffer_intersection, shortest_jaccard_buffer_union =  jaccard_buffer(chosen,shortest,geo_dict)
        shortest_jaccard_buffer = shortest_jaccard_buffer_intersection / shortest_jaccard_buffer_union

        # add to full set in additon to what's already there
        full_set[key].update({
            'chosen_length': round(np.array([length_dict.get(tripid[0],False) for tripid in chosen]).sum()/5280,2),
            'shortest_length': round(np.array([length_dict.get(tripid[0],False) for tripid in shortest]).sum()/5280,2),
            'chosen_detour': round(detour_factor(chosen,shortest,length_dict),2),
            'shortest_jaccard_exact': round(shortest_jaccard_exact,2),
            'shortest_jaccard_exact_intersection': round(shortest_jaccard_exact_intersection,2),
            'shortest_jaccard_exact_union': round(shortest_jaccard_exact_union,2),
            'shortest_jaccard_buffer': round(shortest_jaccard_buffer,2),
            'shortest_jaccard_buffer_intersection': round(shortest_jaccard_buffer_intersection,2),
            'shortest_jaccard_buffer_union': round(shortest_jaccard_buffer_union,2),
        })

    # export new version
    with (config['calibration_fp']/'ready_for_calibration_stats.pkl').open('wb') as fh:
        pickle.dump(full_set,fh)

def file_check(user=False):
    #TODO add a way to exlcude calibration results
    if user:
        calibration_results_dir = config['calibration_fp']/"user_calibration_results"
        post_calibration_routing_dir = config['calibration_fp']/"user_post_calibration_routing"
        post_calibration_loss_dir = config['calibration_fp']/"user_post_calibration_loss"
    else:
        calibration_results_dir = config['calibration_fp']/"calibration_results"
        post_calibration_routing_dir = config['calibration_fp']/"post_calibration_routing"
        post_calibration_loss_dir = config['calibration_fp']/"post_calibration_loss"

    if post_calibration_loss_dir.exists() == False:
        post_calibration_loss_dir.mkdir()
    
    # get the intersection of both
    calibration_results = [x.stem for x in calibration_results_dir.glob('*.pkl')]
    post_calibration_routing = [x.stem for x in post_calibration_routing_dir.glob('*.pkl')]
    calibrations = list(set.intersection(set(calibration_results),set(post_calibration_routing)))
    print('Available calibrations:',list(calibrations))

    return calibration_results_dir, post_calibration_routing_dir, post_calibration_loss_dir, calibrations

import re
def get_calibration_name_parameters(calibration_name,user=False):

    output = {}

    if user:
        output['userid'] = calibration_name.split('_')[0]
        calibration_name = calibration_name.split('_',maxsplit=1)[1]

    # get run number
    pattern = r'\((\d+)\)'
    run_number = re.findall(pattern,calibration_name)

    if len(run_number) == 0:
        output['run_number'] = 0
        output['calibration'] = calibration_name.strip()
    else:
        output['run_number'] = run_number[0]
        output['calibration'] = calibration_name.split('(')[0].strip()
    
    return output

def post_calibration_loss(user=False):
    '''
    Calcualte loss values and metrics for the modeled routes (for all calibration runs)
    '''
    _, _, length_dict, geo_dict, _ = import_calibration_network(config)

    calibration_results_dir, post_calibration_routing_dir, post_calibration_loss_dir, calibrations = file_check(user)

    # import all trips
    with (config['calibration_fp']/'ready_for_calibration.pkl').open('rb') as fh:
        full_set_all = pickle.load(fh)

    for calibration in tqdm(calibrations):
        # load the calibration result (has the estimated betas)
        with (calibration_results_dir / f"{calibration}.pkl").open('rb') as fh:
            calibration_result = pickle.load(fh)

        # subset to the trips used for calibration (in case there was some sort of subsetting)
        full_set = {tripid:item for tripid, item in full_set_all.items() if tripid in calibration_result['trips_calibrated']}
        
        # load the post calibration results (includes the impedance path taken for each trip)
        with (post_calibration_routing_dir / f"{calibration}.pkl").open('rb') as fh:
            results_dict = pickle.load(fh)
        
        modeled_results_dict = {}
        # we could try storing these
        for tripid, item in full_set.items(): # make sure you import this
            chosen = item['matched_edges'].values
            shortest = item['shortest_edges'].values
            
            # NOTE one limitation of this is that if the origin node and destination node change at any point then
            # than this step will break
            od = (item['origin_node'],item['destination_node'])
            modeled = results_dict.get(od)
            if modeled is None:
                print(f"No shortest path found for {tripid} on calibration {calibration}")
                continue
            modeled = modeled['edge_list']
        
            #compute the loss values (store the intermediates too so total vs mean methods can be compared)
            modeled_jaccard_exact_intersection, modeled_jaccard_exact_union =  jaccard_exact(chosen,modeled,length_dict)
            modeled_jaccard_exact = modeled_jaccard_exact_intersection / modeled_jaccard_exact_union
            modeled_jaccard_buffer_intersection, modeled_jaccard_buffer_union =  jaccard_buffer(chosen,modeled,geo_dict)
            modeled_jaccard_buffer = modeled_jaccard_buffer_intersection / modeled_jaccard_buffer_union

            modeled_results_dict[tripid] = {
                'modeled_edges': pd.DataFrame(modeled,columns=['linkid','reverse_link']),
                'modeled_length': round(np.array([length_dict.get(tripid[0],0) for tripid in modeled]).sum()/5280,1),
                'modeled_detour': round(detour_factor(modeled,shortest,length_dict),2),
                'modeled_jaccard_exact': round(modeled_jaccard_exact,2),
                'modeled_jaccard_exact_intersection': round(modeled_jaccard_exact_intersection,2),
                'modeled_jaccard_exact_union': round(modeled_jaccard_exact_union,2),
                'modeled_jaccard_buffer': round(modeled_jaccard_buffer,2),
                'modeled_jaccard_buffer_intersection': round(modeled_jaccard_buffer_intersection,2),
                'modeled_jaccard_buffer_union': round(modeled_jaccard_buffer_union,2),
            }
        
        with (post_calibration_loss_dir / f"{calibration}.pkl").open('wb') as fh:
            pickle.dump(modeled_results_dict,fh)

def post_calibration_betas(user=False):
    '''
    Returns dataframe of all the estimated coefficients in the columns where each row is a model
    '''

    calibration_results_dir, post_calibration_routing_dir, post_calibration_loss_dir, calibrations = file_check(user)

    beta_vals = []
    for calibration in calibrations:
        with (calibration_results_dir / f"{calibration}.pkl").open('rb') as fh:
            calibration_result = pickle.load(fh)
        calibration_params = get_calibration_name_parameters(calibration,user)
        beta_vals.append({**calibration_params,**{x['col']:round(x['beta'],3) for x in calibration_result['betas_tup']}})
    beta_vals = pd.DataFrame().from_records(beta_vals)

    return beta_vals

#TDOO for these add the shortest result too
from collections import defaultdict
def post_calibration_disaggregate(user):
    '''
    Returns a dataframe where the rows are trips and the columns are length, detour, exact, and buffer
    Appends the shortest results to the left
    '''
    
    calibration_results_dir, post_calibration_routing_dir, post_calibration_loss_dir, calibrations = file_check(user)
    
    if user==False:
        all_loss_stats = defaultdict(dict)
        for calibration in tqdm(calibrations):
            with (post_calibration_loss_dir / f"{calibration}.pkl").open('rb') as fh:
                loss_dict = pickle.load(fh)
            output = get_calibration_name_parameters(calibration,user)
            #remove the edges
            for tripid, item in loss_dict.items():
                #remove certain fields
                for key in list(item.keys()):
                    if ('_intersection' in key) | ('_union' in key) | (key=='modeled_edges'):
                        item.pop(key)
                renamed = {(output['calibration'],output['run_number'],key.removeprefix('modeled_')):sub_item for key, sub_item in item.items()}
                all_loss_stats[tripid].update(renamed)
        all_loss_stats = pd.DataFrame.from_dict(all_loss_stats,orient='index')
        all_loss_stats.index.name = 'tripid'
        all_loss_stats.columns = pd.MultiIndex.from_tuples(all_loss_stats.columns, names=["calibration","run","loss"])
    else:
        all_loss_stats = defaultdict(dict)
        for calibration in tqdm(calibrations):
            with (post_calibration_loss_dir / f"{calibration}.pkl").open('rb') as fh:
                loss_dict = pickle.load(fh)
            output = get_calibration_name_parameters(calibration,user)
            #remove the edges
            for tripid, item in loss_dict.items():
                #remove certain fields
                for key in list(item.keys()):
                    if ('_intersection' in key) | ('_union' in key) | (key=='modeled_edges'):
                        item.pop(key)
                renamed = {(output['userid'],output['calibration'],output['run_number'],key.removeprefix('modeled_')):sub_item for key, sub_item in item.items()}
                all_loss_stats[tripid].update(renamed)
        all_loss_stats = pd.DataFrame.from_dict(all_loss_stats,orient='index')
        all_loss_stats.index.name = 'tripid'
        all_loss_stats.columns = pd.MultiIndex.from_tuples(all_loss_stats.columns, names=["subsetid","calibration","run","loss"])
    return all_loss_stats

def shortest_aggregated(user=False):
    '''
    Gets aggregated shortest path stats so they can be appened to the post calibration results
    '''
    # has all the shortest stats
    with (config['calibration_fp']/'ready_for_calibration_stats.pkl').open('rb') as fh:
        shortest = pickle.load(fh)
    # has all the user and trip pairs
    with (config['calibration_fp']/'ready_for_calibration_users.pkl').open('rb') as fh:
        ready_for_calibration_users = pickle.load(fh)

    aggregated_loss =[]
    if user:
        for userid, tripids in ready_for_calibration_users:
            user_shortest = {tripid:item for tripid, item in shortest.items() if tripid in tripids}

            jaccard_exact_mean = np.array([item['shortest_jaccard_exact'] for tripid, item in user_shortest.items()]).mean()
            jaccard_exact_total = np.array([(item['shortest_jaccard_exact_intersection'],item['shortest_jaccard_exact_union']) for tripid, item in user_shortest.items()])
            jaccard_exact_total = jaccard_exact_total[:,0].sum() / jaccard_exact_total[:,1].sum()

            jaccard_buffer_mean = np.array([item['shortest_jaccard_buffer'] for tripid, item in user_shortest.items()]).mean()
            jaccard_buffer_total = np.array([(item['shortest_jaccard_buffer_intersection'],item['shortest_jaccard_buffer_union']) for tripid, item in user_shortest.items()])
            jaccard_buffer_total = jaccard_buffer_total[:,0].sum() / jaccard_buffer_total[:,1].sum()
    
            aggregated_loss.append({
                'userid':userid,
                'jaccard_exact_mean': round(jaccard_exact_mean,2),
                'jaccard_exact_total': round(jaccard_exact_total,2),
                'jaccard_buffer_mean': round(jaccard_buffer_mean,2),
                'jaccard_buffer_total': round(jaccard_buffer_total,2)
            })
    else:
        jaccard_exact_mean = np.array([item['shortest_jaccard_exact'] for tripid, item in shortest.items()]).mean()
        jaccard_exact_total = np.array([(item['shortest_jaccard_exact_intersection'],item['shortest_jaccard_exact_union']) for tripid, item in shortest.items()])
        jaccard_exact_total = jaccard_exact_total[:,0].sum() / jaccard_exact_total[:,1].sum()

        jaccard_buffer_mean = np.array([item['shortest_jaccard_buffer'] for tripid, item in shortest.items()]).mean()
        jaccard_buffer_total = np.array([(item['shortest_jaccard_buffer_intersection'],item['shortest_jaccard_buffer_union']) for tripid, item in shortest.items()])
        jaccard_buffer_total = jaccard_buffer_total[:,0].sum() / jaccard_buffer_total[:,1].sum()
        
        aggregated_loss.append({
            'jaccard_exact_mean': round(jaccard_exact_mean,2),
            'jaccard_exact_total': round(jaccard_exact_total,2),
            'jaccard_buffer_mean': round(jaccard_buffer_mean,2),
            'jaccard_buffer_total': round(jaccard_buffer_total,2)
            })
    
    aggregated_loss = pd.DataFrame.from_records(aggregated_loss)
    return aggregated_loss

def post_calibration_aggregated(user=False):
    '''
    Calculates aggregated stats for all of the calibration results.

    Every row is a calibration result. If user based, then every row is a calibration result for a user.
    
    '''
    calibration_results_dir, post_calibration_routing_dir, post_calibration_loss_dir, calibrations = file_check(user)

    # disaggregated
    aggregated_loss = []
    for calibration in calibrations:
        with (post_calibration_loss_dir / f"{calibration}.pkl").open('rb') as fh:
            loss_dict = pickle.load(fh)

        jaccard_exact_mean = np.array([item['modeled_jaccard_exact'] for tripid, item in loss_dict.items()]).mean()
        jaccard_exact_total = np.array([(item['modeled_jaccard_exact_intersection'],item['modeled_jaccard_exact_union']) for tripid, item in loss_dict.items()])
        jaccard_exact_total = jaccard_exact_total[:,0].sum() / jaccard_exact_total[:,1].sum()

        jaccard_buffer_mean = np.array([item['modeled_jaccard_buffer'] for tripid, item in loss_dict.items()]).mean()
        jaccard_buffer_total = np.array([(item['modeled_jaccard_buffer_intersection'],item['modeled_jaccard_buffer_union']) for tripid, item in loss_dict.items()])
        jaccard_buffer_total = jaccard_buffer_total[:,0].sum() / jaccard_buffer_total[:,1].sum()

        calibration_params = get_calibration_name_parameters(calibration,user)

        aggregated_loss.append({
            **calibration_params, # contains the userid, calibration name and run number
            'jaccard_exact_mean': round(jaccard_exact_mean,2),
            'jaccard_exact_total': round(jaccard_exact_total,2),
            'jaccard_buffer_mean': round(jaccard_buffer_mean,2),
            'jaccard_buffer_total': round(jaccard_buffer_total,2)
        })
    aggregated_loss = pd.DataFrame.from_records(aggregated_loss)
    return aggregated_loss

########################################################################################

# Scripting Calibration Runs

########################################################################################

def full_impedance_calibration(
        calibration_name,
        betas_tup,
        objective_function=jaccard_buffer_mean,
        set_to_zero=[],
        set_to_inf=[],
        batching=False,
        stochastic_optimization_settings={'method':'pso','options':{'maxiter':100,'popsize':25}},
        print_results = False, # default is false
        base_impedance_col='travel_time_min',
        user = None, # tuple with (userid,list_of_trips) or use to subset data (OPTIONAL)
        force_save = False
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

    # # DEBUGGING
    # subset = list(full_set.keys())[0:100]
    # full_set = {tripid:item for tripid, item in full_set.items() if tripid in subset}

    # subset to a user's trips if user argument is provided
    if user is not None:
        full_set = {tripid:item for tripid, item in full_set.items() if tripid in user[1]}

    # TODO clean up how this is done
    # NOTE need a tuple for minimize but this could be replaced with a function that converts a dict to tuple
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
        {'length_dict':length_dict,'geo_dict':geo_dict},#,'trace_dict':traces}, # keyword arguments for loss function
        print_results, #whether to print the results of each iteration (useful when testing the calibration on its own)
        True, #whether to store calibration results
        batching # whether to batch results to help speed up computation time, if yes input the number to batch with
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
    
    if print_results:
        print('Took',str(pd.Timedelta(seconds=end-start).round('s')),'hours')
        print(f"{objective_function.__name__}: {x.fun}")
        print(x)

    # I should be saving the result regardless for posterity
    # if x.success | force_save:
    
    # assemble dictionary of the calibration results
    calibration_result = {
        'betas_tup': tuple({**item,'beta':x.x[idx].round(4)} for idx,item in enumerate(betas_tup)), # contains the betas
        'set_to_zero': set_to_zero,
        'set_to_inf': set_to_inf,
        'settings': stochastic_optimization_settings, # contains the optimization settings
        'objective_function': objective_function.__name__, # objective function used
        'results': x, # stochastic optimization outputs
        'trips_calibrated': set(full_set.keys()), # saves which trips were calibrated
        'past_vals': args[0], # all of the past values/guesses #BUG does not appear to work correcctly
        'runtime': datetime.timedelta(seconds= start - end),
        'time': datetime.datetime.now()
    }

    #NOTE consider keeping these in the same directory
    calibration_result_fp, post_calibration_routing_fp = handle_directories(user,calibration_name)

    with uniquify(calibration_result_fp).open('wb') as fh:
        pickle.dump(calibration_result,fh)
    
    #TODO have a try except for when force save returns negative link values?
    # post process the model results now so they're ready for analysis and visualization
    if x.success:
        modeled_ods = post_calibration_routing(links,turns,turn_G,match_results_to_ods_w_year(full_set),base_impedance_col,set_to_zero,set_to_inf,calibration_result)
        
        with uniquify(post_calibration_routing_fp).open('wb') as fh:
            pickle.dump(modeled_ods,fh)
        
    if x.success:
        return True
    else:
        return False

def handle_directories(user,calibration_name):
    # handle the directories
    if user is not None:
        calibration_result_fp = f"user_calibration_results/{user[0]}_{calibration_name}.pkl"
        post_calibration_routing_fp = f"user_post_calibration_routing/{user[0]}_{calibration_name}.pkl"
    else:
        calibration_result_fp = f"calibration_results/{calibration_name}.pkl"
        post_calibration_routing_fp = f"post_calibration_routing/{calibration_name}.pkl"
    
    calibration_result_fp = config['calibration_fp'] / calibration_result_fp
    post_calibration_routing_fp = config['calibration_fp'] / post_calibration_routing_fp

    if calibration_result_fp.parent.exists() == False:
        calibration_result_fp.parent.mkdir()
    if post_calibration_routing_fp.parent.exists() == False:
        post_calibration_routing_fp.parent.mkdir()
    
    return calibration_result_fp, post_calibration_routing_fp

# Helper function to unpack the dictionary and call example_function
def run_calibration(task):
    task_dict, run_num, NUM_RUNS = task
    task_name = f"{task_dict['calibration_name']} ({run_num+1}/{NUM_RUNS}) "
    success = full_impedance_calibration(**task_dict)
    return task_name, success

def uniquify(path):
    counter = 1
    original_stem = path.stem
    extension = path.suffix

    while path.exists():
        path = path.parent / (original_stem + f" ({str(counter)})" + extension)
        counter += 1

    return path

import folium
def create_custom_route(betas_tup,tripid,full_set,links,turns,turn_G,length_dict,geo_dict,base_impedance_col="travel_time_min",set_to_zero=['bike lane','cycletrack','multi use path'],set_to_inf = ['not_street']):
    
    one_set = {key:item for key,item in full_set.items() if key == tripid}

    start, end, year = match_results_to_ods_w_year(one_set)[0]

    # create a copy of the network to modify
    turn_G_copy = turn_G.copy()

    # if infra is on street (i.e., the link is still traversable but the impedance doesn't apply)
    links.loc[links['year'] > year,set_to_zero] = 0 
    # if it's off-street then assign it a very high cost
    links.loc[(links['year'] > year) & (links.loc[:,set_to_inf]==1).any(axis=1),'link_cost_override'] = True

    # run network update
    betas = [x['beta'] for x in betas_tup]
    rustworkx_routing_funcs.impedance_update(
        betas,betas_tup,
        link_impedance_function,
        base_impedance_col,
        turn_impedance_function,
        links,turns,turn_G_copy)
            
    # re-add virtual links
    added_nodes = rustworkx_routing_funcs.add_virtual_edges([start],[end],links,turns,turn_G_copy)

    # route
    path_lengths, shortest_paths = rustworkx_routing_funcs.rx_shortest_paths((start,end),turn_G_copy)
    rustworkx_routing_funcs.remove_virtual_links(added_nodes,turn_G_copy)

    modeled = [list(x) for x in shortest_paths[0]]
    chosen = one_set[tripid]['matched_edges'].values
    shortest = one_set[tripid]['shortest_edges'].values
    length = round(np.array([length_dict.get(tripid[0],0) for tripid in modeled]).sum()/5280,1)
    detour = round(detour_factor(modeled,shortest,length_dict),2)
    jaccard_exact_val = jaccard_exact(chosen,modeled,length_dict)
    jaccard_exact_val = round(jaccard_exact_val[0] / jaccard_exact_val[1],2)
    jaccard_buffer_val = jaccard_buffer(chosen,modeled,geo_dict)
    jaccard_buffer_val = round(jaccard_buffer_val[0] / jaccard_buffer_val[1],2)

    custom_route = gpd.GeoDataFrame(
        data={'tripid':[tripid],
            'length':length,
            'detour':detour,
            'jaccard_exact':jaccard_exact_val,
            'jaccard_buffer':jaccard_buffer_val,
            'geometry':LineString(get_route_line(shortest_paths[0],geo_dict))
            },
        index=[0],
        crs=links.crs
    ).to_crs('epsg:4326').to_json()
    color='yellow'
    tooltip = folium.GeoJsonTooltip(fields= ['tripid','length','detour','jaccard_exact','jaccard_buffer'])
    custom_route = folium.GeoJson(custom_route,name='Custom',
                style_function=lambda x,
                color=color: {'color': color, 'weight': 12, 'opacity':0.5},
                tooltip=tooltip,
                highlight_function=lambda x: {'color': color, 'weight': 20}
                )
    return custom_route

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
        #     difference_ft = (modeled_edges.length.sum() - chosen_edges.length.sum()).abs()

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