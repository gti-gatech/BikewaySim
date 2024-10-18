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
from bikewaysim.impedance_calibration import stochastic_optimization, post_calibration

########################################################################################

# Objective Functions

# TODO clean this up because these funcitons should be easily generalizable
# try using classes for this later

########################################################################################


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
    chosen = route_utils.get_route_line(chosen,geo_dict)
    other = route_utils.get_route_line(other,geo_dict)
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

def frechet_distance(chosen,modeled,geo_dict):
    '''
    Returns the frechet distance between the chosen and the modeled route. No weighting by route length, just the average.
    '''
    modeled = route_utils.get_route_line(modeled,geo_dict)
    chosen = route_utils.get_route_line(chosen,geo_dict)
    result = similaritymeasures.frechet_dist(modeled,chosen)
    return result

def trace_difference(modeled,geo_dict,tripid,trace_dict):
    '''
    Returns the frechet distance between the chosen and the modeled route. No weighting by route length, just the average.
    '''
    modeled = route_utils.get_route_line(modeled,geo_dict)
    trace = trace_dict[tripid]
    result = similaritymeasures.frechet_dist(modeled,trace)
    return result

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

