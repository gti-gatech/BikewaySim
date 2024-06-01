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

import similaritymeasures


sys.path.insert(0,str(Path.cwd().parent))
from network.src import modeling_turns
#import stochastic_optimization.stochastic_optimization as stochastic_optimization

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
- Change the beta array structure to allow for flexible impedance functions
that can be archived after optimization (not sure if this is possible)

- Dict gives the attribute names


'''

########################################################################################

# Pre-Calibration Functions

########################################################################################


def match_results_to_ods(match_results):
    #for the shortest path routing step, takes match_results and gets all unique od pairs
    ods = [(item['origin_node'],item['destination_node']) for key, item in match_results.items()]
    ods = np.unique(np.array(ods),axis=0)
    ods = [tuple(row) for row in ods]
    return ods

########################################################################################

# Impedance Calibration

########################################################################################

def impedance_calibration(betas:np.array,
                          past_betas,past_vals,
                          betas_links:dict,betas_turns:dict,
                          ods:list,match_results,
                          link_impedance_function,
                          turn_impedance_function,
                          links:pd.DataFrame,turns:pd.DataFrame,turn_G:nx.digraph,
                          objective_function,
                          objective_function_kwargs):
    
    #round the betas
    betas = np.round(betas,1)
    
    #prevent negative link weights
    if (betas < 0).any():
        print('Negative Impedance Coefficient Detected')
        val = 0
        return val
    #print(betas)

    #keep track of all the past betas used for visualization purposes
    past_betas.append(tuple(betas))

    #set link costs accordingly
    #print('Impedance Update')
    impedance_update(betas,betas_links,betas_turns,
                          link_impedance_function,
                          turn_impedance_function,
                          links,turns,turn_G)

    #find least impedance path
    #print('Shortest path routing')
    results_dict = {(start_node,end_node):impedance_path(turns,turn_G,start_node,end_node) for start_node, end_node in ods}

    #calculate the objective function
    #want to be able to provide the objective function and objective func arguements
    #print('calculating objective function')
    val_to_minimize = objective_function(match_results,results_dict,**objective_function_kwargs)

    #round the objective function value
    #TODO this should be objective function dependent
    val_to_minimize = np.round(val_to_minimize,2)

    past_vals.append(val_to_minimize)

    print(betas,val_to_minimize)
    return val_to_minimize

    # if objective_function == "exact_overlap":
    #     vals = [exact_overlap(tripid,item['start_node'],item['end_node'],match_results,results_dict,
    #               length_dict,**objective_function_args) for tripid, item in match_results.items()]
    #     vals = np.array(vals)
    #     if objective_function_args['standardize']:



    # if objective_function == "buffer_overlap":
    #     vals = [buffer_overlap(tripid,item['start_node'],item['end_node'],match_results,results_dict,
    #               geo_dict,objective_function_args['buffer_ft']) for tripid, item in match_results.items()]
        
    # if objective_function == "frechet":
    #     vals = [buffer_overlap(tripid,item['start_node'],item['end_node'],match_results,results_dict,
    #               geo_dict,objective_function_args['rdp_ft']) for tripid, item in match_results.items()]
    
    # if objective_function not in ['exact_overlap','buffer_overlap','frechet']:
    #     print("Select 'exact_overlap','buffer_overlap', or 'frechet' the objective function")

    # return val_to_minimize

    # #turn shortest paths dict to dataframe
    # shortest_paths = pd.DataFrame.from_dict(shortest_paths,orient='index')
    # shortest_paths.reset_index(inplace=True)
    # shortest_paths.columns = ['start','end','linkids','geometry','length']
    # #shortest_paths[['start','end']] = shortest_paths['index'].apply(lambda x: pd.Series(x))
    # #shortest_paths.drop(columns=['index'],inplace=True)

    # #add modeled paths to matched_traces dataframe
    # merged = matched_traces.merge(shortest_paths,on=['start','end'],suffixes=(None,'_modeled'))

    # if exact:
    #     sum_all = merged['length'].sum() * 5280
    #     all_overlap = 0

    #     for idx, row in merged.iterrows():
    #         #find shared edges
    #         chosen_and_shortest = row['linkids_modeled'] & row['linkids']
    #         #get the lengths of those links
    #         overlap_length = links.set_index('linkid').loc[list(chosen_and_shortest)]['length_ft'].sum()
    #         #overlap_length = np.sum([link_lengths.get(link_tup,'error') for link_tup in chosen_and_shortest])
    #         all_overlap += overlap_length

    #     #calculate objective function value
    #     val = all_overlap / sum_all
    #     print('Exact overlap percent is:',np.round(val*100,1),'%')
    
    # #calculate approximate overlap (new approach)
    # else:
    #     #buffer and dissolve generated route and matched route
    #     buffer_ft = 500

    #     merged.set_geometry('geometry',inplace=True)
    #     merged['buffered_geometry'] = merged.buffer(buffer_ft)
    #     merged.set_geometry('buffered_geometry',inplace=True)
    #     merged['area'] = merged.area

    #     merged.set_geometry('geometry_modeled',inplace=True)
    #     merged['buffered_geometry_modeled'] = merged.buffer(buffer_ft)
    #     merged.set_geometry('buffered_geometry_modeled',inplace=True)
    #     merged['area_modeled'] = merged.area

    #     #for each row find intersection between buffered features
    #     merged['intersection'] = merged.apply(lambda row: row['buffered_geometry'].intersection(row['buffered_geometry_modeled']), axis=1)

    #     # merged['intersection'] = merged.apply(
    #     #     lambda row: shapely.intersection(row['buffered_geometry'],row['buffered_geometry_modeled']))
    #     merged.set_geometry('intersection',inplace=True)
    #     merged['intersection_area'] = merged.area

    #     #find the overlap with the total area (not including intersections)
    #     #if the modeled/chosen links are different, then overlap decreases
    #     #punishes cirquitious modeled routes that utilize every link in the chosen one but include extraneous ones
    #     merged['overlap'] = merged['intersection_area'] / (merged['area_modeled'] + merged['area'] - merged['intersection_area'])

    #     #find average overlap (using median to reduce impact of outliers?)
    #     val = merged['overlap'].median()
    #     print('Median overlap percent is:',np.round(val*100,1),'%')
    
    # if follow_up:
    #     return merged

    # return -val#, merged

########################################################################################

# Impedance Routing

########################################################################################

def impedance_path(turns,turn_G,o,d):
    #NOTE: without these it'll throw a 'the result is ambiguous error'
    o = int(o)
    d = int(d)
    
    turn_G, virtual_starts, virtual_ends = modeling_turns.add_virtual_links_new(turns,turn_G,[o],[d])

    length, edge_list = nx.single_source_dijkstra(turn_G,source=o,target=d,weight='weight')
    edge_list = edge_list[1:-1] #chop off the virtual nodes added
    turn_G = modeling_turns.remove_virtual_links_new(turn_G,virtual_starts,virtual_ends)
    return {'length':np.round(length,1), 'edge_list':edge_list}


########################################################################################

# Network Impedance Update

########################################################################################

def impedance_update(betas:np.array,betas_links:dict,betas_turns:dict,
                     link_impedance_function,
                     turn_impedance_function,
                     links:pd.DataFrame,turns:pd.DataFrame,turn_G:nx.digraph):
    '''
    This function takes in the betas, impedance functions, and network objects
    and updates the network graph accordingly.
    '''
    #update link costs
    links = link_impedance_function(betas, betas_links, links)
    cost_dict = dict(zip(links['linkid'],links['link_cost']))
    turns['source_link_cost'] = turns['source_linkid'].map(cost_dict)
    turns['target_link_cost'] = turns['target_linkid'].map(cost_dict)

    #update turn costs
    turns = turn_impedance_function(betas, betas_turns, turns)

    #cacluate new total cost and round to tenth place
    turns['total_cost'] = (turns['source_link_cost'] + turns['target_link_cost'] + turns['turn_cost']).round(1)

    #round the rest too
    turns['source_link_cost'] = turns['source_link_cost'].round(1)
    turns['target_link_cost'] = turns['target_link_cost'].round(1)
    turns['turn_cost'] = turns['turn_cost'].round(1)

    #update turn network graph with final cost
    cols = ['source_linkid','source_reverse_link','target_linkid','target_reverse_link','total_cost']
    updated_edge_costs = {((row[0],row[1]),(row[2],row[3])):row[4] for row in turns[cols].itertuples(index=False)}
    nx.set_edge_attributes(turn_G,values=updated_edge_costs,name='weight')


def back_to_base_impedance(link_impedance_col,links,turns,turn_G):
    '''
    This function reverts the network graph back to base impedance (distance or travel time)
    with all turns as 0 cost
    '''

    #update link costs
    links['link_cost'] = links[link_impedance_col]
    cost_dict = dict(zip(links['linkid'],links['link_cost']))
    turns['source_link_cost'] = turns['source_linkid'].map(cost_dict)
    turns['target_link_cost'] = turns['target_linkid'].map(cost_dict)

    #cacluate new total cost
    turns['total_cost'] = (turns['source_link_cost'] + turns['target_link_cost'])

    #update turn network graph with final cost
    cols = ['source_linkid','source_reverse_link','target_linkid','target_reverse_link','total_cost']
    updated_edge_costs = {((row[0],row[1]),(row[2],row[3])):row[4] for row in turns[cols].itertuples(index=False)}
    nx.set_edge_attributes(turn_G,values=updated_edge_costs,name='weight')


########################################################################################

# Objective Functions

########################################################################################

def exact_overlap(match_results,results_dict,**kwargs):
    #TODO have a condition that ensures that the required kwargs are provided

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

        result.append((intersection_length,chosen_length))
    
    result = np.array(result)

    if kwargs['standardize']:
        #average intersect over chosen length
        result = np.mean(result[:,0] / result[:,1])
    else:
        #total intersect over total chosen length
        result = np.sum(result[:,0]) / np.sum(result)
    
    #return negative result because we want to minimize
    return -result

def first_preference_recovery(match_results,results_dict,**kwargs):
    '''
    Seen in Meister et al. 2024: https://doi.org/10.1016/j.jcmr.2024.100018

    "FPR is the percentage of correct predictions assuming that the predicted choice
    is the one with the highest choice probability"

    In this case, we're just looking at the similarity between the modeled route
    and the chosen route (not choice proababilities). A correct modeled route will
    contain all of the links included in the map matched trip. An overlap threshold
    controls what percentage of intersection between the chosen and modeled route is
    needed to be considered a match. A 100% overlap threshold means that the modeled
    route contains all of the links included in the chosen route. A 0% overlap threshold
    means that the modeled route doesn't need to contain any of the other 

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

        result.append((intersection_length,chosen_length))
    
    result = np.array(result)

    if kwargs['standardize']:
        #average intersect over chosen length
        result = np.mean(result[:,0] / result[:,1])
    else:
        #total intersect over total chosen length
        result = np.sum(result[:,0]) / np.sum(result)
    
    #return negative result because we want to minimize
    return -result


def buffer_overlap(match_results,results_dict,**kwargs):
    
    result = []
    
    for tripid, item in match_results.items():

        start_node = item['origin_node']
        end_node = item['destination_node']

        #retrieve linkids in (linkid:int,reverse_link:boolean) format
        chosen = [tuple(row) for row in match_results[tripid]['matched_edges'].to_numpy()]
        modeled_edges = results_dict[(start_node,end_node)]['edge_list']

        #get geos (non-directional)
        chosen_geo = [kwargs['geo_dict'][linkid[0]] for linkid in chosen]
        #shortest_geo = [kwargs['geo_dict'][linkid[0]] for linkid in shortest]
        modeled_geo = [kwargs['geo_dict'][linkid[0]] for linkid in modeled_edges]

        #turn into linestring and then buffer
        chosen_geo = MultiLineString(chosen_geo).buffer(kwargs['buffer_ft'])
        modeled_geo = MultiLineString(modeled_geo).buffer(kwargs['buffer_ft'])

        chosen_area = chosen_geo.area
        modeled_area = modeled_geo.area
        intersection_area = chosen_geo.intersection(modeled_geo).area
        net_area = chosen_area+modeled_area-intersection_area

        result.append((intersection_area,net_area))

    result = np.array(result)

    if kwargs['standardize']:
        #average intersect over area
        result = np.mean(result[:,0] / result[:,1])
    else:
        #total intersect over total chosen area
        result = np.sum(result[:,0]) / np.sum(result)
 
    return -result


#TODO use frechet area instead https://towardsdatascience.com/gps-trajectory-clustering-with-python-9b0d35660156
def frechet(match_results,results_dict,**kwargs):
    
    result = []
    
    for tripid, item in match_results.items():

        start_node = item['origin_node']
        end_node = item['destination_node']

        #retrieve tuples of the format (linkid:int,reverse_link:boolean)
        chosen = [tuple(row) for row in match_results[tripid]['matched_edges'].to_numpy()]
        #shortest = [tuple(row) for row in match_results[linkid]['shortest_edges'].to_numpy()]
        modeled = results_dict[(start_node,end_node)]['edge_list']

        chosen_geo = [retrieve_coordinates(link,kwargs['geo_dict']) for link in chosen]
        modeled_geo = [retrieve_coordinates(link,kwargs['geo_dict']) for link in modeled]

        #turn to a single line
        chosen_geo = LineString(np.vstack(chosen_geo))
        modeled_geo = LineString(np.vstack(modeled_geo))

        #simplify with rdp
        chosen_coords = np.array(chosen_geo.simplify(kwargs['rdp_ft']).coords)
        modeled_coords = np.array(modeled_geo.simplify(kwargs['rdp_ft']).coords)

        #find frechet distance
        # see https://github.com/topics/trajectory-similarity for documentation
        frechet_distance = similaritymeasures.frechet_dist(chosen_coords,modeled_coords)

        result.append(frechet_distance)

    result = np.array(result).mean()
    
    #can minimize total frechet distance or an average value
    #don't take negative because we're already taking minimum
    return result

#retrieve coordinates, revesing coordinate sequence if neccessary
def retrieve_coordinates(link,geo_dict):
    line = np.array(geo_dict[link[0]].coords)
    if link[1] == True:
        line = line[::-1]
    return line

def objective_function(betas,betas_links,betas_turns,links,
                       pseudo_links,pseudo_G,
                       matched_traces,link_impedance_function,
                       turn_impedance_function,exact,follow_up):

    #prevent negative link weights
    if (betas < 0).any():
        val = 0
        return val

    #TODO bring most of this out as functions
    #use initial/updated betas to calculate link costs
    print('setting link costs')
    links = link_impedance_function(betas, betas_links, links)
    cost_dict = dict(zip(links['linkid'],links['link_cost']))
    
    #TODO is this the best way to accomplish this?
    #get_geo = links.loc[links.groupby(['source','target'])['link_cost'].idxmin(),['source','target','geometry']]

    #add link costs to pseudo_links
    pseudo_links['source_link_cost'] = pseudo_links['source_linkid'].map(cost_dict)
    pseudo_links['target_link_cost'] = pseudo_links['target_linkid'].map(cost_dict)

    #use initial/updated betas to calculate turn costs
    print('setting turn costs')
    pseudo_links = turn_impedance_function(betas, betas_turns, pseudo_links)

    #assign na turns a cost of 0
    #pseudo_links.loc[pseudo_links['turn_cost'].isna(),'turn_cost'] = 0

    #add links and multiply by turn cost
    pseudo_links['total_cost'] = pseudo_links['source_link_cost'] + pseudo_links['target_link_cost'] + pseudo_links['turn_cost']

    #check for negative costs (set them to zero for now?)
    #pseudo_links.loc[pseudo_links['total_cost'] < 0, 'total_cost'] = 1

    #round values
    #pseudo_links['total_cost'] = pseudo_links['total_cost'].round(1)

    #only keep link with the lowest cost
    print('finding lowest cost')
    costs = pseudo_links.set_index(['source','target'])['total_cost']#pseudo_links.groupby(['source','target'])['total_cost'].min()

    # #get linkids used to retreive the correct link geometries for overlap function?
    # #we don't care about direction in this case
    # print('finding link id for lowest cost')
    # source_cols = ['source','source_linkid','source_reverse_link']
    # target_cols = ['target','target_linkid','target_reverse_link']
    # min_links = pseudo_links.loc[pseudo_links.groupby(['source','target'])['total_cost'].idxmin()]
    # source_links = min_links[source_cols]
    # target_links = min_links[target_cols]
    # source_links.columns = ['A_B','linkid','reverse_link']
    # target_links.columns = ['A_B','linkid','reverse_link']
    # #what keeps a_b from not being duplicated here?
    # linkids = pd.concat([source_links,target_links],ignore_index=True).drop_duplicates().set_index('A_B')

    #update edge weights
    print('updating edge weights')
    nx.set_edge_attributes(pseudo_G,values=costs,name='weight')

    #update edge ids (what was this for?)
    
    #do shortest path routing
    shortest_paths = {}
    print(f'Shortest path routing with coefficients: {betas}')    
    for source, targets in matched_traces.groupby('start')['end'].unique().items():

        #add virtual links to pseudo_G
        pseudo_G, virtual_edges = modeling_turns.add_virtual_links(pseudo_links,pseudo_G,source,targets)
        
        #perform shortest path routing for all target nodes from source node
        #(from one to all until target node has been visited)
        for target in targets:  
            #cant be numpy int64 or throws an error
            target = int(target)
            
            try:
                #TODO every result is only a start node, middle node, then end node
                length, node_list = nx.single_source_dijkstra(pseudo_G,source,target,weight='weight')
            except:
                print(source,target)
                length, node_list = nx.single_source_dijkstra(pseudo_G,source,target,weight='weight')

            #get edge list
            edge_list = node_list[1:-1]

            #get geometry from edges
            modeled_edges = links.set_index(['source','target']).loc[edge_list]

            # modeled_edges = links.merge(linkids.loc[edge_list],on=['linkid','reverse_link'],how='inner')
            # modeled_edges = gpd.GeoDataFrame(modeled_edges,geometry='geometry')

            shortest_paths[(source,target)] = {
                'edges': set(modeled_edges['linkid'].tolist()),
                'geometry':MultiLineString(modeled_edges['geometry'].tolist()),#modeled_edges.dissolve()['geometry'].item(),
                'length':MultiLineString(modeled_edges['geometry'].tolist()).length
                }

        #remove virtual links
        pseudo_G = modeling_turns.remove_virtual_edges(pseudo_G,virtual_edges)
    
    print('calculating objective function')

    #turn shortest paths dict to dataframe
    shortest_paths = pd.DataFrame.from_dict(shortest_paths,orient='index')
    shortest_paths.reset_index(inplace=True)
    shortest_paths.columns = ['start','end','linkids','geometry','length']
    #shortest_paths[['start','end']] = shortest_paths['index'].apply(lambda x: pd.Series(x))
    #shortest_paths.drop(columns=['index'],inplace=True)

    #add modeled paths to matched_traces dataframe
    merged = matched_traces.merge(shortest_paths,on=['start','end'],suffixes=(None,'_modeled'))

    if exact:
        sum_all = merged['length'].sum() * 5280
        all_overlap = 0

        for idx, row in merged.iterrows():
            #find shared edges
            chosen_and_shortest = row['linkids_modeled'] & row['linkids']
            #get the lengths of those links
            overlap_length = links.set_index('linkid').loc[list(chosen_and_shortest)]['length_ft'].sum()
            #overlap_length = np.sum([link_lengths.get(link_tup,'error') for link_tup in chosen_and_shortest])
            all_overlap += overlap_length

        #calculate objective function value
        val = all_overlap / sum_all
        print('Exact overlap percent is:',np.round(val*100,1),'%')
    
    #calculate approximate overlap (new approach)
    else:
        #buffer and dissolve generated route and matched route
        buffer_ft = 500

        merged.set_geometry('geometry',inplace=True)
        merged['buffered_geometry'] = merged.buffer(buffer_ft)
        merged.set_geometry('buffered_geometry',inplace=True)
        merged['area'] = merged.area

        merged.set_geometry('geometry_modeled',inplace=True)
        merged['buffered_geometry_modeled'] = merged.buffer(buffer_ft)
        merged.set_geometry('buffered_geometry_modeled',inplace=True)
        merged['area_modeled'] = merged.area

        #for each row find intersection between buffered features
        merged['intersection'] = merged.apply(lambda row: row['buffered_geometry'].intersection(row['buffered_geometry_modeled']), axis=1)

        # merged['intersection'] = merged.apply(
        #     lambda row: shapely.intersection(row['buffered_geometry'],row['buffered_geometry_modeled']))
        merged.set_geometry('intersection',inplace=True)
        merged['intersection_area'] = merged.area

        #find the overlap with the total area (not including intersections)
        #if the modeled/chosen links are different, then overlap decreases
        #punishes cirquitious modeled routes that utilize every link in the chosen one but include extraneous ones
        merged['overlap'] = merged['intersection_area'] / (merged['area_modeled'] + merged['area'] - merged['intersection_area'])

        #find average overlap (using median to reduce impact of outliers?)
        val = merged['overlap'].median()
        print('Median overlap percent is:',np.round(val*100,1),'%')
    
    if follow_up:
        return merged

    return -val#, merged

def follow_up(betas,links,pseudo_links,pseudo_G,matched_traces,exact=False):

    #
    modeled_trips = {}
    
    #use initial/updated betas to calculate link costs
    links = link_impedance_function(betas, links)
    cost_dict = dict(zip(links['linkid'],links['link_cost']))
    
    #TODO is this the best way to accomplish this?
    #get_geo = links.loc[links.groupby(['source','target'])['link_cost'].idxmin(),['source','target','geometry']]

    #add link costs to pseudo_links
    pseudo_links['source_link_cost'] = pseudo_links['source_linkid'].map(cost_dict)
    pseudo_links['target_link_cost'] = pseudo_links['target_linkid'].map(cost_dict)

    #use initial/updated betas to calculate turn costs
    pseudo_links = turn_impedance_function(betas, pseudo_links)

    #assign na turns a cost of 0
    pseudo_links.loc[pseudo_links['turn_cost'].isna(),'turn_cost'] = 0

    #add links and multiply by turn cost
    pseudo_links['total_cost'] = pseudo_links['source_link_cost'] + pseudo_links['target_link_cost'] + pseudo_links['turn_cost']

    #only keep link with the lowest cost
    costs = pseudo_links.groupby(['source','target'])['total_cost'].min()

    #get linkids used
    source_cols = ['source','source_linkid','source_reverse_link']
    target_cols = ['target','target_linkid','target_reverse_link']
    min_links = pseudo_links.loc[pseudo_links.groupby(['source','target'])['total_cost'].idxmin()]
    source_links = min_links[source_cols]
    target_links = min_links[target_cols]
    source_links.columns = ['A_B','linkid','reverse_link']
    target_links.columns = ['A_B','linkid','reverse_link']
    linkids = pd.concat([source_links,target_links],ignore_index=True).drop_duplicates().set_index('A_B')

    #update edge weights
    nx.set_edge_attributes(pseudo_G,values=costs,name='weight')
    
    #do shortest path routing
    shortest_paths = {}
    print(f'Shortest path routing with coefficients: {betas}')    
    for source, targets in matched_traces.groupby('start')['end'].unique().items():

        #add virtual links to pseudo_G
        pseudo_G, virtual_edges = modeling_turns.add_virtual_links(pseudo_links,pseudo_G,source,targets)
        
        #perform shortest path routing for all target nodes from source node
        #(from one to all until target node has been visited)
        for target in targets:  
            #cant be numpy int64 or throws an error
            target = int(target)
            
            try:
                #TODO every result is only a start node, middle node, then end node
                length, node_list = nx.single_source_dijkstra(pseudo_G,source,target,weight='weight')
            except:
                print(source,target)
                length, node_list = nx.single_source_dijkstra(pseudo_G,source,target,weight='weight')

            #get edge list
            edge_list = node_list[1:-1]

            #get geometry from edges
            modeled_edges = links.merge(linkids.loc[edge_list],on=['linkid','reverse_link'],how='inner')
            modeled_edges = gpd.GeoDataFrame(modeled_edges,geometry='geometry')
            
            #
            shortest_paths[(source,target)] = {
                'edges': set(modeled_edges['linkid'].tolist()),
                'geometry':MultiLineString(modeled_edges['geometry'].tolist()),
                'length':modeled_edges.length.sum()
                }

        #remove virtual links
        pseudo_G = modeling_turns.remove_virtual_edges(pseudo_G,virtual_edges)
    
    #turn shortest paths dict to dataframe
    shortest_paths = pd.DataFrame.from_dict(shortest_paths,orient='index')
    shortest_paths.reset_index(inplace=True)
    shortest_paths.columns = ['start','end','linkids','geometry','length']
    #shortest_paths[['start','end']] = shortest_paths['index'].apply(lambda x: pd.Series(x))
    #shortest_paths.drop(columns=['index'],inplace=True)

    #add modeled paths to matched_traces dataframe
    merged = matched_traces.merge(shortest_paths,on=['start','end'],suffixes=(None,'_modeled'))

    if exact:
        sum_all = merged['length'].sum() * 5280
        all_overlap = 0

        for idx, row in merged.iterrows():
            #find shared edges
            chosen_and_shortest = row['linkids_modeled'] & row['linkids']
            #get the lengths of those links
            overlap_length = links.set_index('linkid').loc[list(chosen_and_shortest)]['length_ft'].sum()
            #overlap_length = np.sum([link_lengths.get(link_tup,'error') for link_tup in chosen_and_shortest])
            all_overlap += overlap_length

        #calculate objective function value
        val = all_overlap / sum_all
        print('Exact overlap percent is:',np.round(val*100,1),'%')
    
    #calculate approximate overlap (new approach)
    else:
        #buffer and dissolve generated route and matched route
        buffer_ft = 500

        merged.set_geometry('geometry',inplace=True)
        merged['buffered_geometry'] = merged.buffer(buffer_ft)
        merged.set_geometry('buffered_geometry',inplace=True)
        merged['area'] = merged.area

        merged.set_geometry('geometry_modeled',inplace=True)
        merged['buffered_geometry_modeled'] = merged.buffer(buffer_ft)
        merged.set_geometry('buffered_geometry_modeled',inplace=True)
        merged['area_modeled'] = merged.area

        #for each row find intersection between buffered features
        merged['intersection'] = merged.apply(lambda row: row['buffered_geometry'].intersection(row['buffered_geometry_modeled']), axis=1)

        # merged['intersection'] = merged.apply(
        #     lambda row: shapely.intersection(row['buffered_geometry'],row['buffered_geometry_modeled']))
        merged.set_geometry('intersection',inplace=True)
        merged['intersection_area'] = merged.area

        #find the overlap with the total area (not including intersections)
        #if the modeled/chosen links are different, then overlap decreases
        #punishes cirquitious modeled routes that utilize every link in the chosen one but include extraneous ones
        merged['overlap'] = merged['intersection_area'] / (merged['area_modeled'] + merged['area'] - merged['intersection_area'])

        #find average overlap (using median to reduce impact of outliers?)
        val = merged['overlap'].median()
        print('Median overlap percent is:',np.round(val*100,1),'%')
    
    return -val#, merged         


import folium
import geopandas as gpd
from folium.plugins import MarkerCluster, PolyLineTextPath
from folium.map import FeatureGroup
def visualize_three_no_legend(chosen_line,shortest_line,modeled_line):
    '''
    Standard visualization of the chosen/shortest/modeled lines.
    Provide GeoSeries for each.
    '''
    #reproj
    chosen_line = chosen_line.to_crs(epsg='4326')
    shortest_line = shortest_line.to_crs(epsg='4326')
    modeled_line = modeled_line.to_crs(epsg='4326')

    #start_pt
    start_pt = list(chosen_line.iloc[0].coords)[0]
    end_pt = list(chosen_line.iloc[-1].coords)[-1]

    # reproject
    x_mean = chosen_line.unary_union.centroid.x
    y_mean = chosen_line.unary_union.centroid.y

    # Create a Folium map centered around the mean of the GPS points
    center = [y_mean,x_mean]
    mymap = folium.Map(location=center, zoom_start=13)

    # Convert GeoDataFrames to GeoJSON
    chosen_line_geojson = chosen_line.to_json()
    shortest_line_geojson = shortest_line.to_json()
    modeled_line_geojson = modeled_line.to_json()

    # Create FeatureGroups for each GeoDataFrame
    chosen_line_fg = FeatureGroup(name='Chosen Path')
    shortest_line_fg = FeatureGroup(name='Shortest Path')
    modeled_line_fg = FeatureGroup(name='Modeled Path')

    # Add GeoJSON data to FeatureGroups
    folium.GeoJson(chosen_line_geojson, name='Chosen Path',
                style_function=lambda x: {'color': '#fc8d62', 'weight': 12}).add_to(chosen_line_fg)
    folium.GeoJson(shortest_line_geojson, name='Shortest Path',
                style_function=lambda x: {'color': '#66c2a5', 'weight': 8}).add_to(shortest_line_fg)
    folium.GeoJson(modeled_line_geojson, name='Modeled Path',
                style_function=lambda x: {'color': '#8da0cb','weight': 8}).add_to(modeled_line_fg)

    # Add FeatureGroups to the map
    chosen_line_fg.add_to(mymap)
    shortest_line_fg.add_to(mymap)
    modeled_line_fg.add_to(mymap)

    # Add start and end points with play and stop buttons
    start_icon = folium.Icon(color='green',icon='play',prefix='fa')
    end_icon = folium.Icon(color='red',icon='stop',prefix='fa')
    folium.Marker(location=[start_pt[1], start_pt[0]],icon=start_icon).add_to(mymap)
    folium.Marker(location=[end_pt[1], end_pt[0]],icon=end_icon).add_to(mymap)

    # Add layer control to toggle layers on/off
    folium.LayerControl().add_to(mymap)

    return mymap  # Uncomment if you are using Jupyter notebook

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
#     x = minimize(objective_function, bounds, args=(links,G,ods,trips_df,matched_traces,durations), method='pso')
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