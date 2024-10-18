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
from bikewaysim.impedance_calibration import stochastic_optimization, loss_functions

########################################################################################

# Post Calibration Functions

########################################################################################

def post_calibration_routing_all(user):

    links, turns, length_dict, geo_dict, turn_G = rustworkx_routing_funcs.import_calibration_network(config)
    with (config['calibration_fp']/'ready_for_calibration.pkl').open('rb') as fh:
        full_set = pickle.load(fh)
    no_routing_fps = file_check_post_routing(user)

    for no_routing_fp in tqdm(no_routing_fps):
        
        with no_routing_fp.open('rb') as fh:
            no_routing = pickle.load(fh)
        
        full_set_subset = {tripid:item for tripid, item in full_set.items() if tripid in no_routing['trips_calibrated']}
        full_ods = stochastic_optimization.match_results_to_ods_w_year(full_set_subset)

        #get best val
        if np.isnan(np.nanmin(no_routing['past_vals'])) == False:
            routing_results = post_calibration_routing(
                links,turns,turn_G,
                stochastic_optimization.link_impedance_function,
                stochastic_optimization.turn_impedance_function,
                full_ods,'travel_time_min',
                no_routing['set_to_zero'],no_routing['set_to_inf'],
                no_routing
                )
            with (no_routing_fp.stem / 'test').open('wb') as fh:
                pickle.dump(routing_results,fh)
        else:
            print('best val is nan for',no_routing_fp.stem)
    
def post_calibration_routing(
        links,turns,turn_G,
        link_impedance_function,
        turn_impedance_function,
        full_ods,base_impedance_col,
        set_to_zero,set_to_inf,
        calibration_result):
    
    '''
    Used after the calibration to return the least impedance paths using the calibrated coefficients
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

    # create networks by year    
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

def post_calibration_routing_any(
        betas,
        links,turns,turn_G,
        link_impedance_function,
        turn_impedance_function,
        full_ods,base_impedance_col,
        set_to_zero,set_to_inf,
        calibration_result):
    
    '''
    Use for getting route using the past vals
    '''

    # base_impedance_col = "travel_time_min" # set the base impedance (default is travel time)
    # betas = [x['beta'] for x in calibration_result['betas_tup']] # get betas

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

    # create networks by year    
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
    links, turns, length_dict, geo_dict, turn_G = rustworkx_routing_funcs.import_calibration_network(config)
    with (config['calibration_fp']/'ready_for_calibration.pkl').open('rb') as fh:
        full_set = pickle.load(fh)
    
    for key, item in full_set.items():
        # extract chosen and shortest routes
        chosen = item['matched_edges'].values
        shortest = item['shortest_edges'].values

        #compute the loss values (store the intermediates too so total vs mean methods can be compared)
        shortest_jaccard_exact_intersection, shortest_jaccard_exact_union =  loss_functions.jaccard_exact(chosen,shortest,length_dict)
        shortest_jaccard_exact = shortest_jaccard_exact_intersection / shortest_jaccard_exact_union
        shortest_jaccard_buffer_intersection, shortest_jaccard_buffer_union =  loss_functions.jaccard_buffer(chosen,shortest,geo_dict)
        shortest_jaccard_buffer = shortest_jaccard_buffer_intersection / shortest_jaccard_buffer_union

        # add to full set in additon to what's already there
        full_set[key].update({
            'chosen_length': round(np.array([length_dict.get(tripid[0],False) for tripid in chosen]).sum()/5280,2),
            'shortest_length': round(np.array([length_dict.get(tripid[0],False) for tripid in shortest]).sum()/5280,2),
            'chosen_detour': round(loss_functions.detour_factor(chosen,shortest,length_dict),2),
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






# using try here in case the calibration didn't find any valid set of coefficients that didn't result in negative link attributes
    try:
        # return the shortest paths with the best set of coefficients found
        modeled_ods = post_calibration.post_calibration_routing(links,turns,turn_G,match_results_to_ods_w_year(full_set),base_impedance_col,set_to_zero,set_to_inf,calibration_result)
        # export
        with export_utils.uniquify(routing_fp).open('wb') as fh:
            pickle.dump(modeled_ods,fh)
        # loss values calculated later

    except:
        print('No valid set of coefficients available, skipping')










def post_calibration_loss(user=False):
    '''
    Calcualte loss values and metrics for the modeled routes (for all calibration runs)
    '''
    _, _, length_dict, geo_dict, _ = rustworkx_routing_funcs.import_calibration_network(config)

    calibration_results_dir, post_calibration_routing_dir, post_calibration_loss_dir, calibrations = file_check(user)

    # import all trips
    with (config['calibration_fp']/'ready_for_calibration.pkl').open('rb') as fh:
        full_set_all = pickle.load(fh)

    for calibration in tqdm(calibrations):
        # skip if it's already present
        if (calibration_results_dir / f"{calibration}.pkl").exists():
            continue

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
            skip = False            
            chosen = item['matched_edges'].values
            shortest = item['shortest_edges'].values
            
            # NOTE one limitation of this is that if the origin node and destination node change at any point then
            # than this step will break
            od = (item['origin_node'],item['destination_node'])
            modeled = results_dict.get(od)
            if modeled is None:
                print(f"Missing shortest paths in calibration {calibration}")
                skip = True
                break
            modeled = modeled['edge_list']
        
            #compute the loss values (store the intermediates too so total vs mean methods can be compared)
            modeled_jaccard_exact_intersection, modeled_jaccard_exact_union =  loss_functions.jaccard_exact(chosen,modeled,length_dict)
            modeled_jaccard_exact = modeled_jaccard_exact_intersection / modeled_jaccard_exact_union
            modeled_jaccard_buffer_intersection, modeled_jaccard_buffer_union =  loss_functions.jaccard_buffer(chosen,modeled,geo_dict)
            modeled_jaccard_buffer = modeled_jaccard_buffer_intersection / modeled_jaccard_buffer_union

            modeled_results_dict[tripid] = {
                'modeled_edges': pd.DataFrame(modeled,columns=['linkid','reverse_link']),
                'modeled_length': round(np.array([length_dict.get(tripid[0],0) for tripid in modeled]).sum()/5280,1),
                'modeled_detour': round(loss_functions.detour_factor(modeled,shortest,length_dict),2),
                'modeled_jaccard_exact': round(modeled_jaccard_exact,2),
                'modeled_jaccard_exact_intersection': round(modeled_jaccard_exact_intersection,2),
                'modeled_jaccard_exact_union': round(modeled_jaccard_exact_union,2),
                'modeled_jaccard_buffer': round(modeled_jaccard_buffer,2),
                'modeled_jaccard_buffer_intersection': round(modeled_jaccard_buffer_intersection,2),
                'modeled_jaccard_buffer_union': round(modeled_jaccard_buffer_union,2),
            }
        
        if skip:
            continue

        with (post_calibration_loss_dir / f"{calibration}.pkl").open('wb') as fh:
            pickle.dump(modeled_results_dict,fh)

### Aggregate ###

def post_calibration_metadata(user=False):
    '''
    Returns dataframe with the runtime, optimization settings, and time of estimation
    '''

    calibration_results_dir, post_calibration_routing_dir, post_calibration_loss_dir, calibrations = file_check(user)

    meta_data = []
    for calibration in calibrations:
        with (calibration_results_dir / f"{calibration}.pkl").open('rb') as fh:
            calibration_result = pickle.load(fh)
        calibration_params = get_calibration_name_parameters(calibration,user)
        calibration_result0 = {key:item for key,item in calibration_result.items() if key in ('runtime','time','objective_function','settings')}
        calibration_result0['obj'] = np.min(np.array(calibration_result['past_vals'])[:,-1])
        meta_data0 = {**calibration_params,**calibration_result}
        meta_data.append(meta_data0)
    meta_data = pd.DataFrame().from_records(meta_data)

    return meta_data


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

### Disaggregate ###


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



#### Utility functions ####


def file_check_post_routing(user=False):
    '''
    Coordinates between the results and post calibration directories to make sure that both files exist
    '''
    if user:
        calibration_results_dir = config['calibration_fp']/"user_calibration_results"
        post_calibration_routing_dir = config['calibration_fp']/"user_post_calibration_routing"
    else:
        calibration_results_dir = config['calibration_fp']/"calibration_results"
        post_calibration_routing_dir = config['calibration_fp']/"post_calibration_routing"

    # get the intersection of both
    calibration_results = [x.stem for x in calibration_results_dir.glob('*.pkl')]
    post_calibration_routing = [x.stem for x in post_calibration_routing_dir.glob('*.pkl')]
    
    no_routing = list(set(calibration_results) - set(post_calibration_routing))
    no_routing_fps = [x for x in calibration_results_dir.glob('*.pkl') if x.stem in no_routing]

    return no_routing_fps


def file_check(user=False):
    '''
    Coordinates between the results and post calibration directories to make sure that both files exist
    '''
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
    '''
    Extracts the user number and calbration name from the file name
    '''
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