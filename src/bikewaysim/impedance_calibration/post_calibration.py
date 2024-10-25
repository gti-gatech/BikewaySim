#https://keurfonluu.github.io/stochopy/api/optimize.html

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from bikewaysim.paths import config
from bikewaysim.routing import rustworkx_routing_funcs
from bikewaysim.impedance_calibration import stochastic_optimization, loss_functions, utils, impedance_functions

########################################################################################

# Shortest/Chosen Path Loss/Detour

########################################################################################

def shortest_loss():
    '''
    Calculate loss values and metrics for the shortest path
    '''

    # Retreive shortest and chosen stats first (already calculated the shortest paths)
    links, turns, length_dict, geo_dict, turn_G = rustworkx_routing_funcs.import_calibration_network(config)
    with (config['calibration_fp']/'ready_for_calibration.pkl').open('rb') as fh:
        match_results = pickle.load(fh)
    
    for key, item in match_results.items():
        # extract chosen and shortest routes
        chosen = item['matched_edges'].values
        shortest = item['shortest_edges'].values

        #compute the loss values (store the intermediates too so total vs mean methods can be compared)
        shortest_jaccard_exact_intersection, shortest_jaccard_exact_union =  loss_functions.jaccard_exact(chosen,shortest,length_dict)
        shortest_jaccard_exact = shortest_jaccard_exact_intersection / shortest_jaccard_exact_union
        shortest_jaccard_buffer_intersection, shortest_jaccard_buffer_union =  loss_functions.jaccard_buffer(chosen,shortest,geo_dict)
        shortest_jaccard_buffer = shortest_jaccard_buffer_intersection / shortest_jaccard_buffer_union

        # add to full set in additon to what's already there
        match_results[key].update({
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
        pickle.dump(match_results,fh)


########################################################################################

# Post Calibration Functions

########################################################################################
'''
These get run after model calibration
'''

def post_calibration_routing():
    '''
    Run this with a calibration result to get all of the least impedance routes
    using the calibrated impedance function.    
    '''

    links, turns, length_dict, geo_dict, turn_G = rustworkx_routing_funcs.import_calibration_network(config)
    with (config['calibration_fp']/'ready_for_calibration.pkl').open('rb') as fh:
        match_results = pickle.load(fh)
    
    # get results filepaths
    result_fps, routing_fps, loss_fps = utils.get_directories()

    for result_fp in tqdm(result_fps):
        # skip if it's already been calculated
        if result_fp.stem in [x.stem for x in routing_fps]:
            continue

        with result_fp.open('rb') as fh:
            result = pickle.load(fh)
        betas = [x['beta'] for x in result['betas_tup']] # get betas

        # impedance routing part will handle this
        # # if the best is nan then skip 
        # if np.isnan(np.nanmin(result['past_vals'])):
        #     continue

        # insures that we are only routing the trips that were calibrated with
        # a different function will be used to apply impedances to other trips
        match_results_subset = {tripid:item for tripid, item in match_results.items() if tripid in result['trips_calibrated']}
        ods = utils.match_results_to_ods_w_year(match_results_subset)
 
        # for backwards compatibility
        # these only just now got added to the result dictionary 
        if result.get('link_impedance_function') is None:
            result['link_impedance_function'] = impedance_functions.link_impedance_function
        if result.get('turn_impedance_function') is None:
            result['turn_impedance_function'] = impedance_functions.turn_impedance_function
        if result.get('base_impedance_col') is None:
            result['base_impedance_col'] = 'travel_time_min'
        if result.get('base_link_col') is None:
            result['base_link_col'] = None

        #PROBLEM, need the impedance function used
        modeled_dict = stochastic_optimization.impedance_routing(
            betas,
            result['betas_tup'],
            result['link_impedance_function'],
            result['base_impedance_col'],
            result['base_link_col'],       
            result['turn_impedance_function'],
            links,
            turns,
            turn_G,
            ods,
            result['set_to_zero'],
            result['set_to_inf']
            )
        if modeled_dict is None:
            print('\n',result_fp.stem,'has negative link costs')
            continue
        
        with (config['calibration_fp'] / 'routing' / str(result_fp.stem + '.pkl')).open('wb') as fh:
            pickle.dump(modeled_dict,fh)

def post_calibration_loss(rewrite=False):
    '''
    Calculate loss values and metrics for the modeled routes (for all calibration runs)
    '''
    
    links, turns, length_dict, geo_dict, turn_G = rustworkx_routing_funcs.import_calibration_network(config)
    with (config['calibration_fp']/'ready_for_calibration.pkl').open('rb') as fh:
        match_results = pickle.load(fh)
    
    # get results filepaths
    result_fps, routing_fps, loss_fps = utils.get_directories()

    # cross reference result and routing
    results_stems = [x.stem for x in result_fps]
    routing_stems = [x.stem for x in routing_fps]
    loss_stems = list(set.intersection(set(results_stems),set(routing_stems)))

    for loss_stem in tqdm(loss_stems):
        # skip if it's already present
        if rewrite:    
            if (config['calibration_fp'] / 'loss' / f"{loss_stem}.pkl").exists():
                continue

        # load the calibration result (has the estimated betas)
        with (config['calibration_fp'] / 'results' / f"{loss_stem}.pkl").open('rb') as fh:
            result = pickle.load(fh)

        # subset to the trips used for calibration (in case there was some sort of subsetting)
        match_results_subset = {tripid:item for tripid, item in match_results.items() if tripid in result['trips_calibrated']}
        
        # load the post calibration results (includes the impedance path taken for each trip)
        with (config['calibration_fp'] / 'routing' / f"{loss_stem}.pkl").open('rb') as fh:
            results_dict = pickle.load(fh)
        
        loss_dict = {}
        # we could try storing these
        for tripid, item in match_results_subset.items(): # make sure you import this
            skip = False            
            
            chosen = item['matched_edges'].values
            shortest = item['shortest_edges'].values
            
            # NOTE one limitation of this is that if the origin node and destination node change at any point then
            # than this step will break
            od = (item['origin_node'],item['destination_node'])
            modeled = results_dict.get(od)
            if modeled is None:
                print(f"Missing shortest paths in calibration {loss_stem}")
                skip = True
                break
            modeled = modeled['edge_list']
        
            #compute the loss values (store the intermediates too so total vs mean methods can be compared)
            modeled_jaccard_exact_intersection, modeled_jaccard_exact_union =  loss_functions.jaccard_exact(chosen,modeled,length_dict)
            modeled_jaccard_exact = modeled_jaccard_exact_intersection / modeled_jaccard_exact_union
            modeled_jaccard_buffer_intersection, modeled_jaccard_buffer_union =  loss_functions.jaccard_buffer(chosen,modeled,geo_dict)
            modeled_jaccard_buffer = modeled_jaccard_buffer_intersection / modeled_jaccard_buffer_union

            loss_dict[tripid] = {
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

        with (config['calibration_fp'] / 'loss' / f"{loss_stem}.pkl").open('wb') as fh:
            pickle.dump(loss_dict,fh)

########################################################################################

# Aggregate Model Results

########################################################################################
'''
Functions for retrieving the aggregated model statistics (such as the objective function,
the number of trips used, the number of iterations, population size, etc.)
'''

def metadata_dataframe():
    '''
    Returns dataframe with the runtime, optimization settings, and time of calibration
    '''
    
    result_fps, routing_fps, loss_fps = utils.get_directories()
    
    meta_cols = ['set_to_zero','set_to_inf','objective_function','runtime','time']

    meta_data = []
    for result_fp in result_fps:
        with result_fp.open('rb') as fh:
            result = pickle.load(fh)
        name_params = utils.get_name_parameters(result_fp)

        meta_data0 = {
            **name_params,
            **{key:item for key,item in result.items() if key in meta_cols},
            **result['settings'],
            'num_trips': len(result['trips_calibrated']),
            'status': result['results'].status,
        }
        meta_data0.update(**meta_data0['options'])
        meta_data0.pop('options')
        meta_data.append(meta_data0)
    
    meta_data = pd.DataFrame().from_records(meta_data)

    return meta_data

def betas_dataframe():
    '''
    Returns dataframe of all the estimated coefficients in the columns where each row is a model
    '''

    result_fps, routing_fps, loss_fps = utils.get_directories()

    beta_vals = []
    for result_fp in result_fps:
        with result_fp.open('rb') as fh:
            result = pickle.load(fh)
        name_params = utils.get_name_parameters(result_fp)
        beta_vals.append({**name_params,**{x['col']:round(x['beta'],3) for x in result['betas_tup']}})
    
    beta_vals = pd.DataFrame().from_records(beta_vals)

    return beta_vals

def aggregated_loss_dataframe():
    '''
    Calculates aggregated loss stats for all of the calibration results.

    Every row is a calibration result. If user/subset based, then every row is a calibration result for a user/subset.
    
    '''
    
    result_fps, routing_fps, loss_fps = utils.get_directories()

    # disaggregated
    aggregated_loss = []
    for loss_fp in loss_fps:
        with loss_fp.open('rb') as fh:
            loss_dict = pickle.load(fh)

        jaccard_exact_mean = np.array([item['modeled_jaccard_exact'] for tripid, item in loss_dict.items()]).mean()
        jaccard_exact_total = np.array([(item['modeled_jaccard_exact_intersection'],item['modeled_jaccard_exact_union']) for tripid, item in loss_dict.items()])
        jaccard_exact_total = jaccard_exact_total[:,0].sum() / jaccard_exact_total[:,1].sum()

        jaccard_buffer_mean = np.array([item['modeled_jaccard_buffer'] for tripid, item in loss_dict.items()]).mean()
        jaccard_buffer_total = np.array([(item['modeled_jaccard_buffer_intersection'],item['modeled_jaccard_buffer_union']) for tripid, item in loss_dict.items()])
        jaccard_buffer_total = jaccard_buffer_total[:,0].sum() / jaccard_buffer_total[:,1].sum()

        name_params = utils.get_name_parameters(loss_fp)

        aggregated_loss.append({
            **name_params, # contains the userid, calibration name and run number
            'jaccard_exact_mean': round(jaccard_exact_mean,2),
            'jaccard_exact_total': round(jaccard_exact_total,2),
            'jaccard_buffer_mean': round(jaccard_buffer_mean,2),
            'jaccard_buffer_total': round(jaccard_buffer_total,2)
        })
    aggregated_loss = pd.DataFrame.from_records(aggregated_loss)

    # attach the shortest results at end
    shortest_aggregated_loss = shortest_aggregated_dataframe()
    
    # merge the two
    shortest_aggregated_loss.set_index(['subset','calibration_name','run_num'],inplace=True)
    shortest_aggregated_loss = shortest_aggregated_loss.add_prefix('shortest_')
    aggregated_loss.set_index(['subset','calibration_name','run_num'],inplace=True)

    aggregated_loss = pd.concat([aggregated_loss,shortest_aggregated_loss],ignore_index=False,axis=1).reset_index()

    return aggregated_loss

def shortest_aggregated_dataframe():
    '''
    Gets aggregated shortest path stats so they can be appended to the post calibration results
    '''
    # has all the shortest stats
    with (config['calibration_fp']/'ready_for_calibration_stats.pkl').open('rb') as fh:
        shortest = pickle.load(fh)
    
    result_fps, routing_fps, loss_fps = utils.get_directories()
    
    aggregated_loss =[]
    for result_fp in result_fps:
        with result_fp.open('rb') as fh:
            result = pickle.load(fh)
        tripids = result['trips_calibrated']
        name_params = utils.get_name_parameters(result_fp)

        subset_shortest = {key:item for key,item in shortest.items() if key in tripids}

        jaccard_exact_mean = np.array([item['shortest_jaccard_exact'] for tripid, item in subset_shortest.items()]).mean()
        jaccard_exact_total = np.array([(item['shortest_jaccard_exact_intersection'],item['shortest_jaccard_exact_union']) for tripid, item in subset_shortest.items()])
        jaccard_exact_total = jaccard_exact_total[:,0].sum() / jaccard_exact_total[:,1].sum()

        jaccard_buffer_mean = np.array([item['shortest_jaccard_buffer'] for tripid, item in subset_shortest.items()]).mean()
        jaccard_buffer_total = np.array([(item['shortest_jaccard_buffer_intersection'],item['shortest_jaccard_buffer_union']) for tripid, item in subset_shortest.items()])
        jaccard_buffer_total = jaccard_buffer_total[:,0].sum() / jaccard_buffer_total[:,1].sum()
        
        aggregated_loss.append({
            **name_params,
            'jaccard_exact_mean': round(jaccard_exact_mean,2),
            'jaccard_exact_total': round(jaccard_exact_total,2),
            'jaccard_buffer_mean': round(jaccard_buffer_mean,2),
            'jaccard_buffer_total': round(jaccard_buffer_total,2)
            })
    
    aggregated_loss = pd.DataFrame.from_records(aggregated_loss)
    
    return aggregated_loss


########################################################################################

# Disaggregate Model Results

########################################################################################


#TODO after i re-start calibration runs for these add the shortest result too
from collections import defaultdict
def post_calibration_disaggregate():
    '''
    Returns a dataframe where the rows are trips and the columns are length, detour, exact, and buffer
    Appends the shortest results to the left
    '''

    result_fps, routing_fps, loss_fps = utils.get_directories()
    
    disaggregate_loss_stats = defaultdict(dict)
    for loss_fp in tqdm(loss_fps):
        with loss_fp.open('rb') as fh:
            loss_dict = pickle.load(fh)
        
        name_params = utils.get_name_parameters(loss_fp)
        
        #remove the edges from the dict
        for tripid, item in loss_dict.items():
            #remove certain fields
            for key in list(item.keys()):
                if ('_intersection' in key) | ('_union' in key) | (key=='modeled_edges'):
                    item.pop(key)
            renamed = {(name_params['subset'],name_params['calibration_name'],name_params['run_num'],key.removeprefix('modeled_')):sub_item for key, sub_item in item.items()}
            disaggregate_loss_stats[tripid].update(renamed)

    disaggregate_loss_stats = pd.DataFrame.from_dict(disaggregate_loss_stats,orient='index')
    disaggregate_loss_stats.index.name = 'tripid'
    disaggregate_loss_stats.columns = pd.MultiIndex.from_tuples(disaggregate_loss_stats.columns, names=["subset",'calibration_name',"run","loss"])
    
    return disaggregate_loss_stats

def shortest_disaggregate():

    with (config['calibration_fp']/'ready_for_calibration_stats.pkl').open('rb') as fh:
            shortest = pickle.load(fh)

    shortest_loss_stats = dict()
    for tripid, item in shortest.items():
        for key in list(item.keys()):
            if ('_intersection' in key) | ('_union' in key) | (key=='shortest_edges') | (key=='matched_edges'):
                item.pop(key)
            # item.pop('origin_node')
            # item.pop('destination_node')
            # renamed = {(key.removeprefix('modeled_')):sub_item for key, sub_item in item.items()}
            shortest_loss_stats[tripid] = item

    shortest_disaggregate_loss_stats = pd.DataFrame.from_dict(shortest_loss_stats,orient='index')
    shortest_disaggregate_loss_stats.index.name = 'tripid'

    return shortest_disaggregate_loss_stats


########################################################################################

# Testing Models on Provided Traces

########################################################################################

def validation_workflow(list_of_validations):
    '''
    Provide a list of tuples of len 2 where the first element is the
    model name (e.g. fold_1,validation,0) and the second is a list of
    trips to test the model with
    '''

    # import network and match results
    links, turns, length_dict, geo_dict, turn_G = rustworkx_routing_funcs.import_calibration_network(config)
    with (config['calibration_fp']/'ready_for_calibration.pkl').open('rb') as fh:
        match_results = pickle.load(fh)

    # create directories
    for model_fp, tripids in tqdm(list_of_validations):
            
        # subset to provided traces
        match_results_subset = {tripid:item for tripid, item in match_results.items() if tripid in tripids}
        ods = utils.match_results_to_ods_w_year(match_results_subset)

        validation_routing(model_fp,ods,links,turns,turn_G)
        validation_loss(model_fp,match_results_subset,length_dict,geo_dict)


def validation_routing(model_fp,ods,links,turns,turn_G):
    '''
    Run this with a calibration result to calculate the shortest routes for the inputted tripids

    Intended for validation, results are outputted to the validation directory
    '''

    model_name = model_fp.stem
    routing_fp = config['calibration_fp'] / f'validation/routing/{model_fp.stem}.pkl'

    # import model
    if model_fp.exists() == False:
        print(model_fp)
        print(model_name,'does not exist')
        return
    if routing_fp.exists():
        print(model_name,'already has routing results')
        return

    with model_fp.open('rb') as fh:
        result = pickle.load(fh)
    betas = [x['beta'] for x in result['betas_tup']] # get betas

    # for backwards compatibility
    # these only just now got added to the result dictionary 
    if result.get('link_impedance_function') is None:
        result['link_impedance_function'] = impedance_functions.link_impedance_function
    if result.get('turn_impedance_function') is None:
        result['turn_impedance_function'] = impedance_functions.turn_impedance_function
    if result.get('base_impedance_col') is None:
        result['base_impedance_col'] = 'travel_time_min'
    if result.get('base_link_col') is None:
        result['base_link_col'] = None

    #PROBLEM, need the impedance function used
    modeled_dict = stochastic_optimization.impedance_routing(
        betas,
        result['betas_tup'],
        result['link_impedance_function'],
        result['base_impedance_col'],
        result['base_link_col'],       
        result['turn_impedance_function'],
        links,
        turns,
        turn_G,
        ods,
        result['set_to_zero'],
        result['set_to_inf']
        )
    if modeled_dict is None:
        print('\n',model_fp.stem,'has negative link costs')
    else:
        if (config['calibration_fp'] / 'validation/routing').exists() == False:
            (config['calibration_fp'] / 'validation/routing').mkdir(parents=True)
        with routing_fp.open('wb') as fh:
            pickle.dump(modeled_dict,fh)

def validation_loss(model_fp,match_results_subset,length_dict,geo_dict):
    '''
    Calculate loss values and metrics for the modeled routes (for all calibration runs)
    '''
    
    model_name = model_fp.stem
    routing_fp = config['calibration_fp'] / f'validation/routing/{model_name}.pkl'

    # import model
    if model_fp.exists() == False:
        print('model does not exist')
        return
    if routing_fp.exists() == False:
        print('missing routing data for',model_name)

    # # skip if it's already present
    # if rewrite:    
    #     if (config['calibration_fp'] / 'loss' / f"{loss_stem}.pkl").exists():
    #         continue

    # load the routing result (includes the impedance path taken for each trip)
    with (config['calibration_fp'] / f'validation/routing/{model_name}.pkl').open('rb') as fh:
        results_dict = pickle.load(fh)
    
    loss_dict = {}
    # we could try storing these
    for tripid, item in match_results_subset.items(): # make sure you import this
        skip = False            
        
        chosen = item['matched_edges'].values
        shortest = item['shortest_edges'].values
        
        # NOTE one limitation of this is that if the origin node and destination node change at any point then
        # than this step will break
        od = (item['origin_node'],item['destination_node'])
        modeled = results_dict.get(od)
        if modeled is None:
            # print(f"Missing shortest paths in calibration {loss_stem}")
            skip = True
            break
        modeled = modeled['edge_list']
    
        #compute the loss values (store the intermediates too so total vs mean methods can be compared)
        modeled_jaccard_exact_intersection, modeled_jaccard_exact_union =  loss_functions.jaccard_exact(chosen,modeled,length_dict)
        modeled_jaccard_exact = modeled_jaccard_exact_intersection / modeled_jaccard_exact_union
        modeled_jaccard_buffer_intersection, modeled_jaccard_buffer_union =  loss_functions.jaccard_buffer(chosen,modeled,geo_dict)
        modeled_jaccard_buffer = modeled_jaccard_buffer_intersection / modeled_jaccard_buffer_union

        loss_dict[tripid] = {
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

        if (config['calibration_fp'] / 'validation/loss').exists() == False:
            (config['calibration_fp'] / 'validation/loss').mkdir(parents=True)

        with (config['calibration_fp'] / f"validation/loss/{model_name}.pkl").open('wb') as fh:
            pickle.dump(loss_dict,fh)

def testing_aggregated_loss_dataframe():
    '''
    Calculates aggregated loss stats for testing data.

    Every row is a calibration result. If user/subset based, then every row is a calibration result for a user/subset.
    
    '''
    
    result_fps, routing_fps, loss_fps = utils.get_directories(testing=True)

    # disaggregated
    aggregated_loss = []
    for loss_fp in loss_fps:
        with loss_fp.open('rb') as fh:
            loss_dict = pickle.load(fh)

        jaccard_exact_mean = np.array([item['modeled_jaccard_exact'] for tripid, item in loss_dict.items()]).mean()
        jaccard_exact_total = np.array([(item['modeled_jaccard_exact_intersection'],item['modeled_jaccard_exact_union']) for tripid, item in loss_dict.items()])
        jaccard_exact_total = jaccard_exact_total[:,0].sum() / jaccard_exact_total[:,1].sum()

        jaccard_buffer_mean = np.array([item['modeled_jaccard_buffer'] for tripid, item in loss_dict.items()]).mean()
        jaccard_buffer_total = np.array([(item['modeled_jaccard_buffer_intersection'],item['modeled_jaccard_buffer_union']) for tripid, item in loss_dict.items()])
        jaccard_buffer_total = jaccard_buffer_total[:,0].sum() / jaccard_buffer_total[:,1].sum()

        name_params = utils.get_name_parameters(loss_fp)

        aggregated_loss.append({
            **name_params, # contains the userid, calibration name and run number
            'jaccard_exact_mean': round(jaccard_exact_mean,2),
            'jaccard_exact_total': round(jaccard_exact_total,2),
            'jaccard_buffer_mean': round(jaccard_buffer_mean,2),
            'jaccard_buffer_total': round(jaccard_buffer_total,2)
        })
    aggregated_loss = pd.DataFrame.from_records(aggregated_loss)

    # attach the shortest results at end
    shortest_aggregated_loss = shortest_aggregated_dataframe()
    
    # merge the two
    shortest_aggregated_loss.set_index(['subset','calibration_name','run_num'],inplace=True)
    shortest_aggregated_loss = shortest_aggregated_loss.add_prefix('shortest_')
    aggregated_loss.set_index(['subset','calibration_name','run_num'],inplace=True)

    aggregated_loss = pd.concat([aggregated_loss,shortest_aggregated_loss],ignore_index=False,axis=1).reset_index()

    return aggregated_loss.dropna()


#### Utility functions ####


# def file_check_post_routing(user=False):
#     '''
#     Coordinates between the results and post calibration directories to make sure that both files exist
#     '''
#     if user:
#         calibration_results_dir = config['calibration_fp']/"user_calibration_results"
#         post_calibration_routing_dir = config['calibration_fp']/"user_post_calibration_routing"
#     else:
#         calibration_results_dir = config['calibration_fp']/"calibration_results"
#         post_calibration_routing_dir = config['calibration_fp']/"post_calibration_routing"

#     # get the intersection of both
#     calibration_results = [x.stem for x in calibration_results_dir.glob('*.pkl')]
#     post_calibration_routing = [x.stem for x in post_calibration_routing_dir.glob('*.pkl')]
    
#     no_routing = list(set(calibration_results) - set(post_calibration_routing))
#     no_routing_fps = [x for x in calibration_results_dir.glob('*.pkl') if x.stem in no_routing]

#     return no_routing_fps



# import re
# def get_calibration_name_parameters(calibration_name,user=False):
#     '''
#     Extracts the user number and calbration name from the file name
#     '''
#     output = {}

#     if user:
#         output['userid'] = calibration_name.split('_')[0]
#         calibration_name = calibration_name.split('_',maxsplit=1)[1]

#     # get run number
#     pattern = r'\((\d+)\)'
#     run_number = re.findall(pattern,calibration_name)

#     if len(run_number) == 0:
#         output['run_number'] = 0
#         output['calibration'] = calibration_name.strip()
#     else:
#         output['run_number'] = run_number[0]
#         output['calibration'] = calibration_name.split('(')[0].strip()
    
#     return output