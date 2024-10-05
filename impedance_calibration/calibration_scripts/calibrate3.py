from pathlib import Path
import time
import pandas as pd
import pickle
from stochopy.optimize import minimize
import datetime

from bikewaysim.paths import config
from bikewaysim.impedance_calibration import stochastic_optimization


if __name__ == '__main__':

    # determine the ouput name of the calibration outputs
    calibration_name = 'calibration3_changed'

    # determine variables, impedance type, and search range
    betas_tup = (
        {'col':'2lpd','type':'link','range':[0,3]},
        {'col':'3+lpd','type':'link','range':[0,3]},
        {'col':'(30,40] mph','type':'link','range':[0,3]},
        {'col':'(40,inf) mph','type':'link','range':[0,3]},
        {'col':'[4k,10k) aadt','type':'link','range':[0,3]},
        {'col':'[10k,inf) aadt','type':'link','range':[0,3]},
        {'col':'[4,6) grade','type':'link','range':[0,3]},
        {'col':'[6,inf) grade','type':'link','range':[0,3]},
        {'col':'bike lane','type':'link','range':[-1,0]},
        {'col':'multi use path_original','type':'link','range':[-1,1]},
        # {'col':'cycletrack','type':'link','range':[-1,0]},
        # {'col':'multi use path','type':'link','range':[-1,0]},
        # {'col':'unsig_major_road_crossing','type':'turn','range':[0,2]}
    )

    # determine the objective function to use and other settings
    objective_function = stochastic_optimization.buffer_overlap
    batching = True
    stochastic_optimization_settings = {
        'method':'pso',
        'options': {'maxiter':1,'popsize':2}
    }

    links, turns, length_dict, geo_dict, turn_G = stochastic_optimization.import_calibration_network(config)

    with (config['calibration_fp']/'ready_for_calibration.pkl').open('rb') as fh:
        full_set = pickle.load(fh)

    args = (
        [], # empty list for storing past calibration results
        betas_tup, # tuple containing the impedance spec
        stochastic_optimization.match_results_to_ods(full_set), # list of OD network node pairs needed for shortest path routing
        full_set, # dict containing the origin/dest node and map matched edges
        stochastic_optimization.link_impedance_function, # link impedance function to use
        "travel_time_min", # column with the base the base impedance in travel time or distance
        stochastic_optimization.turn_impedance_function, # turn impedance function to use
        links,turns,turn_G, # network parts
        objective_function, # loss function to use
        {'length_dict':length_dict,'geo_dict':geo_dict},#,'trace_dict':traces}, # keyword arguments for loss function
        False, #whether to print the results of each iteration
        True, #whether to store calibration results
        batching # whether to batch results to help speed up computation time, if yes input the number to batch with
    )
    calibration_result = stochastic_optimization.full_impedance_calibration(betas_tup,args,stochastic_optimization_settings,full_set,calibration_name)

    # #export but don't overwrite
    export_fp = config['calibration_fp'] / f'calibration_results/{calibration_name}.pkl'
    with stochastic_optimization.uniquify(export_fp).open('wb') as fh:
            pickle.dump(calibration_result,fh)