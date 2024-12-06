import pickle
import concurrent.futures
import itertools
import time

from bikewaysim.paths import config
from bikewaysim.impedance_calibration import stochastic_optimization, loss_functions
from bikewaysim.general_utils import print_elapsed_time

if __name__ == '__main__':
    
    start_time = time.time()
    
    NUM_RUNS = 2 # Number of times to run each calibration
    MAX_WORKERS = 2 # For my machine this takes about ~70% CPU and ~80% Memory
    
    print('Found these calibration settings:')
    # print([x['calibration_name'] for x in all_calibrations])
    print('Running each calibration',NUM_RUNS,'times')

    # NOTE comment out lines 41-53 to run calibration on all matched traces
    # Import subsets
    with (config['calibration_fp']/'subsets.pkl').open('rb') as fh:
        subsets = pickle.load(fh)

    # select which subsets to calibrate
    subset_ids = ['random'] # or userid in string format
    subsets = [x for x in subsets if x[0] in subset_ids]
    print([x[0] for x in subsets])

    # model settings placeholder
    all_calibrations = [ {
        'calibration_name': 'pace',    
        'betas_tup': (
            {'col':'2lpd','type':'link','range':[0,3]},
            {'col':'3+lpd','type':'link','range':[0,3]},
            {'col':'(30,inf) mph','type':'link','range':[0,3]},
            {'col':'[4,6) grade','type':'link','range':[0,3]},
            {'col':'[6,inf) grade','type':'link','range':[0,3]},
            {'col':'bike lane','type':'link','range':[-1,3]},
            {'col':'multi use path and cycletrack','type':'link','range':[-1,3]},
            {'col':'unsig_crossing','type':'turn','range':[0,5]},
            {'col':'left_turn','type':'turn','range':[0,5]},
            {'col':'right_turn','type':'turn','range':[0,5]}
        ),
        'set_to_zero': ['bike lane','cycletrack','multi use path'],
        'set_to_inf': ['not_street'],
        'objective_function': loss_functions.jaccard_buffer_mean,
        'stochastic_optimization_settings': {'method':'pso','options':{'maxiter':3,'popsize':4,'xtol':0.05,'ftol':-0.75,'return_all':True}},
        'print_results': True,
        'subset': subsets[0]
    } ]

    # Add users to the calibration list
    all_calibrations = [{**calibration_dict,**{'subset':subset}} for subset, calibration_dict in itertools.product(subsets,all_calibrations)]
    # END COMMENT BLOCK

    # Create a list of (script, run_num) pairs for NUM_RUNS
    tasks = [(calibration_dict, run_num, NUM_RUNS) for calibration_dict in all_calibrations for run_num in range(NUM_RUNS)]

    # Sort tasks by the run number, so that one full run completes first
    tasks = sorted(tasks,key=lambda x: x[1])

    # what's not ideal is that the calibration runs are not learing from past results 
    '''
    Look at the number of iterations vs overlap to see if it's pretty clear on whether were trapped at a local minimum or not
    '''
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for idx, result in enumerate(executor.map(stochastic_optimization.run_calibration, tasks)):
                elapsed_time = print_elapsed_time(time.time()-start_time)    
                if result[1]:
                    print('Completed',result[0],'| Remaining:',len(tasks)-idx+1,'| Elapsed Time',elapsed_time)
                else:
                    print('Exceeded iterations',result[0],'| Remaining:',len(tasks)-idx+1,'| Elapsed Time',elapsed_time)
    except KeyboardInterrupt:
        print('\nExiting...')

    end_time = time.time()
    print('Took',round((end_time-start_time)/60**2,1),'hours')