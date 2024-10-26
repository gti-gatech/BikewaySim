import pickle
import concurrent.futures
import itertools
import time

from bikewaysim.paths import config
from bikewaysim.impedance_calibration import stochastic_optimization
from bikewaysim.general_utils import print_elapsed_time

#NOTE relative import, do not move this file from the directory
from step_1_calibration_experiments import all_calibrations

'''
Runs the validation first
'''


if __name__ == '__main__':
    
    start_time = time.time()
    
    # NUM_RUNS = 1 # Number of times to run each calibration
    MAX_WORKERS = 14 # For my machine this takes about ~70% CPU and ~80% Memory
    
    print('Found these calibration settings:')
    print([x['calibration_name'] for x in all_calibrations])
    print('Running k-fold and bootstrap runs')

    # NOTE comment out lines 41-53 to run calibration on all matched traces
    # Import subsets
    # with (config['calibration_fp']/'validation/training_folds.pkl').open('rb') as fh:
    #     training_folds = pickle.load(fh)

    with (config['calibration_fp']/'validation/bootstrap_samples.pkl').open('rb') as fh:
        bootstrap_samples = pickle.load(fh)

    # # select which subsets to calibrate
    # subset_ids = ['random'] # or uuserid in string format
    # subsets = [x for x in subsets if x[0] in subset_ids]
    # print([x[0] for x in subsets])
    existing_runs = list((config['calibration_fp'] / 'results').glob('bootsample_*.pkl'))
    existing_runs = [x.stem.split(',')[0] for x in existing_runs]
    bootstrap_samples = [x for x in bootstrap_samples if x[0] not in existing_runs]
    print(len(bootstrap_samples),'bootstrap runs remaining')

    all_calibrations = [x for x in all_calibrations if x['calibration_name'] == 'validation']

    # Add users to the calibration list
    # kfold_calibrations = [{**calibration_dict,**{'subset':subset}} for subset, calibration_dict in itertools.product(training_folds,all_calibrations)]
    bootstrap_calibrations = [{**calibration_dict,**{'subset':subset}} for subset, calibration_dict in itertools.product(bootstrap_samples,all_calibrations)]

    # tasks = kfold_calibrations + bootstrap_calibrations
    tasks = bootstrap_calibrations
    
    # END COMMENT BLOCK

    # what's not ideal is that the calibration runs are not learing from past results 
    '''
    Look at the number of iterations vs overlap to see if it's pretty clear on whether were trapped at a local minimum or not
    '''

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for idx, result in enumerate(executor.map(stochastic_optimization.run_validation_and_bootstrap, tasks)):
                elapsed_time = print_elapsed_time(time.time()-start_time)    
                if result[1]:
                    print('Completed',result[0],'| Remaining:',len(tasks)-idx+1,'| Elapsed Time:',elapsed_time)
                else:
                    print('Exceeded iterations',result[0],'| Remaining:',len(tasks)-idx+1,'| Elapsed Time:',elapsed_time)
    except KeyboardInterrupt:
        print('\nExiting...')

    end_time = time.time()
    print('Took',round((end_time-start_time)/60**2,1),'hours')