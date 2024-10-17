import pickle
import concurrent.futures
import itertools
import time

from bikewaysim.paths import config
from bikewaysim.impedance_calibration import stochastic_optimization
from step_1_calibration_experiments import all_calibrations

import time

def print_elapsed_time(seconds):
    # Round the total seconds at the start
    seconds = round(seconds)
    
    # Calculate the elapsed days, hours, and minutes
    days = seconds // 86400  # 1 day = 86400 seconds
    hours = (seconds % 86400) // 3600  # 1 hour = 3600 seconds
    minutes = (seconds % 3600) // 60

    # Build the time string
    if days > 0:
        elapsed_time = f"{days:02} days {hours:02} hours {minutes:02} minutes"
    else:
        elapsed_time = f"{hours:02} hours {minutes:02} minutes"

    # Return the formatted elapsed time
    return elapsed_time

if __name__ == '__main__':
    
    start_time = time.time()
    
    NUM_RUNS = 4#100 # Number of times to run each calibration
    MAX_WORKERS = 12 # For my machine this takes about ~70% CPU and ~80% Memory
    
    print('Using these calibration settings:')
    print([x['calibration_name'] for x in all_calibrations])
    print('Running each calibration',NUM_RUNS,'times')

    # NOTE COMMENT OUT LINE 19-24 TO DO ALL TRIPS CALIBRATION
    # Import users
    with (config['calibration_fp']/'ready_for_calibration_users.pkl').open('rb') as fh:
        ready_for_calibration_users = pickle.load(fh)

    # # if you want to only do a few users
    subset_users = ['fearless']#21]
    ready_for_calibration_users = [x for x in ready_for_calibration_users if x[0] in subset_users]
    print([x[0] for x in ready_for_calibration_users])

    # Add users to the calibration list
    all_calibrations = [{**calibration_dict,**{'user':user}} for user, calibration_dict in itertools.product(ready_for_calibration_users,all_calibrations)]
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
                    print('Completed',result[0],'Remaining:',len(tasks)-idx+1,'Elapsed Time',elapsed_time)
                else:
                    print('Failed to complete',result[0],'Remaining:',len(tasks)-idx+1,'Elapsed Time',elapsed_time)
    except KeyboardInterrupt:
        print('\nExiting...')

    end_time = time.time()
    print('Took',round((end_time-start_time)/60**2,1),'hours')