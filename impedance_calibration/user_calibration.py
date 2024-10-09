from pathlib import Path
import pickle
import concurrent.futures
from tqdm.contrib.concurrent import process_map
import itertools

from bikewaysim.paths import config
from bikewaysim.impedance_calibration import stochastic_optimization
from calibration_experiments import all_calibrations

NUM_RUNS = 1 # Number of times to run each calibration

# Helper function to unpack the dictionary and call example_function
def run_calibration(task):
    task_dict, run_num = task
    task_name = f"{task_dict['calibration_name']} ({run_num+1}/{NUM_RUNS})"
    success = stochastic_optimization.full_impedance_calibration(**task_dict)
    return task_name, success

if __name__ == '__main__':
    print([x['calibration_name'] for x in all_calibrations])
    
    # Settings
    
    # MAX_WORKERS = 10

    # Import users
    with (config['calibration_fp']/'ready_for_calibration_users.pkl').open('rb') as fh:
        ready_for_calibration_users = pickle.load(fh)

    # Add users to the calibration list
    new_all_calibrations = []
    all_calibrations = [{**calibration_dict,**{'user':user}} for user, calibration_dict in itertools.product(ready_for_calibration_users,all_calibrations)]
    
    # Create a list of (script, run_num) pairs for NUM_RUNS
    tasks = [(calibration_dict, run_num) for calibration_dict in all_calibrations for run_num in range(NUM_RUNS)]

    # process_map(run_calibration, tasks)

    try:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for result in executor.map(run_calibration, tasks):
                print('complete')
    except KeyboardInterrupt:
        print('\nExiting...')