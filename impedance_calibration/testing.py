from pathlib import Path
import pickle
import concurrent.futures
from tqdm import tqdm

from bikewaysim.paths import config
from bikewaysim.impedance_calibration import stochastic_optimization
from calibration_experiments import all_calibrations

# Helper function to unpack the dictionary and call example_function
def run_calibration(task):
    task_dict, run_num = task
    task_name = f"{task_dict['calibration_name']} ({run_num+1}/{NUM_RUNS})"

    print('Starting:',task_name)
    success = stochastic_optimization.full_impedance_calibration(**task_dict)
    if success:
        print('Completed:',task_name)
    else:
        print('Failed:',task_name)

    return task_name, success

# use this to add a user arguement
# def add_global_impedance_variables(all_calibrations,full_set):
#     [x.update({'full_set':full_set}) for x in all_calibrations]
#     return
# add_global_impedance_variables(all_calibrations) #updates all_calibrations in place to add full_set

if __name__ == '__main__':
    print([x['calibration_name'] for x in all_calibrations])
    
    # Settings
    NUM_RUNS = 10 # Number of times to run each calibration

    # Create a list of (script, run_num) pairs for NUM_RUNS
    tasks = [(calibration_dict, run_num) for calibration_dict in all_calibrations for run_num in range(NUM_RUNS)]

    try:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(run_calibration, tasks))
    except KeyboardInterrupt:
        print('\nExiting...')