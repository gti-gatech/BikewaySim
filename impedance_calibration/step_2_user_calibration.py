import pickle
import concurrent.futures
import itertools

from bikewaysim.paths import config
from bikewaysim.impedance_calibration import stochastic_optimization
from impedance_calibration.step_1_calibration_experiments import all_calibrations

if __name__ == '__main__':
    print([x['calibration_name'] for x in all_calibrations])
    
    NUM_RUNS = 3 # Number of times to run each calibration

    # Import users
    with (config['calibration_fp']/'ready_for_calibration_users.pkl').open('rb') as fh:
        ready_for_calibration_users = pickle.load(fh)

    # Add users to the calibration list
    new_all_calibrations = []
    all_calibrations = [{**calibration_dict,**{'user':user}} for user, calibration_dict in itertools.product(ready_for_calibration_users,all_calibrations)]
    
    # Create a list of (script, run_num) pairs for NUM_RUNS
    tasks = [(calibration_dict, run_num, NUM_RUNS) for calibration_dict in all_calibrations for run_num in range(NUM_RUNS)]

    try:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for result in executor.map(stochastic_optimization.run_calibration, tasks):
                if result[1]:
                    print('Completed',result[0])
                else:
                    print('Failed to complete',result[0])
    except KeyboardInterrupt:
        print('\nExiting...')