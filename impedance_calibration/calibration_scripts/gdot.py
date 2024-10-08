import pickle
from pathlib import Path

from bikewaysim.paths import config
from bikewaysim.impedance_calibration import stochastic_optimization

# Get the script name (with the full path)
calibration_name = Path(__file__).stem

if __name__ == '__main__':
    # determine variables, impedance type, and search range
    betas_tup = (
        {'col':'multi use path report','type':'link','range':[-1,1]},
        {'col':'bike lane report','type':'link','range':[-1,1]},
        {'col':'lanes report','type':'link','range':[0,2]},
        {'col':'above_4 report','type':'link','range':[0,2]},
        {'col':'unsig_crossing','type':'turn','range':[0,2]}
    )
    set_to_zero = []#['bike lane']
    set_to_inf = []#['multi use path_original']
    # determine the objective function to use and other settings
    objective_function = stochastic_optimization.jaccard_exact_mean
    batching = False
    stochastic_optimization_settings = {
        'method':'pso',
        'options': {'maxiter':100,'popsize':3}
    }
    print_results = False

    with (config['calibration_fp']/'ready_for_calibration.pkl').open('rb') as fh:
        full_set = pickle.load(fh)

    stochastic_optimization.full_impedance_calibration(
        betas_tup,
        set_to_zero,
        set_to_inf,
        objective_function,
        batching,
        stochastic_optimization_settings,
        full_set,
        print_results,
        calibration_name
        )