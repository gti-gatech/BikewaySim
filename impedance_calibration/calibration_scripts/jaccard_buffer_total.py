import pickle
from pathlib import Path

from bikewaysim.paths import config
from bikewaysim.impedance_calibration import stochastic_optimization

# Get the script name (with the full path)
calibration_name = Path(__file__).stem

if __name__ == '__main__':
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
        # {'col':'cycletrack','type':'link','range':[-1,0]},
        # {'col':'multi use path','type':'link','range':[-1,0]},
        {'col':'multi use path_original','type':'link','range':[-1,1]},
        # {'col':'unsig_major_road_crossing','type':'turn','range':[0,2]}
    )
    set_to_zero = []#['bike lane']
    set_to_inf = []#['multi use path_original']

    # determine the objective function to use and other settings
    objective_function = stochastic_optimization.jaccard_buffer_mean
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
