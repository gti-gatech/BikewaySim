import pickle
from pathlib import Path
from importlib import reload

from bikewaysim.paths import config
from bikewaysim.impedance_calibration import stochastic_optimization, loss_functions
from bikewaysim.routing import rustworkx_routing_funcs

with (config['calibration_fp']/'subsets.pkl').open('rb') as fh:
    subsets = pickle.load(fh)
# subsets = [x for x in subsets if x[0] == 'debug']
subsets = [x for x in subsets if x[0] == 'random']

kwargs = {
    'calibration_name': 'debug',    
    'betas_tup': (
        {'col':'2lpd','type':'link','range':[0,3]},
        {'col':'3+lpd','type':'link','range':[0,3]},
        {'col':'(30,inf) mph','type':'link','range':[0,3]},
        # {'col':'[4k,10k) aadt','type':'link','range':[0,3]},
        # {'col':'[10k,inf) aadt','type':'link','range':[0,3]},
        {'col':'[4,6) grade','type':'link','range':[0,3]},
        {'col':'[6,inf) grade','type':'link','range':[0,3]},
        {'col':'bike lane','type':'link','range':[-1,3]},
        {'col':'multi use path and cycletrack','type':'link','range':[-1,3]},
        {'col':'unsig_crossing','type':'turn','range':[0,5]},
        {'col':'left_turn','type':'turn','range':[0,5]}
    ),
    'set_to_zero': ['bike lane','cycletrack','multi use path'],
    'set_to_inf': ['not_street'],
    'objective_function': loss_functions.jaccard_buffer_total,
    'stochastic_optimization_settings': {'method':'pso','options':{'maxiter':75,'popsize':25,'xtol':0.05,'ftol':-0.75,'return_all':True}},
    'print_results': True,
    'subset': subsets[0]
}

'''
Default xtol and ftol are 1e-8

ftol should be set between -0.45 and -1 (until we start getting fitness values above -0.45)
still figuring out a good xtol value but we probably only care about coefficients to the 2nd decimal place
'''


if __name__ == '__main__':
    stochastic_optimization.full_impedance_calibration(**kwargs)