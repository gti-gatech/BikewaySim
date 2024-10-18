import pickle
from pathlib import Path
from importlib import reload

from bikewaysim.paths import config
from bikewaysim.impedance_calibration import stochastic_optimization, loss_functions
from bikewaysim.routing import rustworkx_routing_funcs

kwargs = {
    'calibration_name': 'testing',    
    'betas_tup': (
        {'col':'2lpd','type':'link','range':[0,3]},
        # {'col':'3+lpd','type':'link','range':[-1,3]},
        # {'col':'(30,inf) mph','type':'link','range':[-1,3]},
        # {'col':'[4k,10k) aadt','type':'link','range':[-1,3]},
        # {'col':'[10k,inf) aadt','type':'link','range':[-1,3]},
        # {'col':'[4,6) grade','type':'link','range':[-1,3]},
        # {'col':'[6,inf) grade','type':'link','range':[-1,3]},
        # {'col':'bike lane','type':'link','range':[-1,3]},
        # {'col':'multi use path and cycletrack','type':'link','range':[-1,3]},
        # {'col':'unsig_crossing','type':'turn','range':[0,2]}
    ),
    'set_to_zero': ['bike lane','cycletrack','multi use path'],
    'set_to_inf': ['not_street'],
    'objective_function': loss_functions.jaccard_buffer_mean,
    'stochastic_optimization_settings': {'method':'pso','options':{'maxiter':2,'popsize':4}},
    'print_results': True,
}

#BUG if last results resulted in negative edge weights, then the post caliiration will failreload(stochastic_optimization)

if __name__ == '__main__':
    stochastic_optimization.full_impedance_calibration(**kwargs)