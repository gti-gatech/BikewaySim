from bikewaysim.impedance_calibration import stochastic_optimization

# determine variables, impedance type, and search range
all_calibrations = []

# write this in another py file and import it maybe to clean things up?
#keys are the names of the models
all_calibrations.append({
    'calibration_name': 'condensed',
    'betas_tup': (
        {'col':'2lpd','type':'link','range':[0,3]},
        {'col':'3+lpd','type':'link','range':[0,3]},
        {'col':'(30,inf) mph','type':'link','range':[0,3]},
        {'col':'[4k,10k) aadt','type':'link','range':[0,3]},
        {'col':'[10k,inf) aadt','type':'link','range':[0,3]},
        {'col':'[4,6) grade','type':'link','range':[0,3]},
        {'col':'[6,inf) grade','type':'link','range':[0,3]},
        {'col':'bike lane','type':'link','range':[-1,0]},
        {'col':'cycletrack','type':'link','range':[-1,0]},
        {'col':'multi use path','type':'link','range':[-1,0]},
        {'col':'unsig_crossing','type':'turn','range':[0,2]}
    ),
    'set_to_zero': ['bike lane','cycletrack','multi use path'],
    'set_to_inf': ['not_street'],
    'objective_function': stochastic_optimization.jaccard_buffer_mean,
    'batching': False,
    'stochastic_optimization_settings': {'method':'pso','options':{'maxiter':2,'popsize':2}},
})

#keys are the names of the models
all_calibrations.append({
    'calibration_name': 'condensed2',
    'betas_tup': (
        {'col':'2lpd','type':'link','range':[0,3]},
        {'col':'3+lpd','type':'link','range':[0,3]},
        {'col':'(30,inf) mph','type':'link','range':[0,3]},
        {'col':'[4k,10k) aadt','type':'link','range':[0,3]},
        {'col':'[10k,inf) aadt','type':'link','range':[0,3]},
        {'col':'[4,6) grade','type':'link','range':[0,3]},
        {'col':'[6,inf) grade','type':'link','range':[0,3]},
        {'col':'bike lane','type':'link','range':[-1,0]},
        {'col':'cycletrack','type':'link','range':[-1,0]},
        {'col':'multi use path','type':'link','range':[-1,0]},
        {'col':'unsig_crossing','type':'turn','range':[0,2]}
    ),
    'set_to_zero': ['bike lane','cycletrack','multi use path'],
    'set_to_inf': ['not_street'],
    'objective_function': stochastic_optimization.jaccard_buffer_mean,
    'batching': False,
    'stochastic_optimization_settings': {'method':'pso','options':{'maxiter':2,'popsize':2}},
})