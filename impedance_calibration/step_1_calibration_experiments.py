from bikewaysim.impedance_calibration import stochastic_optimization

'''
This script is for feeding in different calibration settings.

All calibrations is list of dicts. Each calibration result should have a name
but if you're just tweaking certain settings then it may make sense to have multiple
calibration attempts with the same name (e.g., adjusting the population size)

'''

# use this space to create variables that are going to be commonly used
full_model = (
    {'col':'2lpd','type':'link','range':[0,3]},
    {'col':'3+lpd','type':'link','range':[0,3]},
    {'col':'(30,40] mph','type':'link','range':[0,3]},
    {'col':'(40,inf) mph','type':'link','range':[0,3]},
    {'col':'[4k,10k) aadt','type':'link','range':[0,3]},
    {'col':'[10k,inf) aadt','type':'link','range':[0,3]},
    {'col':'[4,6) grade','type':'link','range':[0,3]},
    {'col':'[6,inf) grade','type':'link','range':[0,3]},
    {'col':'bike lane','type':'link','range':[-1,0]},
    {'col':'cycletrack','type':'link','range':[-1,0]},
    {'col':'multi use path','type':'link','range':[-1,0]},
    {'col':'unsig_crossing','type':'turn','range':[0,2]},
)
std_set_to_zero = ['bike lane','cycletrack','multi use path']
std_set_to_inf = ['not_street']

break_stuff = ({'col':'bike lane','type':'link','range':[-20,0]},)

# determine variables, impedance type, and search range
all_calibrations = [
    # {
    #     'calibration_name': 'condensed',
    #     'betas_tup': (
    #         {'col':'2lpd','type':'link','range':[0,3]},
    #         {'col':'3+lpd','type':'link','range':[0,3]},
    #         {'col':'(30,inf) mph','type':'link','range':[0,3]},
    #         {'col':'[4k,10k) aadt','type':'link','range':[0,3]},
    #         {'col':'[10k,inf) aadt','type':'link','range':[0,3]},
    #         {'col':'[4,6) grade','type':'link','range':[0,3]},
    #         {'col':'[6,inf) grade','type':'link','range':[0,3]},
    #         {'col':'bike lane','type':'link','range':[-1,0]},
    #         {'col':'cycletrack','type':'link','range':[-1,0]},
    #         {'col':'multi use path','type':'link','range':[-1,0]},
    #         {'col':'unsig_crossing','type':'turn','range':[0,2]}
    #     ),
    #     'set_to_zero': std_set_to_zero,
    #     'set_to_inf': std_set_to_inf,
    #     'objective_function': stochastic_optimization.jaccard_buffer_mean
    # },
    # {
    #     'calibration_name': 'gdot',
    #     'betas_tup': (
    #         {'col':'multi use path report','type':'link','range':[-1,1]},
    #         {'col':'bike lane report','type':'link','range':[-1,1]},
    #         {'col':'lanes report','type':'link','range':[0,2]},
    #         {'col':'above_4 report','type':'link','range':[0,2]},
    #         {'col':'unsig_crossing','type':'turn','range':[0,2]}
    #     ),
    #     'objective_function': stochastic_optimization.jaccard_exact_mean
    # },
    # {
    #     'calibration_name': 'jaccard_buffer_mean_no_bike',
    #     'betas_tup': full_model,
    #     'objective_function': stochastic_optimization.jaccard_buffer_mean
    # },
    {
        'calibration_name': 'jaccard_buffer_mean',
        'betas_tup': full_model,
        'set_to_zero': std_set_to_zero,
        'set_to_inf': std_set_to_inf,
        'objective_function': stochastic_optimization.jaccard_buffer_mean,
        'stochastic_optimization_settings': {'method':'pso','options':{'maxiter':500,'popsize':25}},
    },
    {
        'calibration_name': 'jaccard_buffer_mean',
        'betas_tup': full_model,
        'set_to_zero': std_set_to_zero,
        'set_to_inf': std_set_to_inf,
        'objective_function': stochastic_optimization.jaccard_buffer_mean,
        'stochastic_optimization_settings': {'method':'pso','options':{'maxiter':500,'popsize':35}},
    },
    {
        'calibration_name': 'jaccard_buffer_mean',
        'betas_tup': full_model,
        'set_to_zero': std_set_to_zero,
        'set_to_inf': std_set_to_inf,
        'objective_function': stochastic_optimization.jaccard_buffer_mean,
        'stochastic_optimization_settings': {'method':'pso','options':{'maxiter':500,'popsize':50}},
    },
    # {
    #     'calibration_name': 'jaccard_buffer_total_no_bike',
    #     'betas_tup': full_model,
    #     'objective_function': stochastic_optimization.jaccard_buffer_total
    # },
    # {
    #     'calibration_name': 'jaccard_buffer_total',
    #     'betas_tup': full_model,
    #     'set_to_zero': std_set_to_zero,
    #     'set_to_inf': std_set_to_inf,
    #     'objective_function': stochastic_optimization.jaccard_buffer_total
    # },
    # {
    #     'calibration_name': 'jaccard_exact_mean_no_bike',
    #     'betas_tup': full_model,
    #     'objective_function': stochastic_optimization.jaccard_exact_mean
    # },
    # {
    #     'calibration_name': 'jaccard_exact_mean',
    #     'betas_tup': full_model,
    #     'set_to_zero': std_set_to_zero,
    #     'set_to_inf': std_set_to_inf,
    #     'objective_function': stochastic_optimization.jaccard_exact_mean
    # },
    # {
    #     'calibration_name': 'jaccard_exact_total_no_bike',
    #     'betas_tup': full_model,
    #     'objective_function': stochastic_optimization.jaccard_exact_total
    # },
    # {
    #     'calibration_name': 'jaccard_excact_total',
    #     'betas_tup': full_model,
    #     'set_to_zero': std_set_to_zero,
    #     'set_to_inf': std_set_to_inf,
    #     'objective_function': stochastic_optimization.jaccard_exact_total
    # }
    # {
    #     'calibration_name': 'break stuff',
    #     'betas_tup': break_stuff,
    #     'objective_function': stochastic_optimization.jaccard_buffer_mean,
    #     'stochastic_optimization_settings': {'method':'pso','options':{'maxiter':4,'popsize':2}},
    #     'print_results': True,
    #     'force_save': True
    # }
]