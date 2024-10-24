from bikewaysim.impedance_calibration import stochastic_optimization, loss_functions

'''
This script is for feeding in different calibration settings.

All calibrations is list of dicts. Each calibration result should have a name
but if you're just tweaking certain settings then it may make sense to have multiple
calibration attempts with the same name (e.g., adjusting the population size)

'''

# model that i used in the report
gdot_model = (
        {'col':'multi use path report','type':'link','range':[-1,1]},
        {'col':'bike lane report','type':'link','range':[-1,1]},
        {'col':'lanes report','type':'link','range':[0,2]},
        {'col':'above_4 report','type':'link','range':[0,2]},
        {'col':'unsig_crossing','type':'turn','range':[0,2]}
        )

# use this space to create variables that are going to be commonly used
full_model = (
    {'col':'2lpd','type':'link','range':[0,1]},
    {'col':'3+lpd','type':'link','range':[0,1]},
    {'col':'(30,inf) mph','type':'link','range':[0,1]},
    {'col':'[4k,10k) aadt','type':'link','range':[0,1]},
    {'col':'[10k,inf) aadt','type':'link','range':[0,1]},
    {'col':'[4,6) grade','type':'link','range':[0,3]},
    {'col':'[6,inf) grade','type':'link','range':[0,3]},
    {'col':'bike lane','type':'link','range':[-1,0]},
    {'col':'cycletrack','type':'link','range':[-1,0]},
    {'col':'multi use path','type':'link','range':[-1,0]},
    {'col':'unsig_crossing','type':'turn','range':[0,2]},
    )

# removes traffic volume since it didn't seem to be effective
model3 = (
        {'col':'2lpd','type':'link','range':[0,2]},
        {'col':'3+lpd','type':'link','range':[0,2]},
        {'col':'(30,inf) mph','type':'link','range':[0,2]},
        # {'col':'[4k,10k) aadt','type':'link','range':[0,2]},
        # {'col':'[10k,inf) aadt','type':'link','range':[0,2]},
        {'col':'[4,6) grade','type':'link','range':[0,3]},
        {'col':'[6,inf) grade','type':'link','range':[0,3]},
        {'col':'bike lane','type':'link','range':[-1,0]},
        {'col':'cycletrack','type':'link','range':[-1,0]},
        {'col':'multi use path','type':'link','range':[-1,0]},
        {'col':'unsig_crossing','type':'turn','range':[0,2]},
        {'col':'left_turn','type':'turn','range':[0,2]},
        {'col':'right_turn','type':'turn','range':[0,2]},
        )

std_set_to_zero = ['bike lane','cycletrack','multi use path']
std_set_to_inf = ['not_street']

break_stuff = ({'col':'bike lane','type':'link','range':[-1,5]},)

# determine variables, impedance type, and search range
# step_2_run_calibration.py will run all of the settings that are not commented out below
all_calibrations = [
    # try out the different loss functions
    {
        'calibration_name': 'validation',
        'betas_tup': model3,
        'set_to_zero': std_set_to_zero,
        'set_to_inf': std_set_to_inf,
        'objective_function': loss_functions.jaccard_buffer_mean,
        'overwrite': True,
        'stochastic_optimization_settings': {'method':'pso','options':{'maxiter':10,'popsize':25,'return_all':True,'ftol':-0.65}}
    },
    # {
    #     'calibration_name': 'no traffic',
    #     'betas_tup': model3,
    #     'set_to_zero': std_set_to_zero,
    #     'set_to_inf': std_set_to_inf,
    #     'objective_function': loss_functions.jaccard_exact_total,
    #     'stochastic_optimization_settings': {'method':'pso','options':{'maxiter':100,'popsize':25,'return_all':True,'ftol':-0.65}}
    # },
    # {
    #     'calibration_name': 'no traffic',
    #     'betas_tup': model3,
    #     'set_to_zero': std_set_to_zero,
    #     'set_to_inf': std_set_to_inf,
    #     'objective_function': loss_functions.jaccard_buffer_mean,
    #     'stochastic_optimization_settings': {'method':'pso','options':{'maxiter':100,'popsize':25,'return_all':True,'ftol':-0.65}}
    # },
    # {
    #     'calibration_name': 'no traffic',
    #     'betas_tup': model3,
    #     'set_to_zero': std_set_to_zero,
    #     'set_to_inf': std_set_to_inf,
    #     'objective_function': loss_functions.jaccard_buffer_total,
    #     'stochastic_optimization_settings': {'method':'pso','options':{'maxiter':100,'popsize':25,'return_all':True,'ftol':-0.65}}
    # },
    # {
    #     'calibration_name': 'gdot',
    #     'betas_tup': gdot_model,
    #     'objective_function': loss_functions.jaccard_exact_mean,
    #     'stochastic_optimization_settings': {'method':'pso','options':{'maxiter':100,'popsize':25,'return_all':True,'ftol':-0.65}}
    # },
    
    
    
    
    # {
    #     'calibration_name': 'cycletrack',
    #     'betas_tup':  (
    #         {'col':'2lpd','type':'link','range':[0,3]},
    #         {'col':'3+lpd','type':'link','range':[0,3]},
    #         {'col':'(30,inf) mph','type':'link','range':[0,3]},
    #         # {'col':'(40,inf) mph','type':'link','range':[0,3]},
    #         {'col':'[4k,10k) aadt','type':'link','range':[0,3]},
    #         {'col':'[10k,inf) aadt','type':'link','range':[0,3]},
    #         {'col':'[4,6) grade','type':'link','range':[0,3]},
    #         {'col':'[6,inf) grade','type':'link','range':[0,3]},
    #         {'col':'bike lane','type':'link','range':[-1,0]},
    #         {'col':'multi use path and cycletrack','type':'link','range':[-1,0]},
    #         # {'col':'multi use path','type':'link','range':[-1,0]},
    #         {'col':'unsig_crossing','type':'turn','range':[0,2]},
    #     ),
    #     'set_to_zero': std_set_to_zero,
    #     'set_to_inf': std_set_to_inf,
    #     'objective_function': loss_functions.jaccard_buffer_mean,
    #     'stochastic_optimization_settings': {'method':'pso','options':{'maxiter':100,'popsize':25},'constraints':'shrink'},
    # },
    
    
    
    
    
    # {
    # 'calibration_name': 'rider_type',
    # 'betas_tup': full_model,
    # 'set_to_zero': std_set_to_zero,
    # 'set_to_inf': std_set_to_inf,
    # 'objective_function': stochastic_optimization.jaccard_buffer_mean,
    # },
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
    #     'calibration_name': 'no_dates',
    #     'betas_tup': full_model,
    #     'objective_function': loss_functions.jaccard_buffer_mean
    # },
    # {
    #     'calibration_name': 'baseline',
    #     'betas_tup': full_model,
    #     'set_to_zero': std_set_to_zero,
    #     'set_to_inf': std_set_to_inf,
    #     'objective_function': loss_functions.jaccard_buffer_mean,
    #     'stochastic_optimization_settings': {'method':'pso','options':{'maxiter':100,'popsize':25}},
    # },
    # {
    #     'calibration_name': 'cycletrack',
    #     'betas_tup':  (
    #         {'col':'2lpd','type':'link','range':[0,3]},
    #         {'col':'3+lpd','type':'link','range':[0,3]},
    #         {'col':'(30,inf) mph','type':'link','range':[0,3]},
    #         # {'col':'(40,inf) mph','type':'link','range':[0,3]},
    #         {'col':'[4k,10k) aadt','type':'link','range':[0,3]},
    #         {'col':'[10k,inf) aadt','type':'link','range':[0,3]},
    #         {'col':'[4,6) grade','type':'link','range':[0,3]},
    #         {'col':'[6,inf) grade','type':'link','range':[0,3]},
    #         {'col':'bike lane','type':'link','range':[-1,0]},
    #         {'col':'multi use path and cycletrack','type':'link','range':[-1,0]},
    #         # {'col':'multi use path','type':'link','range':[-1,0]},
    #         {'col':'unsig_crossing','type':'turn','range':[0,2]},
    #     ),
    #     'set_to_zero': std_set_to_zero,
    #     'set_to_inf': std_set_to_inf,
    #     'objective_function': loss_functions.jaccard_buffer_mean,
    #     'stochastic_optimization_settings': {'method':'pso','options':{'maxiter':100,'popsize':25},'constraints':'shrink'},
    # },
    # {
    #     'calibration_name': 'jaccard_buffer_mean',
    #     'betas_tup': full_model,
    #     'set_to_zero': std_set_to_zero,
    #     'set_to_inf': std_set_to_inf,
    #     'objective_function': stochastic_optimization.jaccard_buffer_mean,
    #     'stochastic_optimization_settings': {'method':'pso','options':{'maxiter':500,'popsize':35}},
    #     'print_results':True
    # },
    # {
    #     'calibration_name': 'jaccard_buffer_mean',
    #     'betas_tup': full_model,
    #     'set_to_zero': std_set_to_zero,
    #     'set_to_inf': std_set_to_inf,
    #     'objective_function': stochastic_optimization.jaccard_buffer_mean,
    #     'stochastic_optimization_settings': {'method':'pso','options':{'maxiter':500,'popsize':50}},
    #     'print_results':True
    # },
    # {
    #     'calibration_name': 'cmaes',
    #     'betas_tup': full_model,
    #     'set_to_zero': std_set_to_zero,
    #     'set_to_inf': std_set_to_inf,
    #     'objective_function': stochastic_optimization.jaccard_buffer_mean,
    #     'stochastic_optimization_settings': {'method':'cmaes','options':{'maxiter':500,'popsize':35}},
    #     'print_results':True
    # },
    # {
    #     'calibration_name': 'de',
    #     'betas_tup': full_model,
    #     'set_to_zero': std_set_to_zero,
    #     'set_to_inf': std_set_to_inf,
    #     'objective_function': stochastic_optimization.jaccard_buffer_mean,
    #     'stochastic_optimization_settings': {'method':'de','options':{'maxiter':500,'popsize':35}},
    #     'print_results':True
    # },
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
    #     'force_save': False
    # }
    # {
    #     'calibration_name': 'turns',
    #     'betas_tup': (
    #         {'col':'2lpd','type':'link','range':[0,2]},
    #         {'col':'3+lpd','type':'link','range':[0,2]},
    #         {'col':'(30,inf) mph','type':'link','range':[0,2]},
    #         {'col':'[4k,10k) aadt','type':'link','range':[0,2]},
    #         {'col':'[10k,inf) aadt','type':'link','range':[0,2]},
    #         {'col':'[4,6) grade','type':'link','range':[0,3]},
    #         {'col':'[6,inf) grade','type':'link','range':[0,3]},
    #         {'col':'bike lane','type':'link','range':[-1,0]},
    #         {'col':'cycletrack','type':'link','range':[-1,0]},
    #         {'col':'multi use path','type':'link','range':[-1,0]},
    #         {'col':'unsig_crossing','type':'turn','range':[0,2]},
    #         {'col':'left_turn','type':'turn','range':[0,2]},
    #         {'col':'right_turn','type':'turn','range':[0,2]},
    #     ),
    #     'set_to_zero': std_set_to_zero,
    #     'set_to_inf': std_set_to_inf,
    #     'objective_function': loss_functions.jaccard_buffer_mean,
    #     'stochastic_optimization_settings': {'method':'pso','options':{'maxiter':100,'popsize':25,'return_all':True}}
    # },
    # {
    #     'calibration_name': 'combined_turns',
    #     'betas_tup': (
    #         {'col':'2lpd','type':'link','range':[0,2]},
    #         {'col':'3+lpd','type':'link','range':[0,2]},
    #         {'col':'(30,inf) mph','type':'link','range':[0,2]},
    #         {'col':'[4k,10k) aadt','type':'link','range':[0,2]},
    #         {'col':'[10k,inf) aadt','type':'link','range':[0,2]},
    #         {'col':'[4,6) grade','type':'link','range':[0,3]},
    #         {'col':'[6,inf) grade','type':'link','range':[0,3]},
    #         {'col':'bike lane','type':'link','range':[-1,0]},
    #         {'col':'cycletrack','type':'link','range':[-1,0]},
    #         {'col':'multi use path','type':'link','range':[-1,0]},
    #         # {'col':'unsig_crossing','type':'turn','range':[0,2]},
    #         {'col':'left_or_right_turn','type':'turn','range':[0,2]},
    #         # {'col':'right_turn','type':'turn','range':[0,2]},
    #     ),
    #     'set_to_zero': std_set_to_zero,
    #     'set_to_inf': std_set_to_inf,
    #     'objective_function': loss_functions.jaccard_buffer_mean,
    #     'stochastic_optimization_settings': {'method':'pso','options':{'maxiter':100,'popsize':25,'return_all':True}}
    # }
]