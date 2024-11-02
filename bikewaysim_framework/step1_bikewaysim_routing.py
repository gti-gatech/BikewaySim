import pickle
import concurrent.futures
import itertools
import time
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from bikewaysim.paths import config
from bikewaysim.impedance_calibration import stochastic_optimization
from bikewaysim import general_utils
from bikewaysim.routing import rustworkx_routing_funcs

def run_code(task):
    network_fp = task[0]
    ods = task[1]
    with network_fp.open('rb') as fh:
        turn_G = pickle.load(fh)
    path_lengths, shortest_paths = rustworkx_routing_funcs.rx_shortest_paths(ods,turn_G)
    modeled_results = {(od[0],od[1]):{'length':path_length,'edge_list':shortest_path} for path_length, shortest_path, od in zip(path_lengths,shortest_paths,ods)}
    return network_fp.stem, modeled_results

if __name__ == '__main__':
      
    start_time = time.time()
    
    MAX_WORKERS = 12
    
    od_matrix = pd.read_csv(config['bikewaysim_fp']/'od_matrix.csv')
    ods = list(set(zip(od_matrix['orig_N'],od_matrix['dest_N'])))

    networks = [
        config['bikewaysim_fp']/'current_traveltime.pkl',
        config['bikewaysim_fp']/'current_impedance.pkl',
        config['bikewaysim_fp']/'future_impedance.pkl'
    ]

    # split ods into groups of 1000 each
    chunked_ods = general_utils.chunks(ods,1000)
    tasks = [(x,y) for x,y in itertools.product(networks,chunked_ods)]
    
    completed_tasks = defaultdict(dict)

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for name, result in tqdm(executor.map(run_code, tasks),total=len(tasks)):
                completed_tasks[name].update(result)

    except KeyboardInterrupt:
        print('\nExiting...')

    with (config['bikewaysim_fp']/'parallel_routing.pkl').open('wb') as fh:
        pickle.dump(completed_tasks,fh)

    end_time = time.time()
    print('Took',round((end_time-start_time)/60**2,1),'hours')