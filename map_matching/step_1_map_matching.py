import pandas as pd
import numpy as np
import geopandas as gpd
import pickle
import time
import concurrent.futures
from tqdm import tqdm
from shapely.ops import Point, LineString
import matplotlib.pyplot as plt
from importlib import reload

from bikewaysim.paths import config
from bikewaysim.map_matching import map_match
from bikewaysim.network import prepare_network
from bikewaysim.general_utils import print_elapsed_time

def batch_run_map_match(idx):
    with (config['matching_fp'] / f'coords_dict_{idx}.pkl').open('rb') as fh:
        coords_dict = pickle.load(fh)

    match_dict = {tripid:map_match.leuven_match(trace,matching_settings,map_con,exploded_links) for tripid, trace in tqdm(coords_dict.items(),total=len(coords_dict))}

    # Now match_dict contains the processed results
    with (config['matching_fp'] / f'match_dict_{idx}.pkl').open('wb') as fh:
        pickle.dump(match_dict,fh)

if __name__ == '__main__':
    start_time = time.time()
    
    # import network
    with (config['matching_fp'] / 'map_con.pkl').open('rb') as fh:
        exploded_links, exploded_nodes = pickle.load(fh)
    map_con = map_match.make_network(exploded_links,exploded_nodes)
    
    # import the match settings
    with (config['matching_fp']/'match_settings.pkl').open('rb') as fh:
        matching_settings = pickle.load(fh)

    # figure out how many splits there were
    coords_dict_fps = (config['matching_fp'] / f'coords_dict_').glob('*.pkl')
    tasks = [int(x.split('_')[-1]) for x in coords_dict_fps]

    try:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for result in executor.map(batch_run_map_match, tasks):
                print_elapsed_time(time.time()-start_time)
    except KeyboardInterrupt:
        print('\nExiting...')

    end_time = time.time()
    print_elapsed_time(time.time()-start_time)    