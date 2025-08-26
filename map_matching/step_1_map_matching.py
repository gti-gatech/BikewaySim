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
import os

from bikewaysim.paths import config
from bikewaysim.map_matching import map_match, map_match_utils
from bikewaysim.network import prepare_network
from bikewaysim.general_utils import print_elapsed_time

# import network
with (config['matching_fp'] / 'map_con.pkl').open('rb') as fh:
    exploded_links, exploded_nodes = pickle.load(fh)
map_con = map_match.make_network(exploded_links,exploded_nodes)

# import the match settings
with (config['matching_fp']/'match_settings.pkl').open('rb') as fh:
    matching_index, matching_settings = pickle.load(fh)

# import traces
with (config['matching_fp'] / "coords_dict.pkl").open('rb') as fh:
    coords_dict = pickle.load(fh)

def create_batches(coords_dict, num_trips=500):
    """
    Split the coords_dict into batches of num_trips each.
    """
    small_coords = []
    small_dict = {}
    for idx, (tripid, item) in enumerate(coords_dict.items()):
        # 500 each
        if (idx % num_trips == 0) & (idx != 0):
            small_dict[tripid] = item
            small_coords.append(small_dict)
            small_dict = {}
        else:
            small_dict[tripid] = item
    if small_dict:  # Add the last batch if it exists
        small_coords.append(small_dict)
    
    return small_coords

def batch_run_map_match(task):
    idx, batch_dict = task
    match_dict = {tripid:map_match.leuven_match(trace,matching_settings,map_con,exploded_links) for tripid, trace in tqdm(batch_dict.items(),total=len(batch_dict))}

    # Now match_dict contains the processed results
    with (config['matching_fp'] / f'match_dict_{idx}.pkl').open('wb') as fh:
        pickle.dump(match_dict,fh)

def combine_results():
    fps = config['matching_fp'].glob("match_dict_*.pkl")
    match_dict = {}
    i = 0
    for fp in fps:
        if fp.parts[-1] == 'match_dict_full.pkl':
            continue
        with fp.open('rb') as fh:
            small_match_dict = pickle.load(fh)
        match_dict.update(small_match_dict)
        i += len(small_match_dict)
        del small_match_dict

        # delete the intermediate files
        os.remove(fp)

    with (config['matching_fp'] / f'match_dict_full_{matching_index}.pkl').open('wb') as fh:
        pickle.dump(match_dict,fh)

if __name__ == '__main__':
    
    start_time = time.time()
    
    # Create batches of trips to match (speeds up the process)
    batched_trips = create_batches(coords_dict, num_trips=500)
    tasks = [(idx, batch) for idx, batch in enumerate(batched_trips)]

    try:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for result in executor.map(batch_run_map_match, tasks):
                print_elapsed_time(time.time()-start_time)
    except KeyboardInterrupt:
        print('\nExiting...')

    # combines the results from the batches into a single match_dict and exports it
    combine_results()

    end_time = time.time()
    print_elapsed_time(time.time()-start_time)