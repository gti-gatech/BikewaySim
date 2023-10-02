from bikewaysim_lite import make_bikeshed
from pathlib import Path
import geopandas as gpd
import pandas as pd
import pickle
from tqdm import tqdm

network_fp = Path.home() / 'Documents/TransitSimData/networks/final_network.gpkg'

with (Path.home() / 'Documents/TransitSimData/Data/snapped_tazs.pkl').open('rb') as fh:
    snapped_tazs = pickle.load(fh)

links_c = gpd.read_file(network_fp,layer='links')
nodes = gpd.read_file(network_fp,layer='nodes')
origins = ['553','1071','1005','1377']
modes = ['bike','walk','bikewalk']
buffer_size = 100
impedance_col = 'dist'

for origin in tqdm(origins):
    for mode in modes:
        if (mode == 'bike') | (mode == 'bikewalk'):
            radius = 5280 * 2
        elif mode == 'walk':
            radius = 5280 * 0.625
        else:
            print('Mode not recognized')
            break
        snapped = snapped_tazs[origin]
        bikeshed, bikeshed_node = make_bikeshed(links_c,nodes,snapped,radius,buffer_size,impedance_col)
        bikeshed.to_file(Path.home()/f'Documents/TransitSimData/Data/{mode}_{impedance_col}/visuals/{origin}.gpkg',layer='bikeshed')
        
        bounds = bikeshed.copy()
        bounds_dissolved = bounds.dissolve()
        bounds_dissolved.set_geometry(bounds_dissolved.convex_hull,inplace=True)
        bounds_dissolved.to_file(Path.home()/f'Documents/TransitSimData/Data/{mode}_{impedance_col}/visuals/{origin}.gpkg',layer=f'{mode}_bounds')


