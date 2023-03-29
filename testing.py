# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 16:11:09 2023

@author: tpassmore6
"""

import geopandas as gpd
from pathlib import Path
from prepare_network import *
from conflation_tools import *
from network_filter import *

studyarea_name = 'bikewaysim'
working_dir = Path.home() / Path(f'Documents/NewBikewaySimData')

#%%

#import
osm = gpd.read_file(working_dir / Path(f'{studyarea_name}/filtered.gpkg'),layer='osm_links_road')
osm_n = gpd.read_file(working_dir / Path(f'{studyarea_name}/filtered.gpkg'),layer='osm_nodes_road')
osm_bike = gpd.read_file(working_dir / Path(f'{studyarea_name}/filtered.gpkg'),layer='osm_links_bike')
osm_bike_n = gpd.read_file(working_dir / Path(f'{studyarea_name}/filtered.gpkg'),layer='osm_nodes_bike')
#here = gpd.read_file(working_dir / Path(f'{studyarea_name}/filtered.gpkg'),layer='here_links_road')
#arc_bike = gpd.read_file(working_dir / Path('Data/ARC/Regional_Bikeway_Inventory_2022.geojson')).to_crs('epsg:2240')

#get node count
osm_bike_n['num_links'] = osm_bike_n['osm_N'].map(osm_bike['osm_A'].append(osm_bike['osm_B']).value_counts())
dead_ends = osm_bike_n[osm_bike_n['num_links']==1]

#remove dead ends already connected to road network
dead_ends = dead_ends[-dead_ends['osm_N'].isin(osm_n['osm_N'])]

#%%

#split osm road links
split_lines, split_points, unmatched_join_nodes = split_lines_create_points(dead_ends, 'osm', osm, 'osm', 40)

#add nodes
osm_n = pd.concat([osm_n,split_points])

#add ref ids to split links
split_lines = add_ref_ids(split_lines,osm_n,'osm')

osm = add_split_links(osm,split_lines,'osm')

osm = pd.concat([osm,osm_bike])

osm.to_file(Path.home()/Path('Downloads/test.gpkg'),layer='conflated')
#add to network
#osm, osm_n = add_split_links_nodes(osm, osm_n, split_lines, split_points, 'osm')

#%% give ref ids






#split_lines.to_file(Path.home()/Path('Downloads/test.gpkg'),layer='split_lines')
#remove isolates?



#dead_ends.to_file(Path.home()/Path('Downloads/test.gpkg'),layer='dead_ends')


