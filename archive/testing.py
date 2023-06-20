# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 16:11:09 2023

@author: tpassmore6
"""

import geopandas as gpd
from pathlib import Path
import numpy as np

from prepare_network import *
from conflation_tools import *
from network_filter import *
from network_reconcile import *


studyarea_name = 'bikewaysim'
working_dir = Path.home() / Path(f'Documents/NewBikewaySimData')
network = 'osm'

#import links
links = gpd.read_file(working_dir / Path(f'{studyarea_name}/filtered.gpkg'),layer='osm_links_roadbike')

#bring in attribute data
attr_fp = Path.home() / "Documents/NewBikewaySimData" / studyarea_name / "osm.pkl"
attr = pd.read_pickle(attr_fp)

#attach attribute data to filtered links
links = pd.merge(links,attr,on=['osm_A','osm_B','osm_A_B'])

#init columns
columns_to_add = ['bl','pbl','mu','<25mph','25-30mph','>30mph','1lpd','2-3lpd','>4lpd']

for col in columns_to_add:
    links[network + '_' + col] = 0

#%% bike facil

#find bike lanes
#get all bike specific columns (find features with cycle, foot, or bike in tags but not motorcycle)
bike_columns = [x for x in links.columns.to_list() if (('cycle' in x) | ('bike' in x)) & ('motorcycle' not in x)]
foot_columns = [x for x in links.columns.to_list() if ('foot' in x)]
bike_columns = bike_columns + foot_columns + ['lit']
bl_cond = ((links[bike_columns] == 'lane').any(axis=1)) & (links['highway'] != 'cycleway')
#bl.to_file(Path.home() / 'Downloads/testing.gpkg', layer = 'bls')

#anything that contains lane is a bike lane
links.loc[bl_cond,network+'_bl'] = 1


#find protected bike lanes
pbl_cond = (links['highway'] == 'cycleway') | links['highway_1']=='cycleway'#& (links['foot'] == 'no')
links.loc[pbl_cond, network+'_pbl'] = 1

#find mups
mups = ['path','footway','pedestrian','steps']
mup_cond = (links['highway'].isin(mups)) | ((links['highway'] == 'cycleway') & (links['foot'] != 'no'))
links.loc[mup_cond, network+'_mu'] = 1

#resolve conflicts
if (links[[network+'_mu',network+'_pbl',network+'_bl']].sum(axis=1) > 1).any():
    print('more than one bike facility detected')

#%% speed limit

#change nones to NaN
links['maxspeed'] = pd.to_numeric(links['maxspeed'],errors='coerce').fillna(links['maxspeed'])

#covert to ints
func = lambda x: int(str.split(x,' ')[0]) if type(x) == str else (np.nan if x == None else int(x))
links['maxspeed'] = links['maxspeed'].apply(func)

links[(links['maxspeed'] < 25), network+'_<25mph'] = 1
links[(links['maxspeed'] >= 25) & (links['maxspeed'] <= 30), network+'_25-35mph'] = 1
links[(links['maxspeed'] > 30), network+'_>30mph'] = 1

# =============================================================================
# # old
# #speed categories
# bins = [0,24,30,999]
# names = ['< 25','25-30','> 30']
# 
# #replace
# links['maxspeed'] = pd.cut(links['maxspeed'],bins=bins,labels=names)
# links['maxspeed'] = links['maxspeed'].astype(str)
# =============================================================================

#%% number of lanes

#convert none to nan
links['lanes'] = links['lanes'].apply(lambda x: np.nan if x == None else x)

#make sure numeric
links['lanes'] = pd.to_numeric(links['lanes'])

links.loc[links['lanes'] < 3, network+'1lpd'] = 1
links.loc[(links['lanes'] >= 3) & (links['lanes'] < 6), network+'2-3lpd'] = 1
links.loc[links['lanes'] >= 8, network+'>4lpd'] = 1


# =============================================================================
# #speed cats
# bins = [0,3,6,999]
# names = ['one lane', 'two or three lanes', 'four or more']
# 
# #replace
# links['lanes'] = pd.cut(links['lanes'],bins=bins,labels=names)
# links['lanes'] = links['lanes'].astype(str)
# =============================================================================


#OTHER ATTRIBUTES
# =============================================================================
# 
# # road directionality
# links.loc[(links['oneway']=='-1') | (links['oneway'] is None),'oneway'] = 'NA'
# 
# # parking
# parking_columns = [x for x in links.columns.to_list() if 'parking' in x]
# parking_vals = ['parallel','marked','diagonal','on_street']
# links.loc[(links[parking_columns].isin(parking_vals)).any(axis=1),'parking_pres'] = 'yes'
# links.loc[(links[parking_columns]=='no').any(axis=1),'parking_pres'] = 'yes'
# links.drop(columns=parking_columns,inplace=True)
# 
# # sidewalk presence
# sidewalks = [x for x in links.sidewalk.unique().tolist() if x not in [None,'no','none']]
# links['sidewalk_pres'] = 'NA'
# links.loc[links['sidewalk'].isin(sidewalks),'sidewalk_pres'] = 'yes'
# links.loc[links['sidewalk'].isin([None,'no','none']),'sidewalk_pres'] = 'no'
# links.drop(columns=['sidewalk'],inplace=True)
# =============================================================================

final_cols = ['A','B','A_B'] + columns_to_add
final_cols = [ network + '_' + x for x in final_cols]

links = links[final_cols+['geometry']]
links.to_final(Path.home() / 'Downloads/testing.gpkg', layer = 'osm')

#%%
conflicted.to_file(Path.home() / 'Downloads/testing.gpkg', layer = 'conflicted')
links[pbl_cond].to_file(Path.home() / 'Downloads/testing.gpkg', layer = 'pbls')



#%%

studyarea_name = 'bikewaysim'
working_dir = Path.home() / Path(f'Documents/NewBikewaySimData')

links = gpd.read_file(working_dir / studyarea_name / 'reconciled_network.gpkg',layer='links')
nodes = gpd.read_file(working_dir / studyarea_name / 'reconciled_network.gpkg',layer='nodes')

#make all nan and NONEs NAs
links = links.replace('nan',np.nan)

#r
#@links.drop(columns=['']

#%% BIKE FACILS

#set all links with speed less than 25 to mu




bike_cols = ['highway','highway_1','crossing','SPEED_CAT','pbl','bl','mu']
test = links[bike_cols]



#%% SPEED

#examine speed
speed = links[['SPEED_CAT','maxspeed','geometry']]

#set all nas to <25
nas = speed[speed[['SPEED_CAT','maxspeed']].isna().all(axis=1)]





#%%

#import
osm = gpd.read_file(working_dir / Path(f'{studyarea_name}/filtered.gpkg'),layer='osm_links_road')
osm_n = gpd.read_file(working_dir / Path(f'{studyarea_name}/filtered.gpkg'),layer='osm_nodes_road')
osm_bike = gpd.read_file(working_dir / Path(f'{studyarea_name}/filtered.gpkg'),layer='osm_links_bike')
osm_bike_n = gpd.read_file(working_dir / Path(f'{studyarea_name}/filtered.gpkg'),layer='osm_nodes_bike')
#here = gpd.read_file(working_dir / Path(f'{studyarea_name}/filtered.gpkg'),layer='here_links_road')
#arc_bike = gpd.read_file(working_dir / Path('Data/ARC/Regional_Bikeway_Inventory_2022.geojson')).to_crs('epsg:2240')


#%% get node count
osm_bike_n['num_links'] = osm_bike_n['osm_N'].map(osm_bike['osm_A'].append(osm_bike['osm_B']).value_counts())

# get rid of nodes that already in road layer
test = osm_bike_n[-osm_bike_n['osm_N'].isin(osm_n['osm_N'])]

#use nearest to make connectors
#need ot think about implication of just using split vs join by nearest

#remove if connection found

#use dead ends to 


#%%

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


