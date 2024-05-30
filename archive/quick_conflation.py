# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 22:48:22 2022

@author: tpassmore6
"""

from pathlib import Path
import os
import partridge as ptg
import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
import osmnx as ox
import time
from itertools import combinations, chain, permutations, product
from scipy.spatial import cKDTree
from tqdm import tqdm
from datetime import datetime
import pickle


#%% functions

def largest_comp_and_simplify(links,nodes): 
    #create undirected graph
    G = nx.Graph()  # create directed graph
    for ind, row2 in links.iterrows():
        # forward graph, time stored as minutes
        G.add_edges_from([(str(row2['A']), str(row2['B']))])

    #only keep largest component
    largest_cc = max(nx.connected_components(G), key=len)
    
    #simplify graph connect links connect with nodes of degree 2
    #simple_graph = ox.simplification.simplify_graph(largest_cc)
    
    #get nodes
    nodes = nodes[nodes['N'].isin(largest_cc)]
    #get links
    links = links[links['A'].isin(largest_cc) & links['B'].isin(largest_cc)]
    
    return links,nodes

def create_reverse_links(links,network_name):
    
    links_rev = links.rename(columns={'A':'B','B':'A'})

    if 'wrongway' in links_rev.columns:
        #newwrongway
        links_rev['newwrongway'] = 0
    
        #set wrongways to rightways and vice versa
        links_rev.loc[links_rev['wrongway']==1,'newwrongway'] = 0
        links_rev.loc[(links_rev['oneway']==1) & (links_rev['wrongway']==0),'newwrongway'] = 1 
        
        #rename and drop
        links_rev.drop(columns=['wrongway'],inplace=True)
        links_rev.rename(columns={'newwrongway':'wrongway'},inplace=True)

    #add to other links
    links = links.append(links_rev).reset_index(drop=True)

    #make A_B col
    links['A_B'] = links['A'] + '_' + links['B']

    return links

# #%% extend lines code
# from shapely.geometry import *
# import shapely
# import pylab

# #https://stackoverflow.com/questions/33159833/shapely-extending-line-feature
# #https://shapely.readthedocs.io/en/stable/manual.html

# line = LineString([(3, 0), (3, 5), (7, 9.5)])

# #get last segment
# last_segment = LineString(line.coords[-2:])

# #scale using last point as origin
# scaled_last_segment = shapely.affinity.scale(last_segment, xfact=10, yfact=10, origin=last_segment.boundary[0])

# #reconstruct with new extrapolated segment
# new_line = LineString([*line.coords[:-2],*scaled_last_segment.coords])

#%%

#set working directory to where data is stored
user_directory = os.fspath(Path.home())
file_directory = r'\Documents\BikewaySimData\to_conflate'
homeDir = user_directory+file_directory
os.chdir(homeDir)

#settings
studyareaname = 'marta'
network_name = 'osm'

#import OSM (the bike/road links one)
base_links = gpd.read_file(f'{studyareaname}.gpkg',layer='osm_links',driver='GPKG')
base_nodes = gpd.read_file(f'{studyareaname}.gpkg',layer='osm_nodes',driver='GPKG')

#import HERE
join_links = gpd.read_file(f'{studyareaname}.gpkg',layer='here_links',driver='GPKG')
join_nodes = gpd.read_file(f'{studyareaname}.gpkg',layer='here_nodes',driver='GPKG')

#import ARC bike facilities
arc_bike = gpd.read_file('C:/Users/tpassmore6/Documents/BikewaySimData/base_shapefiles/arc/Regional_Bikeway_Inventory_2022.geojson')
arc_bike.to_crs('epsg:2240',inplace=True)

#create unique ids for each osm link
base_links.reset_index(inplace=True)

#ref cols
base_links.rename(columns={'osm_A':'A','osm_B':'B','osm_A_B':'A_B'},inplace=True)
base_nodes.rename(columns={'osm_ID':'N'},inplace=True)

#simplify base network if needed
base_links, base_nodes = largest_comp_and_simplify(base_links,base_nodes)
base_nodes.to_file(f'{studyareaname}.gpkg',layer='conflated_nodes',driver='GPKG')

#%% add here attributes

#dissolve join features via attributes
dissolve_attr = [x for x in list(join_links.columns) if 'here' not in x]
dissolve_attr = [x for x in dissolve_attr if 'geometry' not in x]
dissolved_links = join_links.dissolve(by=dissolve_attr)

#buffer dissolved links
dissolved_links.geometry = dissolved_links.buffer(50)

#extend osm links by 50 ft (FUTURE)

# find where base links intersects dissolved links (returns linestring)
intersected_links = gpd.overlay(base_links, dissolved_links, how='intersection')

#find greatest overlap and assign attributes
intersected_links['intersected_length'] = intersected_links.length

#group by osm_a_b and get maxid
max_overlap = intersected_links.groupby('index')['intersected_length'].idxmax()

#get here_id to osm_id and join back
matches = intersected_links.loc[max_overlap,['index','here_A_B']]

#add here id to links then add here attributes
conflated_links = base_links.merge(matches,on='index',how='left').merge(join_links.drop(columns=['geometry']),on='here_A_B',how='left')

#%% add arc bike attributes

arc_bike = arc_bike[['FID','Name','spec','facil','geometry']]
arc_bike.rename(columns={'Name':'arcname'},inplace=True)

#buffer
arc_bike.geometry = arc_bike.buffer(50)

#intersect
intersected_links = gpd.overlay(conflated_links,arc_bike, how='intersection')

#update length
intersected_links['intersected_length'] = intersected_links.length

#group by index and get maxid
max_overlap = intersected_links.groupby('index')['intersected_length'].idxmax()

#get here_id to osm_id and join back
matches = intersected_links.loc[max_overlap,['index','FID']]

#add here id to links then add here attributes
conflated_links = conflated_links.merge(matches,on='index',how='left').merge(arc_bike.drop(columns=['geometry']),on='FID',how='left')

#%% prepare network and export


#make sure str
conflated_links['A'] = conflated_links['A'].astype(str)
conflated_links['B'] = conflated_links['B'].astype(str)
conflated_links['A_B'] = conflated_links['A_B'].astype(str)

#make a wrongway column
conflated_links['oneway'] = 0
conflated_links.loc[conflated_links['DIR_TRAVEL']=='oneway','oneway'] = 1

conflated_links['wrongway'] = 0
conflated_links.loc[conflated_links['DIR_TRAVEL']=='wrongway','wrongway'] = 1

#create reverse links
conflated_links = create_reverse_links(conflated_links,network_name)


#%%calculate link costs

#attributes
attr = ['bl','pbl','mu','below25','25-30','above30','1laneper','2to3lanesper','4ormorelanesper','wrongway']

#shorten
cl = conflated_links.copy()

#reformat
cl['bl'] = 0
cl.loc[cl['facil']=='Bike Lane','bl'] = 1
cl['pbl'] = 0
cl.loc[cl['facil']=='Protected Bike Lane','pbl'] = 1
cl['mu'] = 0
cl.loc[cl['facil']=='Multi-Use Path','mu'] = 1

cl['below25'] = 0
cl.loc[cl['SPEED_CAT']=='< 25','below25'] = 1
cl['25-30'] = 0
cl.loc[cl['SPEED_CAT']=='25-30','25-30'] = 1
cl['above30'] = 0
cl.loc[cl['SPEED_CAT']=='> 30','above30'] = 1

cl['1laneper'] = 0
cl.loc[cl['LANE_CAT']=='1','1laneper'] = 1
cl['2to3lanesper'] = 0
cl.loc[cl['LANE_CAT']=='2-3','2to3lanesper'] = 1
cl['4ormorelanesper'] = 0
cl.loc[cl['LANE_CAT']=='> 4','4ormorelanesper'] = 1

cl['wrongway'] = 0

#filter
cl = cl[['A','B','A_B','geometry']+attr]

def link_costs(links):
    
    #distance cost in minutes with bike going 9mph
    links['dist'] = links.length #/ 5280 / 10 * 60

    #add the negatives in the future
    costs = {'bl':0.25,
             'pbl':0.50,
             'mu':0.80,
             'below25':0.25,
             '25-30':0.30,
             'above30':1,
             '1laneper':0,
             '2to3lanesper':0.30,
             '4ormorelanesper':1,
             'wrongway':2       
             }

    cols = ['bl','pbl','below25','25-30','above30','1laneper','2to3lanesper',
            '4ormorelanesper','wrongway']
    for col in cols:
        links.loc[links['mu']==1,col] = 0
    
    
    #per dist
    #do this in a matrix way later
    
    #no additional impedance for seperated infrastructure
    #future implementation
    #links.loc[links['mu'] == 1 | links['pbl'] == 1]
    
    good = costs['bl']*links['bl'] + costs['pbl']*links['pbl'] + costs['mu']*links['mu'] + \
        costs['below25']*links['below25'] + costs['1laneper']*links['1laneper']
    
    bad = costs['25-30']*links['25-30'] + costs['above30']*links['above30'] + \
        costs['2to3lanesper']*links['2to3lanesper'] + costs['4ormorelanesper']*links['4ormorelanesper'] + \
            costs['wrongway']*links['wrongway']
    
    costs = links['dist'] * (1-good+bad)
    
    return costs


#filter
cl['imp_dist'] = link_costs(cl)
cl = cl[['A','B','A_B','dist','imp_dist','wrongway','geometry']]

#make sure no negative costs
cl.loc[cl['imp_dist'] < 0,'imp_dist'] = 0.01

#turn dists to ints to save space?
cl['dist'] = cl['dist'].round(3)
cl['imp_dist'] = cl['imp_dist'].round(3)

#export for quick check
cl.to_file(f'{studyareaname}.gpkg',layer='conflated_links',driver='GPKG')


# #need to carve out exceptions for no name streets?
# def dissolve_by_attr(buffered_links,join_name,study_area):
#     #expected filepath pattern for retrieving attributes
#     fp = f'processed_shapefiles/{join_name}/{join_name}_{study_area}_network.gpkg'
    
#     #bring in attributes
#     #may need to change by link type
#     if os.path.exists(fp):
#         #import link attributes (geometry not necessary)
#         attr = gpd.read_file(fp,layer='base_links',ignore_geometry=True)
        
#         #for each network, specify which attributes you want to use to check
#         if join_name == 'here':
#             #use street name, functional classification, speed category, and lane category
#             columns = ['ST_NAME','FUNC_CLASS','SPEED_CAT','LANE_CAT']
#         elif join_name == 'osm':
#             columns = ['osmid']
#         else:
#             columns = None
        
#         #check if there were attributes specified
#         if columns is not None:
#             #turn attribute comlumns into tuple
#             attr['attr_tup'] = [tuple(x) for x in attr[columns].values.tolist()]
            
#             #filter attr df
#             attr = attr[[f'{join_name}_A_B','attr_tup']+columns]
            
#             #merge with join links
#             links = pd.merge(buffered_links,attr,on=f'{join_name}_A_B')
            
#             #get list of dissolved ids
#             #group = links.groupby('attr_tup')
            
#             #disolve geo
#             dissolved_buffer = links.dissolve('attr_tup')
            
#             #reset index
#             dissolved_buffer = dissolved_buffer.reset_index()
            
#             #filter columns
#             dissolved_buffer = dissolved_buffer[[f'{join_name}_A_B','attr_tup','geometry']+columns]
        
#         return dissolved_buffer, columns

# def add_attributes(base_links, join_links, join_name, buffer_ft, study_area, export=True):
    
#     export_fp = rf'processed_shapefiles/conflation/attribute_transfer/attribute_transfer_{join_name}.gpkg'
    
#     #give base_links a temp column so each row has unique identifier
#     # A_B doesn't work because there might be duplicates from the split links step
#     base_links['temp_ID'] = np.arange(base_links.shape[0]).astype(str)
    
#     #calculate original base_links link length
#     base_links['length'] = base_links.geometry.length
    
#     #create copy of join links to use for bufferring
#     buffered_links = join_links.copy()
    
#     #buffer the join links by tolerance_ft
#     buffered_links['geometry'] = buffered_links.buffer(buffer_ft)
    
#     #make sure it's the active geometry
#     buffered_links.set_geometry('geometry',inplace=True)
    
#     #filter join links to just link id and geometry
#     buffered_links = buffered_links[[f'{join_name}_A_B','geometry']]
    
#     #export buffered links
#     buffered_links.to_file(export_fp,layer='buffered_links',driver='GPKG')

#     #intersect with just buffered_links no dissolve 
#     #just_buffer = gpd.overlay(base_links, buffered_links, how='intersection')
    
#     #dissolve the buffers according to attribute data
#     dissolved_buffer, columns = dissolve_by_attr(buffered_links,join_name,study_area)
    
#     #dissolved buffer export
#     dissolved_buffer.drop(columns=['attr_tup']).to_file(export_fp,layer='dissolved_buffer',driver='GPKG')
    
#     #intersect join buffer and base links
#     overlapping_links = gpd.overlay(base_links, dissolved_buffer, how='intersection')
    
#     #re-calculate the link length to see which join buffers had the greatest overlap of base links
#     overlapping_links['percent_overlap'] = (overlapping_links.geometry.length / overlapping_links['length'] )
#     #just_buffer['percent_overlap'] = (just_buffer.geometry.length / just_buffer['length'])
    
#     ##
#     #export overlapping to examine
#     overlapping_links.drop(columns=['attr_tup']).to_file(export_fp,layer='buffered_overlap',driver='GPKG')
#     #just_buffer.to_file(export_fp,layer='just_buffer_overlap',driver='GPKG')
#     ##  
    
#     #select matches with greatest link overlap
#     max_overlap_idx = overlapping_links.groupby('temp_ID')['percent_overlap'].idxmax().to_list()
    
#     #get matching links from overlapping links
#     match_links = overlapping_links.loc[max_overlap_idx]
    
#     # #get list of here links that are overlapping
#     # attr_tup = match_links['attr_tup'].drop_duplicates().to_list()
#     # list_of_lists = [group.get_group(x)[f'{join_name}_A_B'].to_list() for x in attr_tup]
#     # flattened_list = [item for sublist in list_of_lists for item in sublist]
    
#     # #join back with intersected data to get percent overlap
#     # #prolly shouldn't use flattened list for this?
#     # partial_overlap = overlapping_links[overlapping_links[f'{join_name}_A_B_2'].isin(flattened_list)]
#     # partial_overlap = partial_overlap[partial_overlap['percent_overlap'] < 0.8]
#     # partial_overlap = join_links[-(join_links[f'{join_name}_A_B'].isin(partial_overlap[f'{join_name}_A_B_2']))] 
    
#     # #find links that need to be added
#     # rem_join_links = join_links[-(join_links[f'{join_name}_A_B'].isin(flattened_list))] 
    
#     # #add in the partial overlap links
#     # rem_join_links = rem_join_links.append(partial_overlap)
    
#     #join with base links
#     base_links = pd.merge(base_links,match_links[['temp_ID',f'{join_name}_A_B_2']+columns],on='temp_ID',)

#     #resolve join link id
#     base_links.loc[base_links[f'{join_name}_A_B'].isnull(),f'{join_name}_A_B'] = base_links.loc[base_links[f'{join_name}_A_B'].isnull(),f'{join_name}_A_B_2']
    
#     #drop added columns
#     base_links.drop(columns=['temp_ID','length',f'{join_name}_A_B_2'],inplace=True)
    
#     #test_export
#     base_links.to_file(export_fp, layer='base_links_w_join_attr', driver='GPKG')
    
#     #drop the attr columns
#     base_links.drop(columns=columns,inplace=True)
    
#     return base_links




