# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 17:37:50 2022

@author: tpassmore6
"""
#imports
#import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np

import os
from pathlib import Path

user_directory = os.fspath(Path.home()) #get home directory and convert to path string
file_directory = r'\Documents\BikewaySimData'

#change this to where you stored this folder
os.chdir(user_directory+file_directory)


#%% bikeshed code
def make_bikeshed(links,nodes,n,time_min,impedance_col):
    DGo = nx.DiGraph()  # create directed graph
    for ind, row2 in links.iterrows():
        # forward graph, time stored as minutes
        DGo.add_weighted_edges_from([(str(row2['A']), str(row2['B']), float(row2[impedance_col]))])
                                    #weight='forward', name=row2['name'])
    #create bikeshed
    #https://networkx.org/documentation/stable/reference/generated/networkx.generators.ego.ego_graph.html
    #https://geonetworkx.readthedocs.io/en/latest/5_Isochrones.html
    bikeshed = nx.ego_graph(DGo, radius=time_min, n=n, distance='weight')
            
    #merge
    links['A_B_tup'] = list(zip(links['A'],links['B']))

    #get all the links that contained in the egograph
    bikeshed = links.loc[links['A_B_tup'].isin(list(bikeshed.edges)),:]
    bikeshed_node = nodes.loc[nodes['N']==n,:]
    
    #drop tuple column
    bikeshed.drop(columns=['A_B_tup'],inplace=True)
    
    #drop dual links to get accurate size
    df_dup = drop_duplicate_links(bikeshed)
    print(df_dup.length.sum()/5280)
    #print the bikeshed size (assuming ft) (dividing by two to account for double links?)
    print(f'{n} bikeshed size: {bikeshed.length.sum()/5280}')    
    return bikeshed, bikeshed_node


def drop_duplicate_links(links):
    df_dup = pd.DataFrame(np.sort(links[["A","B"]], axis=1), columns=["A","B"])
    df_dup.drop_duplicates(inplace=True)
    merged = pd.merge(links,df_dup,how='inner', left_index = True, right_index = True, suffixes=(None,'_drop')).drop(columns={'A_drop','B_drop'})
    return merged

#%%

# #betweenness_centrality code
# def centrality(links,nodes,impedance_col):
#     DGo = nx.DiGraph()  # create directed graph
#     for ind, row2 in links.iterrows():
#         # forward graph, time stored as minutes
#         DGo.add_weighted_edges_from([(str(row2['A']), str(row2['B']), float(row2[impedance_col] / 5280 / bk_speed_mph * 60))])
#                                     #weight='forward', name=row2['name'])
#     #calculate centrality
#     #https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.edge_betweenness_centrality.html#networkx.algorithms.centrality.edge_betweenness_centrality
#     #links = nx.edge_betweenness_centrality(DGo, weight='weight')    

#     return links



#btw_centrality = centrality(links,nodes,'per_dist',9)             
    
## TRB CODE
# #%%bring in network data


# #specify what network node/taz you want to calculate the bikeshed for
# #ntwrk = 'dist'
# #links = gpd.read_file(rf'trb2023/new_network.geojson')
# #links = gpd.read_file(rf'processed_shapefiles/prepared_network/{ntwrk}/links/links.geojson')
# #nodes = gpd.read_file(rf'processed_shapefiles/prepared_network/{ntwrk}/nodes/nodes.geojson')

# links = gpd.read_file('trb2023/network.gpkg',layer='links')
# imp_links = gpd.read_file('trb2023/network.gpkg',layer='imp_links')
# nodes = gpd.read_file('trb2023/network.gpkg',layer='nodes')

# study_area = gpd.read_file(r'trb2023/studyarea_new.gpkg')
# #taz_centroids = gpd.read_file(r'tu_delft_mac/tu_delft/tazs.gpkg',layer='centroids').to_crs('epsg:2240')

# #%% 
# fp=r'trb2023/link_cost_changes.gpkg'

# #where did link costs go up?
# links[links['per_dist'] > links['dist']].to_file(fp,layer='increased')

# #where did they decrease?
# links[links['per_dist'] < links['dist']].to_file(fp,layer='decreased')

# #where were the network improvements?
# imp_links[imp_links['imp_dist'] != imp_links['per_dist']].to_file(fp,layer='imp_dist')


# #%%
# #find study area centroid
# study_area['geometry'] = study_area.centroid
# study_area.set_geometry('geometry',inplace=True)

# #find nearest function
# closest_node = gpd.sjoin_nearest(study_area,nodes)['N'].item()

# #specify which column contains the link costs 
# link_costs = ['dist','per_dist','imp_dist']

# #%%
# bikeshed = {}
# fp = r'trb2023/bikesheds.gpkg'
# for link_cost in link_costs:
#     if link_cost == 'imp_dist':
#         bikeshed[link_cost], bikeshed_node = make_bikeshed(imp_links,nodes,closest_node,10,link_cost)
#         bikeshed[link_cost].to_file(fp,layer=f'{link_cost}_bikshed',driver='GPKG')
#     else:   
#         bikeshed[link_cost], bikeshed_node = make_bikeshed(links,nodes,closest_node,10,link_cost)
#         bikeshed[link_cost].to_file(fp,layer=f'{link_cost}_bikshed',driver='GPKG')
    
# #export the source node (link cost doesnt matter)
# bikeshed_node.to_file(fp,layer='node',driver='GPKG')

# #%%

# #find removed links
# rem_links = bikeshed['dist'][-(bikeshed['dist']['A_B'].isin(bikeshed['per_dist']['A_B']))]
# rem_links.to_file(fp,layer='removed')

# #find added links
# add_links = bikeshed['per_dist'][-bikeshed['per_dist']['A_B'].isin(bikeshed['dist']['A_B'])]
# add_links.to_file(fp,layer='add')

# #find added links from improvement
# add_impr = bikeshed['imp_dist'][-bikeshed['imp_dist']['A_B'].isin(bikeshed['per_dist']['A_B'])]
# add_impr.to_file(fp,layer='imp add')

#%% for transitsim

import pickle

links = gpd.read_file(r'C:/Users/tpassmore6/Documents/BikewaySimData/to_conflate/marta.gpkg',layer='conflated_links')
nodes = gpd.read_file(r'C:/Users/tpassmore6/Documents/BikewaySimData/to_conflate/marta.gpkg',layer='conflated_nodes')

#user_directory = os.fspath(Path.home())
file_directory = r"\Documents\TransitSimData\Data" #directory of bikewaysim outputs
homeDir = user_directory+file_directory
os.chdir(homeDir)

select_tazs = ['553','411','1071','288']
mode = 'bike'
impedance = 'dist'

if not os.path.exists(rf'Outputs\{mode}_{impedance}'):
    os.makedirs(rf'Outputs\{mode}_{impedance}') 

#import snapped tazs
with open(r'snapped_tazs.pkl','rb') as fh:
    snapped_tazs_dict = pickle.load(fh)

centroids = gpd.read_file('base_layers.gpkg',layer='centroids',driver='GPKG') 

centroids['tazN'] = centroids['FID_1'].map(snapped_tazs_dict)


for taz in select_tazs:
    n = centroids.loc[centroids['FID_1']==taz,'tazN'].item()
    bikeshed, bikeshed_node = make_bikeshed(links,nodes,n,5280*2.5,'dist')

    mode = 'bike'
    #export
    bikeshed.to_file(rf'Outputs\{mode}_{impedance}\{taz}.gpkg',layer=f'{mode}shed')
    
    #do convex hull
    bounds = bikeshed.copy()
    bounds = bounds.dissolve()
    bounds.set_geometry(bounds.convex_hull,inplace=True)
    bounds.to_file(rf'Outputs\{mode}_{impedance}\{taz}.gpkg',layer=f'{mode}shed bounds')
    
for taz in select_tazs:
    n = centroids.loc[centroids['FID_1']==taz,'tazN'].item()
    bikeshed, bikeshed_node = make_bikeshed(links,nodes,n,5280*0.5,'dist')

    mode = 'walk'
    #export
    bikeshed.to_file(rf'Outputs\{mode}_{impedance}\{taz}.gpkg',layer=f'{mode}shed')
    
    #do convex hull
    bounds = bikeshed.copy()
    bounds = bounds.dissolve()
    bounds.set_geometry(bounds.convex_hull,inplace=True)
    bounds.to_file(rf'Outputs\{mode}_{impedance}\{taz}.gpkg',layer=f'{mode}shed bounds')
