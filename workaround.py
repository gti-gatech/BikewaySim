# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 10:12:50 2022

@author: tpassmore6
"""

import networkx as nx
import pandas as pd
import geopandas as gpd
import numpy as np

from itertools import chain
import time
from tqdm import tqdm

import os
from pathlib import Path

#change this to where you stored this folder
os.chdir(Path.home() / Path("Documents/BikewaySimData"))


#take in od data find the neearest node in the network
def snap_ods_to_network(df_points,df_nodes):
    df_nodes_raw = df_nodes.copy()

    #put all ods into one
    origs = df_points[['ori_id','ori_lat','ori_lon']].rename(
        columns={'ori_id':'id','ori_lat':'lat','ori_lon':'lon'})
    dests = df_points[['dest_id','dest_lat','dest_lon']].rename(
        columns={'dest_id':'id','dest_lat':'lat','dest_lon':'lon'})
    comb = origs.append(dests).drop_duplicates()

    comb['geometry'] = gpd.points_from_xy(comb['lon'], comb['lat'], crs='epsg:4326')

    #needs to be projected coordinate system
    comb = gpd.GeoDataFrame(comb).to_crs('epsg:2240')

    comb.columns = comb.columns + '_pts'
    df_nodes_raw.columns = df_nodes_raw.columns + '_nds'
    
    comb = comb.set_geometry('geometry_pts')
    df_nodes_raw = df_nodes_raw.set_geometry('geometry_nds')
    
    #find closest node
    closest_node = ckdnearest(comb, df_nodes_raw)

    # o = origin
    # d = destination
    # o_d = distance between origin and nearest network node
    # o_t = walking time between origin and nearest network node
    # ox_sq = a rounded X coord for origin
    
    #rename columns to make dataframe
    closest_node = closest_node.drop(columns=['lat_pts','lon_pts','geometry_pts','geometry_nds','lon_nds','lat_nds'])
    origs = closest_node.rename(columns={'id_pts':'ori_id','N_nds':'o_node','X_nds':'ox','Y_nds':'oy','dist':'o_d'})
    dests = closest_node.rename(columns={'id_pts':'dest_id','N_nds':'d_node','X_nds':'dx','Y_nds':'dy','dist':'d_d'})
    
    #merge back to df_points
    df_points = pd.merge(df_points, origs, on='ori_id', how='left')
    df_points = pd.merge(df_points, dests, on='dest_id', how='left')
    
    return df_points
    

def create_graph(links,impedance_col):
    DGo = nx.DiGraph()  # create directed graph
    for ind, row2 in links.iterrows():
        DGo.add_weighted_edges_from([(str(row2['A']), str(row2['B']), float(row2[impedance_col]))],weight=impedance_col)   
    return DGo
        

def find_shortest(links,nodes,ods,impedance_col):
    #record the starting time
    time_start = time.time()
    
    #create network graph
    DGo = create_graph(links,impedance_col)    
    
    #initialize empty dicts
    all_impedances = {}
    all_nodes = {}
    all_paths = {}
    
    #listcheck
    ods['tup'] = list(zip(ods.o_node,ods.d_node))
    listcheck = set(ods['tup'].to_list())
    
    #from each unique origin
    for origin in tqdm(ods.o_node.unique()):
        #run dijkstra's algorithm (no limit to links considered)
        impedances, paths = nx.single_source_dijkstra(DGo,origin,weight='weight')    
        
        #iterate through dijkstra results
        for key in impedances.keys():
            #check if trip is in one of the ones we want
            if (origin,key) in listcheck:
                #for each trip id find impedance
                all_impedances[(origin,key)] = impedances[key]
              
                #convert from node list to edge list
                node_list = paths[key]
                edge_list = [ node_list[i]+'_'+node_list[i+1] for i in range(len(node_list)-1)]
                
                #store
                all_nodes[(origin,key)] = node_list
                all_paths[(origin,key)] = edge_list
  
    #add impedance column to ods dataframe
    ods[f'{impedance_col}'] = ods['tup'].map(all_impedances)
    
    #calculate betweenness centrality
    node_btw_centrality = pd.Series(list(chain(*[all_nodes[key] for key in all_nodes.keys()]))).value_counts()
    edge_btw_centrality = pd.Series(list(chain(*[all_paths[key] for key in all_paths.keys()]))).value_counts()
    
    #add betweenness centrality as network attribute
    nodes[f'{impedance_col}_btw_cntrlty'] = nodes['N'].map(node_btw_centrality)
    links[f'{impedance_col}_btw_cntrlty'] = links['A_B'].map(edge_btw_centrality)
    
    #get standardized betweenness centrality
    nodes[f'{impedance_col}_std_btw_cntrlty'] = nodes[f'{impedance_col}_btw_cntrlty'] / nodes[f'{impedance_col}_btw_cntrlty'].sum()
    links[f'{impedance_col}_std_btw_cntrlty'] = links[f'{impedance_col}_btw_cntrlty'] / links[f'{impedance_col}_btw_cntrlty'].sum()

    print('Creating paths')
    for key in tqdm(all_paths.keys()):
        #get geo (this takes a bit)
        all_paths[key] = links[links['A_B'].isin(all_paths[key])].dissolve().geometry.item()

    #add geo column from dict
    ods['geometry'] = ods['tup'].map(all_paths)
    
    #create gdf
    ods = gpd.GeoDataFrame(ods,geometry='geometry',crs='epsg:2240')
    
    #drop tuple
    ods.drop(columns=['tup'],inplace=True)

    print(f'took {round(((time.time() - time_start)/60), 2)} minutes')
    
    return ods, links, nodes

#%% i think this is just for testing
# ods = pd.read_csv(r'bikewaysim_outputs/samples_in/all_tazs.csv')
# links = gpd.read_file(r'trb2023/network.gpkg',layer='links')
# imp_links = gpd.read_file('trb2023/network.gpkg',layer='imp_links')
# nodes = gpd.read_file(r'trb2023/network.gpkg',layer='nodes')

# #get network nodes
# ods = snap_ods_to_network(ods,nodes)

# #filter ods
# ods = ods[['trip_id','o_node','d_node']]

# impedance_cols = ['dist','per_dist','imp_dist']

# impedance_col = 'dist'
# ods, links, nodes = find_shortest(links, nodes, ods, impedance_col)

# #exporting
# #export_time = time.time()
# #all_paths.to_file(r'trb2023\outputs.gpkg',layer=impedance_col,driver='GPKG')
# #print(f'export took {round(((time.time() - export_time)/60), 2)} minutes')
# links.to_file(r'trb2023\outputs.gpkg',layer=impedance_col+'_new',driver='GPKG')
# nodes.to_file(r'trb2023\outputs.gpkg',layer=impedance_col+'_nodesnew',driver='GPKG')



#%% this is for running all


ods = pd.read_csv(r'bikewaysim_outputs/samples_in/all_tazs.csv')
links = gpd.read_file(r'trb2023/network.gpkg',layer='links')
imp_links = gpd.read_file('trb2023/network.gpkg',layer='imp_links')
nodes = gpd.read_file(r'trb2023/network.gpkg',layer='nodes')

#reduce od size (if needed)
#ods=ods.iloc[0:100,:]


#get network nodes
ods = snap_ods_to_network(ods,nodes)

#filter ods
ods = ods[['trip_id','o_node','d_node']]
ods['tup'] = list(zip(ods.o_node,ods.d_node))

impedance_cols = ['dist','per_dist','imp_dist']

for impedance_col in impedance_cols:
    if impedance_col =='imp_dist':
        ods, imp_links = find_shortest(imp_links, nodes, ods, impedance_col)
    else:
        ods, links = find_shortest(links,nodes,ods,impedance_col)
        
#%%

#export
# ods.to_csv(r'trb2023\impedances.csv',index=False)

#replace if both na
links['dist_btw_cntrlty'].fillna(0,inplace=True)
links['per_dist_btw_cntrlty'].fillna(0,inplace=True)

#links.loc[links['dist_btw_cntrlty'] == 0 & links['per_dist_btw_cntrlty'] == 0,'difference'] = np.nan


#calculate physical link distance


links['difference'] = links['dist_btw_cntrlty'] - links['per_dist_btw_cntrlty']

# links.to_file(r'trb2023\outputs.gpkg',layer='btw_cntrlty',driver='GPKG')
# imp_links.to_file(r'trb2023\outputs.gpkg',layer='imp_btw_cntrlty',driver='GPKG')




#%% analyze


#find avg percent detour
#ods['percent_detour'] = (ods['per_dist']-ods['dist']) / ods['dist'] * 100
#avg_percent_detour = ods.percent_detour.mean()


#%%
dist_trips = gpd.read_file(r'C:/Users/tpassmore6/Documents/BikewaySimData/trb2023/outputs.gpkg',layer='dist')
per_dist_trips = gpd.read_file(r'C:/Users/tpassmore6/Documents/BikewaySimData/trb2023/outputs.gpkg',layer='per_dist')
imp_dist_trips = gpd.read_file(r'C:/Users/tpassmore6/Documents/BikewaySimData/trb2023/outputs.gpkg',layer='imp_dist')


#get length
dist_trips['dist_length'] = dist_trips.length / 5280
per_dist_trips['per_dist_length'] = per_dist_trips.length / 5280
imp_dist_trips['imp_dist_length'] = imp_dist_trips.length / 5280

#drop geo
dist_trips.drop(columns=['geometry'],inplace=True)
per_dist_trips.drop(columns=['geometry'],inplace=True)
imp_dist_trips.drop(columns=['geometry'],inplace=True)

#merge together
test = pd.merge(dist_trips,per_dist_trips,on='trip_id')
test = pd.merge(test,imp_dist_trips,on='trip_id')

#find diff
test['percent_detour'] = (test['per_dist_length'] - test['dist_length']) / test['dist_length'] * 100
#test['percent_detour_imp']= (test['imp_dist_length'] - test['dist_length']) / test['dist_length'] * 100
#test['percent_detour_change'] = test['percent_detour'] - test['percent_detour_imp']

#split into ods
test['taz'] = test.trip_id.str.split('_',expand=True)[0]

#group
by_taz = test.groupby('taz')['percent_detour'].mean().reset_index()

#join to taz
tazs = gpd.read_file(r'trb2023/tazs.gpkg',layer='tazs')

tazs.MTAZ10 = tazs.MTAZ10.astype(np.int64).astype(str)
#
tazs = pd.merge(tazs,by_taz,left_on='MTAZ10',right_on='taz')

tazs.to_file(r'trb2023\outputs.gpkg',layer='zone_percent_detour',driver='GPKG')

#%%

dist_trips = gpd.read_file(r'C:/Users/tpassmore6/Documents/BikewaySimData/trb2023/outputs.gpkg',layer='dist')
imp_dist_trips = gpd.read_file(r'C:/Users/tpassmore6/Documents/BikewaySimData/trb2023/outputs.gpkg',layer='imp_dist')

#get length
dist_trips['dist_length'] = dist_trips.length / 5280
imp_dist_trips['imp_dist_length'] = imp_dist_trips.length / 5280

#drop geo
dist_trips.drop(columns=['geometry'],inplace=True)
imp_dist_trips.drop(columns=['geometry'],inplace=True)

#merge together
test = pd.merge(dist_trips,imp_dist_trips,on='trip_id')

#find diff
test['percent_detour'] = (test['imp_dist_length'] - test['dist_length']) / test['dist_length'] * 100

#split into ods
test['taz'] = test.trip_id.str.split('_',expand=True)[0]




#group
by_taz = test.groupby('taz')['percent_detour'].mean().reset_index()

#join to taz
tazs = gpd.read_file(r'trb2023/tazs.gpkg',layer='tazs')

tazs.MTAZ10 = tazs.MTAZ10.astype(np.int64).astype(str)
#
tazs = pd.merge(tazs,test,left_on='MTAZ10',right_on='taz')

tazs.to_file(r'trb2023\outputs.gpkg',layer='zone_percent_detour_imp',driver='GPKG')
#%%

ods['taz'] = ods.trip_id.str.split('_',expand=True)[0]

#aggregate to tazs
by_taz = ods.groupby('taz')['dist','per_dist','imp_dist'].mean().reset_index()

#find avg change
by_taz['imp_change'] = by_taz['imp_dist'] - by_taz['per_dist']

#get tazs
tazs = gpd.read_file(r'trb2023/tazs.gpkg',layer='tazs')

tazs.MTAZ10 = tazs.MTAZ10.astype(np.int64).astype(str)
#
tazs = pd.merge(tazs,by_taz,left_on='MTAZ10',right_on='taz')

#export
#tazs.to_file(r'trb2023\outputs.gpkg',layer='zonal_stats',driver='GPKG')


#%%

#find avg impedance
ods.dist.mean()
ods.per_dist.mean()
ods.imp_dist.mean()

# import pickle

# def backup(ods,all_paths_geo_dict,links):
#     backup_dict = {'ods':ods,'all_paths_geo_dict':all_paths_geo_dict,
#                    'links':links}
    
#     with open(r'C:\Users\tpassmore6\Documents\BikewaySimData\bikewaysim_outputs.pkl', 'wb') as fh:
#         pickle.dump(backup_dict,fh)
    
# def load_backup():
#     with open('processed_shapefiles/conflation/backup.pkl', 'rb') as fh:
#         backup_dict = pickle.load(fh)
    
#         ods = backup_dict['ods']
#         all_paths_geo_dict = backup_dict['all_paths_geo_dict']
#         links = backup_dict['links']
    
#     return ods,all_paths_geo_dict,links

#export



#backups(ods,all_paths_geo_dict,links)


#test = pd.Series(all_paths).value_counts()


#create path lines
#create_paths(links,all_paths)



#%%


#filter ods
# ods_filt = ods[['trip_id','o_node','d_node']]

# #get only actual dests
# dests = ods.loc[ods['o_node']==origin,'d_node'].to_list()

#%%


# #ods['dist']=None


# origin = '2044584878'

# #get shortest path
# impedances, paths = nx.single_source_dijkstra(DGo,origin,weight='weight')

# all_impedances = {} 
# all_paths = {}




# test = pd.DataFrame.from_dict(impedances,orient='index').reset_index()
# test.columns = ['d_node','dist']
# test['o_node'] = origin

# df2 = pd.merge(ods,test,on=['o_node','d_node'])

# ods.update(df2)

# #turn impedances to series
# impedances = pd.DataFrame(impedances).reset_index()

# test = pd.merge(ods_filt,impedances,left_on=['o_node','d_node'],right_on=['',''],how='left')


# shortest_paths.append({impedance[x] for x in impedances if })




# #%%

# #find all possible shortst paths
# shortest_paths = find_shortest(DGo)
