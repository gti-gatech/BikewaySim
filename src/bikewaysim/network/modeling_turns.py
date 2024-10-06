#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:13:02 2023

@author: tannerpassmore
"""
import networkx as nx
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiLineString, Point, LineString
import warnings
import numpy as np
import pyproj

warnings.filterwarnings("ignore")

def make_multidigraph(network_df, source='A', target='B', linkid ='linkid', oneway='oneway', fwd_azimuth='fwd_azimuth', bck_azimuth='bck_azimuth'):
    '''
    Takes a non-directed graph stored as a list of edges. 
    '''
    MDG = nx.MultiDiGraph()  # Create a MultiDiGraph
    #itertuples used to maintain the type
    for row in network_df[[source, target, linkid, oneway, fwd_azimuth, bck_azimuth]].itertuples(index=False):
        edge_data = {linkid: row[2],'reverse_link': False, 'azimuth': row[4]}
        MDG.add_edge(row[0], row[1], **edge_data)  # Add edge with linkid attribute
        
        #new version doesn't care if oneway
        edge_data['reverse_link'] = True 
        
        #reverse the azimuth
        edge_data['azimuth'] = row[5]
        MDG.add_edge(row[1], row[0], **edge_data)     
        
    return MDG

def add_virtual_links(pseudo_df,pseudo_G,links_df,start_node:int,end_nodes:list):

    '''
    Adds directed virtual links with length 0 needed to perform routing on the pseudo-dual graph network graph.
    
    Notes:
        network_df must have a source and target column with those names
        psudeo_df must have two columns for each source and target link
            for the source link: source_A and source_B
            for the target link: target_A and target_B
        run remove_virtual links afterwards to remove these virtual links
    '''    

    #grab all pseudo graph edges that contain the starting node in the SOURCE_A column (going away from starting node)
    starting_set = pseudo_df.loc[pseudo_df['source_A'] == start_node,['source_A','source']].drop_duplicates()
    starting_set.columns = ['source','target']

    #add starting virtual edges
    for row in starting_set[['source','target']].itertuples(index=False):
        weight = links_df.at[(links_df['linkid']==row[1][0]) & (links_df['reverse_link']==row[1][1]),'link_cost']
        edge_data = {'weight':weight}
        pseudo_G.add_edge(row[0],row[1],**edge_data)

    #grab all pseudo graph edges that contain the starting node in the TARGET_B column (going towards the starting node)
    ending_set = pseudo_df.loc[pseudo_df['target_B'].isin(set(end_nodes)),['target','target_B']].drop_duplicates()
    ending_set.columns = ['source','target']
    
    #add ending virtual edges
    for row in ending_set[['source','target']].itertuples(index=False):
        edge_data = {'weight':0}
        pseudo_G.add_edge(row[0],row[1],**edge_data)

    return pseudo_G, pd.concat([starting_set,ending_set],ignore_index=True)

def remove_virtual_edges(pseudo_G,virtual_edges):
    '''
    Reverses add_virtual_links
    '''
    for row in virtual_edges.itertuples(index=False):
        pseudo_G.remove_edge(row[0],row[1])
    return pseudo_G

def add_virtual_links_new(pseudo_df,pseudo_G,links_df,start_nodes:list,end_nodes:list):

    '''
    Adds directed virtual links with length 0 needed to perform routing on the pseudo-dual graph network graph.
    
    Input: turns dataframe, turn graph list of origin(s), list of destination(s)

    Process:
        - For origins: match to source_A and create an edge: origin node -> (source_linkid,source_reverse_direction)
        - For destinations: match to target_B and create an edge: destination node -> (target_linkid,target_reverse_direction)
        - Run remove_virtual links afterwards to remove these virtual links

    '''    
    #grab all pseudo graph edges that contain the starting node in the SOURCE_A column (going away from starting node)
    starting_set = pseudo_df.loc[pseudo_df['source_A'].isin(set(start_nodes)),['source_A','source_linkid','source_reverse_link']].drop_duplicates().to_numpy()
    ending_set = pseudo_df.loc[pseudo_df['target_B'].isin(set(end_nodes)),['target_linkid','target_reverse_link','target_B']].drop_duplicates().to_numpy()
    
    #add starting virtual edges
    #if the year is 
    for row in starting_set:
        # weight = links_df.loc[(links_df['linkid']==row[1]) & (links_df['reverse_link']==row[2]),'link_cost'].tolist()[0]
        weight = links_df.loc[(links_df[['linkid','reverse_link']]==(row[1],row[2])).all(axis=1),'link_cost'].item()
        edge_data = {'weight':weight}
        pseudo_G.add_edge(int(row[0]),(int(row[1]),bool(row[2])),**edge_data)
    
    #add virtual edges with list comp
    edge_data = {'weight':0}
    # [pseudo_G.add_edge(row[0],(row[1],row[2]),**edge_data) for row in starting_set]
    [pseudo_G.add_edge((int(row[0]),bool(row[1])),int(row[2]),**edge_data) for row in ending_set] # is this actually faster?
    return pseudo_G, starting_set, ending_set

def remove_virtual_links_new(pseudo_G,starting_set,ending_set):
    '''
    Reverses add_virtual_links
    '''
    [pseudo_G.remove_edge(row[0],(row[1],row[2])) for row in starting_set]
    [pseudo_G.remove_edge((row[0],row[1]),row[2]) for row in ending_set]
    return pseudo_G

def find_azimuth(row):
    coords = np.array(row.geometry.coords)
    lat1 = coords[0,1]
    lat2 = coords[-1,1]
    lon1 = coords[0,0]
    lon2 = coords[-1,0]

    geodesic = pyproj.Geod(ellps='WGS84')
    fwd_azimuth,back_azimuth,distance = geodesic.inv(lon1, lat1, lon2, lat2)

    # print('Forward Azimuth:',fwd_azimuth)
    # print('Back Azimuth:',back_azimuth)
    # print('Distance:',distance)
    return pd.Series([np.round(fwd_azimuth,1) % 360, np.round(back_azimuth,1) % 360],index=['fwd_azimuth','bck_azimuth'])

def create_pseudo_dual_graph(edges,source_col='A',target_col='B',linkid_col='linkid',oneway_col='oneway'):
    
    #prevent name changing from happening on the orginal dataframe
    edges = edges.copy()

    #simplify column names and remove excess variables
    edges.rename(columns={source_col:'A',target_col:'B',linkid_col:'linkid',oneway_col:'oneway'},inplace=True)
    edges = edges[['A','B','linkid','oneway','geometry']]
    
    #re-calculate azimuth (now azimuth)
    prev_crs = edges.crs
    edges.to_crs('epsg:4326',inplace=True)
    edges[['fwd_azimuth','bck_azimuth']] = edges.apply(lambda row: find_azimuth(row), axis=1)
    edges.to_crs(prev_crs,inplace=True)

    #turn into undirected graph network with multiple edges
    G = make_multidigraph(edges)
    directed_edges = nx.to_pandas_edgelist(G)
    directed_edges.rename(columns={'source':'A','target':'B'},inplace=True)

    #use networkx line graph function to create pseudo dual graph
    G_turns = nx.line_graph(G)
    turns = nx.to_pandas_edgelist(G_turns)
    del G_turns # clear up memory

    #get expanded tuples to columns for exporting purposes
    turns[['source_A','source_B','source_Z']] = pd.DataFrame(turns['source'].tolist(), index=turns.index)
    turns[['target_A','target_B','target_Z']] = pd.DataFrame(turns['target'].tolist(), index=turns.index)
    
    #drop the duplicate edges (these are addressed in the merge step)
    #line_graph doesn't carry over the linkid, so it resets multi-edges to 0 or 1
    #using edge lookup doesn't really speed this up
    turns.drop(columns=['source','target','source_Z','target_Z'],inplace=True)
    turns.drop_duplicates(inplace=True)

    #merge directed_edges and turns to add linkid and reverse_link keys back in
    turns = turns.merge(directed_edges,left_on=['source_A','source_B'],right_on=['A','B'])
    turns.rename(columns={'linkid':'source_linkid',
                            'reverse_link':'source_reverse_link',
                            'azimuth':'source_azimuth'},inplace=True)
    turns.drop(columns=['A','B'],inplace=True)
    
    turns = turns.merge(directed_edges,left_on=['target_A','target_B'],right_on=['A','B'])
    turns.rename(columns={'linkid':'target_linkid',
                            'reverse_link':'target_reverse_link',
                            'azimuth':'target_azimuth'},inplace=True)
    turns.drop(columns=['A','B'],inplace=True)
    
    #remove backtracking onto the same link (won't affect matching)
    backtracking = turns['source_linkid'] == turns['target_linkid']
    turns = turns[~backtracking]  
    
    #change in azimuth
    turns['azimuth_change'] = (turns['target_azimuth'] - turns['source_azimuth']) % 360
    
    #angle here
    '''
    straight < 30 or > 330
    right >= 30 and <= 150
    uturn > 150 and less than 210 could also be really sharp turns
    left >= 210 and <= 270 
    '''
    straight = (turns['azimuth_change'] > 330) | (turns['azimuth_change'] < 30) 
    right = (turns['azimuth_change'] >= 30) & (turns['azimuth_change'] <= 150)
    backwards = (turns['azimuth_change'] > 150) & (turns['azimuth_change'] < 210)
    left = (turns['azimuth_change'] >= 210) & (turns['azimuth_change'] <= 330)
    
    turns.loc[straight,'turn_type'] = 'straight'
    turns.loc[right,'turn_type'] = 'right'
    turns.loc[backwards,'turn_type'] = 'uturn'
    turns.loc[left,'turn_type'] = 'left'

    #create new source and target columns
    turns['source'] = tuple(zip(turns['source_A'],turns['source_B']))
    turns['target'] = tuple(zip(turns['target_A'],turns['target_B']))

    #drop the tuples for saving
    turns.drop(columns=['source','target'],inplace=True)

    return directed_edges, turns

def turn_gdf(links,turns):

    geo_dict = dict(zip(links['linkid'],links['geometry']))
    turns['source_geo'] = turns['source_linkid'].map(geo_dict)
    turns['target_geo'] = turns['target_linkid'].map(geo_dict)
    turns['geometry'] = turns.apply(lambda row: MultiLineString([row['source_geo'],row['target_geo']]),axis=1)
    turns.drop(columns=['source_geo','target_geo'],inplace=True)
    turns_gdf = gpd.GeoDataFrame(turns,crs=links.crs)

    return turns_gdf



# def create_turn_df(edges,source_col='source',target_col='target',linkid_col='linkid',oneway_col='oneway',keep_uturns=False):
    
#     #simplify column names and remove excess variables
#     edges.rename(columns={source_col:'source',target_col:'target',linkid_col:'linkid',oneway_col:'oneway'},inplace=True)
#     edges = edges[['source','target','linkid','oneway','geometry']]
    
#     #re-calculate azimuth (now azimuth)
#     prev_crs = edges.crs
#     edges.to_crs('epsg:4326',inplace=True)
#     edges[['fwd_azimuth','bck_azimuth']] = edges.apply(lambda row: find_azimuth(row), axis=1)
#     edges.to_crs(prev_crs,inplace=True)
#     #edges['azimuth'] = edges.apply(lambda row: add_azimuth(row),axis=1)

#     #turn into directed graph network wiht multiple edges
#     G = make_multidigraph(edges)
#     df_edges = nx.to_pandas_edgelist(G)

#     #use networkx line graph function to create pseudo dual graph
#     G_line = nx.line_graph(G)
#     df_line = nx.to_pandas_edgelist(G_line)

#     #get expanded tuples to columns for exporting purposes
#     df_line[['source_A','source_B','source_Z']] = pd.DataFrame(df_line['source'].tolist(), index=df_line.index)
#     df_line[['target_A','target_B','target_Z']] = pd.DataFrame(df_line['target'].tolist(), index=df_line.index)
    
#     #drop the duplicate edges (these are addressed in the merge step)
#     #line_graph doesn't carry over the linkid, so it resets multi-edges to 0 or 1
#     df_line.drop(columns=['source','target','source_Z','target_Z'],inplace=True)
#     df_line.drop_duplicates(inplace=True)

#     #merge df_edges and df_line to add linkid and reverse_link keys back in
#     df_line = df_line.merge(df_edges,left_on=['source_A','source_B'],right_on=['source','target'])
#     df_line.rename(columns={'linkid':'source_linkid',
#                             'reverse_link':'source_reverse_link',
#                             'azimuth':'source_azimuth'},inplace=True)
#     df_line.drop(columns=['source','target'],inplace=True)
    
#     df_line = df_line.merge(df_edges,left_on=['target_A','target_B'],right_on=['source','target'])
#     df_line.rename(columns={'linkid':'target_linkid',
#                             'reverse_link':'target_reverse_link',
#                             'azimuth':'target_azimuth'},inplace=True)
#     df_line.drop(columns=['source','target'],inplace=True)
    
#     #remove u-turns
#     if keep_uturns == False:
#         u_turn = (df_line['source_A'] == df_line['target_B']) & (df_line['source_B'] == df_line['target_A'])
#         df_line = df_line[-u_turn]  
    
#     #change in azimuth
#     df_line['azimuth_change'] = (df_line['target_azimuth'] - df_line['source_azimuth']) % 360
    
#     #angle here
#     '''
#     straight < 30 or > 330
#     right >= 30 and <= 150
#     backwards > 150 and less than 210
#     left >= 210 and <= 270 
    
#     '''
#     straight = (df_line['azimuth_change'] > 330) | (df_line['azimuth_change'] < 30) 
#     right = (df_line['azimuth_change'] >= 30) & (df_line['azimuth_change'] <= 150)
#     backwards = (df_line['azimuth_change'] > 150) & (df_line['azimuth_change'] < 210)
#     left = (df_line['azimuth_change'] >= 210) & (df_line['azimuth_change'] <= 330)
    
#     df_line.loc[straight,'turn_type'] = 'straight'
#     df_line.loc[right,'turn_type'] = 'right'
#     df_line.loc[backwards,'turn_type'] = 'uturn'
#     df_line.loc[left,'turn_type'] = 'left'

#     #create new source and target columns
#     df_line['source'] = tuple(zip(df_line['source_A'],df_line['source_B']))
#     df_line['target'] = tuple(zip(df_line['target_A'],df_line['target_B']))

#     #remove duplicate edges (duplicates still retained in df_edges and df_line)
#     #pseudo_df = df_line[['source','target']].drop_duplicates()

#     return df_edges, df_line

def make_turn_graph(df_line):
    #pseudo graph too
    # not sure how to have shortest path algorithm report back the currect multigraph result
    # instead we use pseudo_df to know which link pairs had the lowest cost
    pseudo_G = nx.DiGraph()
    df_line = df_line[['source_linkid','source_reverse_link','target_linkid','target_reverse_link']].drop_duplicates().to_numpy()
    edge_data = {'weight':1}
    [pseudo_G.add_edge((row[0],row[1]),(row[2],row[3]),**edge_data) for row in df_line]        
    return pseudo_G


# def add_azimuth(row):
#     lat1 = row['geometry'].coords.xy[0][1]
#     lat2 = row['geometry'].coords.xy[-1][1]
#     lon1 = row['geometry'].coords.xy[0][0]
#     lon2 = row['geometry'].coords.xy[-1][0]

#     azimuth = calculate_azimuth(lat1,lon1,lat2,lon2)
    
#     return azimuth

# #from osmnx
# #it asks for coordinates in decimal degrees but returns all barings as 114 degrees?
# def calculate_azimuth(lat1, lon1, lat2, lon2):
#     """
#     Calculate the compass azimuth(s) between pairs of lat-lon points.

#     Vectorized function to calculate initial azimuths between two points'
#     coordinates or between arrays of points' coordinates. Expects coordinates
#     in decimal degrees. Bearing represents the clockwise angle in degrees
#     between north and the geodesic line from (lat1, lon1) to (lat2, lon2).

#     Parameters
#     ----------
#     lat1 : float or numpy.array of float
#         first point's latitude coordinate
#     lon1 : float or numpy.array of float
#         first point's longitude coordinate
#     lat2 : float or numpy.array of float
#         second point's latitude coordinate
#     lon2 : float or numpy.array of float
#         second point's longitude coordinate

#     Returns
#     -------
#     azimuth : float or numpy.array of float
#         the azimuth(s) in decimal degrees
#     """
#     # get the latitudes and the difference in longitudes, all in radians
#     lat1 = np.radians(lat1)
#     lat2 = np.radians(lat2)
#     delta_lon = np.radians(lon2 - lon1)

#     # calculate initial azimuth from -180 degrees to +180 degrees
#     y = np.sin(delta_lon) * np.cos(lat2)
#     x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
#     initial_azimuth = np.degrees(np.arctan2(y, x))

#     # normalize to 0-360 degrees to get compass azimuth
#     return initial_azimuth % 360




# #%% create pseudo graph

# #import links

# #create pseudo graph
# pseudo_edges, pseudo_G = create_pseudo_dual_graph(edges)



# #%% for routing


# pseudo_G, virtual_edges = add_virtual_links(edges, df_line, pseudo_G, start_node, end_nodes, 'weight')


# #calculate total cost
# df_line['turn_cost'] = df_line['turn_type'].map(turn_costs)
# (df_line['weight_turns'] = df_line['weight_x'] + df_line['weight_y']) * df_line['turn_cost']

# df_line['linkid'] = list(zip(df_line['source'],df_line['target']))
# turn_dict = df_line.set_index('linkid').to_dict('index')

# pseudo_G = make_graph(df_line)
# pseudo_G_turn_costs = make_graph(df_line,weight='weight_turns')

# #dijkstra takes a single source and finds path to all possible target nodes
# start_node = 69531971
# end_nodes = [69279745]

# #make tuple columns for easier matching
# df_line['source'] = list(zip(df_line['source_A'],df_line['source_B']))
# df_line['target'] = list(zip(df_line['target_A'],df_line['target_B']))




# #peform routing
# impedances, paths = nx.single_source_dijkstra(pseudo_G,start_node,weight="weight")

# #remove virtual edges
# pseudo_G = remove_virtual_edges(pseudo_G,virtual_edges)

# #%% analyze

# end_node = end_nodes[0]
    
# path = paths[end_node]

# #return list of edges without the virtual links
# edge_list = path[1:-1]

# #return list of turns taken
# turn_list = [ (edge_list[i],edge_list[i+1]) for i in range(len(edge_list)-1)]

# #get edge geometry
# edge_gdf = [edge_dict.get(id,0) for id in edge_list]

# #get turn geometry
# turn_gdf = [turn_dict.get(id,0) for id in turn_list]
   
# #turn edges into gdf
# edge_gdf = pd.DataFrame.from_records(edge_gdf)
# turn_gdf = pd.DataFrame.from_records(turn_gdf)

# #turn into gdf
# edge_gdf = gpd.GeoDataFrame(edge_gdf,geometry='geometry',crs=crs)
# turn_gdf = gpd.GeoDataFrame(turn_gdf,geometry='geometry',crs=crs)

# #export for examination
# edge_gdf.to_file(fp/'testing.gpkg',layer=f"{start_node}_{end_node}")
# turn_gdf.drop(columns=['source','target','linkid_x','linkid_y']).to_file(fp/'testing_turns.gpkg',layer=f"{start_node}_{end_node}")


# #%% for turn costs 

# #add virtual edges
# pseudo_G, virtual_edges = add_virtual_links(edges, df_line, pseudo_G_turn_costs, start_node, end_nodes, 'weight')

# #peform routing
# impedances, paths = nx.single_source_dijkstra(pseudo_G_turn_costs,start_node,weight="weight")

# #remove virtual edges
# pseudo_G = remove_virtual_edges(pseudo_G,virtual_edges)

# #%% analyze

# end_node = end_nodes[0]
    
# path = paths[end_node]

# #return list of edges without the virtual links
# edge_list = path[1:-1]

# #return list of turns taken
# turn_list = [ (edge_list[i],edge_list[i+1]) for i in range(len(edge_list)-1)]

# #get edge geometry
# edge_gdf = [edge_dict.get(id,0) for id in edge_list]

# #get turn geometry
# turn_gdf = [turn_dict.get(id,0) for id in turn_list]
   
# #turn edges into gdf
# edge_gdf = pd.DataFrame.from_records(edge_gdf)
# turn_gdf = pd.DataFrame.from_records(turn_gdf)

# #turn into gdf
# edge_gdf = gpd.GeoDataFrame(edge_gdf,geometry='geometry',crs=crs)
# turn_gdf = gpd.GeoDataFrame(turn_gdf,geometry='geometry',crs=crs)

# #export for examination
# edge_gdf.to_file(fp/'testing.gpkg',layer=f"{start_node}_{end_node}")
# turn_gdf.drop(columns=['source','target','linkid_x','linkid_y']).to_file(fp/'testing_turns.gpkg',layer=f"{start_node}_{end_node}")


