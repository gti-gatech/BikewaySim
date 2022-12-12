# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 05:41:07 2022

@author: tpassmore6
"""

import os
from pathlib import Path
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import time


user_directory = os.fspath(Path.home()) #get home directory and convert to path string
file_directory = r'\Documents\BikewaySimData\to_conflate' #directory of bikewaysim network processing code

#change this to where you stored this folder
os.chdir(user_directory+file_directory)

#%%
    
def prepare_network(links,nodes,link_costs):
    
    #record the starting time
    time_start = time.time()
    
    #group by attributes and get list of edges with identical attributes
    #use this in the network reduction step
    
    #prepare nodes
    nodes = create_bws_nodes(nodes)

    #prepare links
    links = create_bws_links(links)

    #drop unconnect links and remove interstitial nodes
    links,nodes = largest_comp_and_simplify(links,nodes)

    #create reverse streets
    links = create_reverse_links(links)

    # if link_costs == True:
    #     #calculate link costs
    #     links['per_dist'] = link_costs(links)
        
    print(f'took {round(((time.time() - time_start)/60), 2)} minutes')
    
    return links, nodes


def create_bws_links(links):
    #rename ID column
    a_cols = links.columns.tolist()
    a_cols = [a_cols for a_cols in a_cols if "_A" in a_cols]
    
    b_cols = links.columns.tolist()
    b_cols = [b_cols for b_cols in b_cols if "_B" in b_cols]
    
    a_b_cols = links.columns.tolist()
    a_b_cols = [a_b_cols for a_b_cols in a_b_cols if "A_B" in a_b_cols]
    
    #replace with desired hierarchy
    links = links.rename(columns={a_cols[0]:'A'})
    links = links.rename(columns={b_cols[0]:'B'})
    links = links.rename(columns={a_b_cols[0]:'A_B'})
    
    return links


def create_bws_nodes(nodes):
    #find original crs
    orig_crs = nodes.crs
    
    #make UTM coords columns
    nodes['X'] = nodes.geometry.x
    nodes['Y'] = nodes.geometry.y
    
    #convert to geographic
    nodes = nodes.to_crs("epsg:4326")
    
    #make lat/lon cols
    nodes['lon'] = nodes.geometry.x
    nodes['lat'] = nodes.geometry.y

    #convert back
    nodes = nodes.to_crs(orig_crs)
    
    #rename ID column
    id_cols = nodes.columns.tolist()
    id_cols = [id_cols for id_cols in id_cols if "_ID" in id_cols]
    
    #replace with desired hierarchy
    nodes = nodes.rename(columns={id_cols[0]:'N'})
    
    #filter to needed columns
    nodes = nodes[['N','X','Y','lon','lat','geometry']]
    
    return nodes

def create_reverse_links(links):
    
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

def largest_comp_and_simplify(links,nodes): 
    #create undirected graph
    G = nx.Graph()  # create directed graph
    for ind, row2 in links.iterrows():
        # forward graph, time stored as minutes
        G.add_edges_from([(str(row2['A']), str(row2['B']))])

    #only keep largest component
    largest_cc = max(nx.connected_components(G), key=len)
    
    #simplify graph connect links connect with nodes of degree 2
    simple_graph = ox.simplification.simplify_graph(largest_cc)
    
    #get nodes
    nodes = nodes[nodes['N'].isin(simple_graph)]
    #get links
    links = links[links['A'].isin(simple_graph) & links['B'].isin(simple_graph)]
    
    return links,nodes
    
def link_costs(links):
    
    #distance cost in minutes with bike going 9mph
    links['dist'] = links.length / 5280 / 8 * 60

    #add the negatives in the future
    costs = {'bl':0.25,
             'pbl':0.95,
             'mu':0.95,
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

#%% 

# def simplify_graph(G,links):
    
#     if nx.degree(G)


# import numpy as np
# import matplotlib.pyplot as plt
# import networkx as nx


# def contract(g):
#     """
#     Contract chains of neighbouring vertices with degree 2 into a single edge.

#     Arguments:
#     ----------
#     g -- networkx.Graph or networkx.DiGraph instance

#     Returns:
#     --------
#     h -- networkx.Graph or networkx.DiGraph instance
#         the contracted graph

#     """

#     # create subgraph of all nodes with degree 2
#     is_chain = [node for node, degree in g.degree() if degree == 2]
#     chains = g.subgraph(is_chain)

#     # contract connected components (which should be chains of variable length) into single node
#     components = [chains.subgraph(c) for c in nx.components.connected_components(chains)]

#     hyper_edges = []
#     for component in components:
#         end_points = [node for node, degree in component.degree() if degree < 2]
#         candidates = set([neighbor for node in end_points for neighbor in g.neighbors(node)])
#         connectors = candidates - set(list(component.nodes()))
#         hyper_edge = list(connectors)
#         weights = [component.get_edge_data(*edge)['weight'] for edge in component.edges()]
#         hyper_edges.append((hyper_edge, np.sum(weights)))

#     # initialise new graph with all other nodes
#     not_chain = [node for node in g.nodes() if not node in is_chain]
#     h = g.subgraph(not_chain).copy()
#     for hyper_edge, weight in hyper_edges:
#         h.add_edge(*hyper_edge, weight=weight)

#     return h

# # # create weighted graph
# # edges = np.random.randint(0, 20, size=(int(400*0.2), 2))
# # weights = np.random.rand(len(edges))
# # g = nx.Graph()
# # for edge, weight in zip(edges, weights):
# #     g.add_edge(*edge, weight=weight)
# # h = nx.algorithms.minimum_spanning_tree(g)

# # contract
# i = contract(DGo)

# # # plot
# # pos = nx.spring_layout(h)

# # fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
# # nx.draw(h, pos=pos, ax=ax1)
# # nx.draw(i, pos=pos, ax=ax2)
# # plt.show()
# # create subgraph of all nodes with degree 2
# is_chain = [node for node, degree in DGo.degree() if degree == 2]
# chains = g.subgraph(is_chain)





#%% bring in links

# fp = 'processed_shapefiles/conflation/finalized_networks/here_trb.gpkg'
# links = gpd.read_file(fp,layer='links')
# fp = 'processed_shapefiles/here/here_bikewaysim_network.gpkg'
# nodes = gpd.read_file(fp,layer='roadbike_nodes')


#osm
road_links = gpd.read_file(r'processed_shapefiles\osm\osm_marta_network.gpkg',layer='road_links')
road_nodes = gpd.read_file(r'processed_shapefiles\osm\osm_marta_network.gpkg',layer='road_nodes')

bike_links = gpd.read_file(r'processed_shapefiles\osm\osm_marta_network.gpkg',layer='bike_links')
bike_nodes = gpd.read_file(r'processed_shapefiles\osm\osm_marta_network.gpkg',layer='bike_nodes')

#%%combine layers
roadbike_links = road_links.append(bike_links)
roadbike_nodes = road_nodes.append(bike_nodes).drop_duplicates()

#%%

# roadbike_links.rename(columns={'osm_A':'A','osm_B':'B'},inplace=True)


# #create undirected graph
# G = nx.Graph()  # create directed graph
# for ind, row2 in roadbike_links.iterrows():
#     # forward graph, time stored as minutes
#     G.add_edges_from([(str(row2['A']), str(row2['B']))])

# #only keep largest component
# largest_cc = max(nx.connected_components(G), key=len)

# #%%

# #get nodes
# roadbike_nodes.rename(columns={'osm_ID':'N'},inplace=True)
# roadbike_nodes = roadbike_nodes[roadbike_nodes['N'].isin(largest_cc)]
    
# #use nodes to get links
# roadbike_links = roadbike_links[roadbike_links['A'].isin(largest_cc) & roadbike_links['B'].isin(largest_cc)]

#%%
links, nodes = prepare_network(roadbike_links,roadbike_nodes,link_costs=False)

#export
print('exporting...')
links.to_file(r'C:\Users\tpassmore6\Documents\TransitSimData\Data\osm_network.gpkg',layer='links',driver='GPKG')
nodes.to_file(r'C:\Users\tpassmore6\Documents\TransitSimData\Data\osm_network.gpkg',layer='nodes',driver='GPKG')
print('done.')

# #make improvements
# improvements = links.copy()
# changes = '10TH ST NW'
# #add mu
# improvements.loc[improvements['ST_NAME']==changes,'mu'] = 1

# #bring in new feature
# new = gpd.read_file('trb2023/network_improvements.gpkg',layer='new')

# #make reverse
# new = create_reverse_links(new)

# #add to network
# improvements = improvements.append(new)

# #add impr cost
# improvements['imp_dist'] = link_costs(improvements)

# #%% export
# links.to_file('trb2023/network.gpkg',layer='links',driver='GPKG')
# improvements.to_file('trb2023/network.gpkg',layer='imp_links',driver='GPKG')
# nodes.to_file('trb2023/network.gpkg',layer='nodes',driver='GPKG')
   