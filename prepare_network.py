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


user_directory = os.fspath(Path.home()) #get home directory and convert to path string
file_directory = r"/Documents/BikewaySimData" #directory of bikewaysim network processing code

#change this to where you stored this folder
os.chdir(user_directory+file_directory)

#%%

# def bws_network(links,nodes, oneway_col, improvement = None):
#     #read in data
#     links = gpd.read_file(linksfp)
#     nodes = gpd.read_file(nodesfp)

#     #get attributes (if needed)

#     #create nodes file
#     nodes = create_bws_nodes(nodes)
    
#     #filter links
#     links = create_bws_links(links)

#     #calculate distances
#     links = get_link_costs_osm(links,improvement)
    
#     #create reverse links for two way streets
#     links = create_reverse_links(links, oneway_col)
    
#     return links, nodes
    
    
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

    #filter to those that are two way
    #links_rev = links_rev[(links_rev['oneway'] != 1)]

    #add to other links
    links = links.append(links_rev).reset_index(drop=True)

    #make A_B col
    links['A_B'] = links['A'] + '_' + links['B']

    return links

def largest_comp(links,nodes): 
    #create undirected graph
    G = nx.Graph()  # create directed graph
    for ind, row2 in links.iterrows():
        # forward graph, time stored as minutes
        G.add_edges_from([(str(row2['A']), str(row2['B']))])
                                    #weight='forward', name=row2['name'])
    
        #only keep largest component
        largest_cc = max(nx.connected_components(G), key=len)
    
    #get nodes
    nodes = nodes[nodes['N'].isin(largest_cc)]
    
    #use nodes to get links
    links = links[links['A'].isin(nodes['N']) & links['B'].isin(nodes['N'])]
    
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

# def improvement_costs(links, improvement):
#     #add a protected bike lane
#     links.loc[links['A_B'].isin(improvements),'pbl'] = 1 
    
#     links['imp_dist'] = link_costs(links)
    
#     return links

# def add_improvements(links):
#     #bring in new features
#     new_features = gpd.read_file('trb2023/network_improvements.gpkg',layer='new')
    
#     #bring in changes
#     changes = gpd.read_file('trb2023/network_improvements.gpkg',layer='changes')

#     #


#%% bring in links

fp = 'processed_shapefiles/conflation/finalized_networks/here_trb.gpkg'
links = gpd.read_file(fp,layer='links')
fp = 'processed_shapefiles/here/here_bikewaysim_network.gpkg'
nodes = gpd.read_file(fp,layer='roadbike_nodes')

#create nodes file
nodes = create_bws_nodes(nodes)

#filter links
links = create_bws_links(links)

#create reverse streets
links = create_reverse_links(links)

#drop unconnected links
links,nodes = largest_comp(links,nodes)

#calculate link costs
links['per_dist'] = link_costs(links)

#make improvements
improvements = links.copy()
changes = '10TH ST NW'
#add mu
improvements.loc[improvements['ST_NAME']==changes,'mu'] = 1

#bring in new feature
new = gpd.read_file('trb2023/network_improvements.gpkg',layer='new')

#make reverse
new = create_reverse_links(new)

#add to network
improvements = improvements.append(new)

#add impr cost
improvements['imp_dist'] = link_costs(improvements)

#%% export
links.to_file('trb2023/network.gpkg',layer='links',driver='GPKG')
improvements.to_file('trb2023/network.gpkg',layer='imp_links',driver='GPKG')
nodes.to_file('trb2023/network.gpkg',layer='nodes',driver='GPKG')
   