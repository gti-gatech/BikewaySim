# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 05:41:07 2022

@author: tpassmore6
"""
import geopandas as gpd
import pandas as pd
import networkx as nx
import time

def prepare_network(links:gpd.GeoDataFrame,nodes:gpd.GeoDataFrame,spd_mph:float,prevent_wrongway:bool=True):
    '''
    This function takes in a links and nodes geodataframe and formats it into
    a routable network graph for use in BikewaySim. The nodes geodataframe must
    have an ID column wiht the suffix "_ID". The links geodataframe must have
    columns specifying the starting node and ending node with the suffixes 
    "_A" and "_B" specifying start and end node ID respectively.

    The links file is then reduced to the largest connected network and reverse
    links are created. The length of the links is calculated and added to a "dist"
    column. Then the link costs are calculated based on the attributes specificed in
    the costs dictionary. The keys of that dictionary must correspond to the columns of
    the links geodataframe.

    '''
    
    #find code to remove interstitial nodes to reduce network size
    
    #prepare nodes
    nodes = create_bws_nodes(nodes)

    #prepare links
    links = create_bws_links(links)

    #drop unconnected links and (remove interstitial nodes future)
    links,nodes = largest_comp_and_simplify(links,nodes)

    #create reverse streets
    #links = create_reverse_links(links)

    #calculate distance for distance/travel time based impedance
    links['dist'] = links.length
    links['mins'] = links['dist'] / 5280 / spd_mph * 60

    #round
    links['dist'] = links['dist'].round(2)
    links['mins'] = links['mins'].round(2)
    
    return links, nodes

def create_bws_links(links):
    #rename ID column
    cols = links.columns.tolist()
    
    a_cols = [a_cols for a_cols in cols if ("_A" in a_cols)]
    b_cols = [b_cols for b_cols in cols if ("_B" in b_cols)]
    
    #warn if more than one column
    if (len(a_cols) > 1) | (len(b_cols) > 1):
        print('warning, more than one id column present')
    
    #replace with desired hierarchy
    links = links.rename(columns={a_cols[0]:'A'})
    links = links.rename(columns={b_cols[0]:'B'})
    
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
    id_cols = [id_cols for id_cols in id_cols if "_N" in id_cols]

    #warn if more than one column
    if (len(id_cols) > 1):
        print('warning, more than one id column present')
    
    #replace with desired hierarchy
    nodes = nodes.rename(columns={id_cols[0]:'N'})
    
    #filter to needed columns
    nodes = nodes[['N','X','Y','lon','lat','geometry']]
    
    return nodes

def create_reverse_links(links,allow_wrongway=False):
    '''
    This code creates a column that indicates the direction of links.

    Only works for OSM right now

    When a network graph is made, this column can be used to filter out
    wrongway travel if desired.
    '''

    links['wrongway'] = False

    if allow_wrongway:
        links_rev = links.copy().rename(columns={'A':'B','B':'A'})
        links_rev.loc[links['oneway']==True] = True
    else:
        links_rev = links[links['oneway']==False].copy().rename(columns={'A':'B','B':'A'})

    #add to other links
    links = pd.concat([links,links_rev],ignore_index=True)

    return links

def largest_comp_and_simplify(links,nodes,A='A',B='B',N='N'): 
    
    print('Before connected components: Links',links.shape[0],'Nodes',nodes.shape[0])
    # #optional arguement
    # if net_name is not None:
    #     A = net_name + '_A'
    #     B = net_name + '_B'
    #     N = net_name + '_N'
    # else:
    #     A = 'A'
    #     B = 'B'
    #     N = 'N'
    
    #create undirected graph
    G = nx.Graph()  # create directed graph
    for row in links[[A,B]].itertuples(index=False):
        # forward graph, time stored as minutes
        G.add_edges_from([(row[0],row[1])])

    #only keep largest component
    largest_cc = max(nx.connected_components(G), key=len)

    #TODO future project simplify graph connect links connect with nodes of degree 2
    #S = G.subgraph(largest_cc).copy()
    #simple_graph = ox.simplification.simplify_graph(S)
    #there is a contract function too
    
    #get nodes
    nodes = nodes[nodes[N].isin(largest_cc)]
    #get links
    links = links[links[A].isin(largest_cc) & links[B].isin(largest_cc)]
    
    print('After connected components: Links',links.shape[0],'Nodes',nodes.shape[0])
    return links,nodes
    
def link_costs(links:pd.DataFrame(),costs:dict,imp_name:str):
    
    #get list of columns names to use
    cols = list(costs.keys())
    
    #initalize a impedance multiply factor (sum of all coefficients times attribute values)
    links['imp_factor'] = 1

    # if row is a mup then set all other column values to zero
    cols.remove('mu')
    links.loc[links['mu']==1,cols] = 0
    cols.append('mu')

    for col in cols:
        # calculate impedance
        links['imp_factor'] = links['imp_factor'] + (links[col] * costs[col])
    
    links[imp_name] = links['mins'] * links['imp_factor']
    
    #check for negative impedances
    if (links[imp_name] < 0).any():
        print('Warning: negative link impedance present!')

    return links

def apply_costs(links,cost_dicts,export_fp):
    return

def add_ref_ids_plain(links,nodes):
    '''
    This function adds reference columns to links from the nodes id column.
    Assumes node columns are N, A, B whereas add_ref_ids uses the network name.
    '''
    for_matching = links.copy()
    #make the first point the active geometry
    for_matching['pt_geometry'] = for_matching.apply(start_node_geo, geom='geometry', axis=1)
    for_matching.set_geometry('pt_geometry',inplace=True)
    for_matching.drop(columns='geometry',inplace=True)
    #find nearest node from starting node and add to column
    links['A'] = ckdnearest(for_matching,nodes,return_dist=False)['N']
    
    #repeat for end point
    for_matching = links.copy()
    #make the first point the active geometry
    for_matching['pt_geometry'] = for_matching.apply(end_node_geo, geom='geometry', axis=1)
    for_matching.set_geometry('pt_geometry',inplace=True)
    for_matching.drop(columns='geometry',inplace=True)
    #find nearest node from starting node and add to column
    links['B'] = ckdnearest(for_matching,nodes,return_dist=False)['N']

    #check for missing reference ids
    if links['A'].isnull().any() | links['B'].isnull().any():
        print("There are missing reference ids")
    else:
        print("Reference IDs successfully added to links.")
    return links