# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 05:41:07 2022

@author: tpassmore6
"""
import geopandas as gpd
import networkx as nx
import osmnx as ox
import time

def prepare_network(links:gpd.GeoDataFrame,nodes:gpd.GeoDataFrame,costs:dict,imp_name:str):
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

    #record the starting time
    time_start = time.time()
    
    #TODO group by attributes and get list of edges with identical attributes (for network reduction/simplification)
    
    #prepare nodes
    nodes = create_bws_nodes(nodes)

    #prepare links
    links = create_bws_links(links)

    #drop unconnected links and remove interstitial nodes
    links,nodes = largest_comp_and_simplify(links,nodes)

    #create reverse streets
    links = create_reverse_links(links)

    #calculate distance for distance/travel time based impedance
    links['dist'] = links.length
    
    #calculate attribute based impedance
    links = link_costs(links,costs,imp_name)
        
    print(f'Took {round(((time.time() - time_start)/60), 2)} minutes to prepare the network.')
    
    return links, nodes

def create_bws_links(links):
    #rename ID column
    a_cols = links.columns.tolist()
    a_cols = [a_cols for a_cols in a_cols if "_A" in a_cols]
    
    b_cols = links.columns.tolist()
    b_cols = [b_cols for b_cols in b_cols if "_B" in b_cols]
    
    a_b_cols = links.columns.tolist()
    a_b_cols = [a_b_cols for a_b_cols in a_b_cols if "A_B" in a_b_cols]
    
    #warn if more than one column
    if (len(a_cols) > 1) | (len(b_cols) > 1):
        print('warning, more than one id column present')
    
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

    #warn if more than one column
    if (len(a_cols) > 1) | (len(b_cols) > 1):
        print('warning, more than one id column present')
    
    #replace with desired hierarchy
    nodes = nodes.rename(columns={id_cols[0]:'N'})
    
    #filter to needed columns
    nodes = nodes[['N','X','Y','lon','lat','geometry']]
    
    return nodes

def create_reverse_links(links):
    
    #flip start and end node ids
    links_rev = links.rename(columns={'A':'B','B':'A'})

    try:
        #newwrongway
        links_rev['newwrongway'] = 0
    
        #set wrongways to rightways and vice versa
        links_rev.loc[links_rev['wrongway']==1,'newwrongway'] = 0
        links_rev.loc[(links_rev['oneway']==1) & (links_rev['wrongway']==0),'newwrongway'] = 1 

        #rename and drop
        links_rev.drop(columns=['wrongway'],inplace=True)
        links_rev.rename(columns={'newwrongway':'wrongway'},inplace=True)
    except:
        print("No 'wrongway' column")

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

    #TODO future project simplify graph connect links connect with nodes of degree 2
    #S = G.subgraph(largest_cc).copy()
    #simple_graph = ox.simplification.simplify_graph(S)
    
    #get nodes
    nodes = nodes[nodes['N'].isin(largest_cc)]
    #get links
    links = links[links['A'].isin(largest_cc) & links['B'].isin(largest_cc)]
    
    return links,nodes
    
def link_costs(links,costs,imp_name:str):
    
    #get list of columns names to use
    cols = list(costs.keys())
    
    #initalize a impedance multiply factor (sum of all coefficients times attribute values)
    links['imp_factor'] = 1

    for col in cols:
        # if row is a mup then set all other column values to zero
        links.loc[links['mu']==1,col] = 0

        # calculate impedance
        links['imp_factor'] = links['imp_factor'] + (links[col] * costs[col])
    
    links['imp'] = links['dist'] * links['imp_factor']
    
    #check for negative impedances
    if (links['imp'] < 0).any():
        print('Warning: negative link impedance present!')

    return links
