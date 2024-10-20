import rustworkx as rx
import numpy as np
import pandas as pd
import pickle

from bikewaysim.paths import config

'''
Contains all of the functions needed for shortest path routing and handling the turn network

'''

def import_calibration_network(config):
    '''
    Backend function for loading calibration network files
    '''
    # import the calibration network
    with (config['calibration_fp']/"calibration_network.pkl").open('rb') as fh:
        links,turns = pickle.load(fh)
    # make the length and geo dict
    length_dict = dict(zip(links['linkid'],links.length))
    geo_dict = dict(zip(links['linkid'],links.geometry))
    # form turn graph
    turn_G = make_turn_graph(turns)
    return links, turns, length_dict, geo_dict, turn_G

def make_turn_graph(turns):
    '''
    Takes in the turns dataframe and returns a rustworkx network
    '''

    turn_G = rx.PyDiGraph()
    
    edges_to_add = turns[['source_linkid','source_reverse_link','target_linkid','target_reverse_link']].values
    turns1 = turns[['source_linkid','source_reverse_link']]
    turns2 = turns[['target_linkid','target_reverse_link']]
    turns1.columns = ['linkid','reverse_link']
    turns2.columns = ['linkid','reverse_link']
    to_add = pd.concat([turns1,turns2],ignore_index=True,axis=0).drop_duplicates()
    
    edge_weights = 1 # sets an initial standard weight
    turn_G.add_nodes_from([tuple(x) for x in to_add.values]) # add the nodes
    node_to_idx, _ = rx_conversion_helpers(turn_G) # get the conversion to node indeces
    edges_to_add = [(node_to_idx[(x[0],x[1])],node_to_idx[(x[2],x[3])],edge_weights) for x in edges_to_add]
    turn_G.add_edges_from(edges_to_add)

    return turn_G

def rx_conversion_helpers(G_rx):
    '''
    NOTE: regenerate this everytime the nodes are added/removed from the graph
    ONLY valid if data node payloads are unique (they don't have to be in rustworkx)
    '''
    # create a dict for mapping back and forth (only valid if we're positive that each node value is unique)
    node_to_idx = dict(zip(G_rx.nodes(),G_rx.node_indices()))
    idx_to_node = dict(zip(G_rx.node_indices(),G_rx.nodes()))

    return node_to_idx, idx_to_node

def add_virtual_edges(starts,ends,links,turns,G_rx):

    #make dict of link costs
    link_costs = links[['linkid','reverse_link','link_cost']].set_index(['linkid','reverse_link']).squeeze().to_dict()

    # first get the connecting links
    starting_edges = turns.loc[turns['source_A'].isin(set(starts)),['source_A','source_linkid','source_reverse_link']].drop_duplicates().values
    ending_edges = turns.loc[turns['target_B'].isin(set(ends)),['target_linkid','target_reverse_link','target_B']].drop_duplicates().values

    # get tuples of the edges we need to add
    starting_edges = [(int(x[0]),(int(x[1]),bool(x[2])),link_costs.get((x[1],x[2]))) for x in starting_edges]
    ending_edges = [((int(x[0]),bool(x[1])),int(x[2]),0) for x in ending_edges]

    # this makes sure we don't add any duplicate nodes to the graph
    new_nodes = list(set(starts + ends) - set(G_rx.nodes()))
    added_nodes = G_rx.add_nodes_from(new_nodes)

    node_to_idx, _ = rx_conversion_helpers(G_rx)

    # get idx edges
    starting_virtual_edges = [(node_to_idx[a],node_to_idx[b],weight) for a, b, weight in starting_edges] 
    ending_virtual_edges = [(node_to_idx[a],node_to_idx[b],weight) for a, b, weight in ending_edges] 

    # add these to graph
    G_rx.add_edges_from(starting_virtual_edges + ending_virtual_edges)

    return added_nodes # use this to remove virtual edges later on

def remove_virtual_links(added_nodes,G_rx):
    '''Removes the virtual links added'''
    G_rx.remove_nodes_from(added_nodes)

def rx_shortest_paths(ods,G_rx):
    '''
    Takes in a list of origin and destination nodes and find the shortest path according
    to the set impedance. Can work on a set of ods or one at a time.

    rustworkx version needs two dicts for converting back and forth between the node indces of the network graph
    and the (linkid,reverse_link) structure used in the 
    '''
    
    node_to_idx, idx_to_node = rx_conversion_helpers(G_rx)

    if isinstance(ods,tuple): # if there is only one od provided, convert to list
        ods = [ods]

    ods = [(node_to_idx[start],node_to_idx[end]) for start, end in ods]
    results_rx = [rx.dijkstra_shortest_paths(G_rx,start,end,weight_fn=float) for start, end in ods]
    shortest_paths = [[idx_to_node[x] for x in i[1:-1] if isinstance(idx_to_node[x],tuple)] for sublist in results_rx for i in sublist.values()]
    path_lengths = [np.sum([G_rx.get_edge_data(x,y) for x,y in list(zip(unpacked,unpacked[1:]))]) for path in results_rx for unpacked in [list(z) for z in path.values()]]
    
    return path_lengths, shortest_paths # these are lists of the results

def rx_shortest_paths_year(ods,G_rx,G_rx_dict):
    '''
    Takes in a list of origin and destination nodes and find the shortest path according
    to the set impedance. Can work on a set of ods or one at a time. This version differese in that
    it takes a dict of rustworkx pydigraphs that represent the network at different years.

    rustworkx version needs two dicts for converting back and forth between the node indces of the network graph
    and the (linkid,reverse_link) structure used in the 
    '''
    
    node_to_idx, idx_to_node = rx_conversion_helpers(G_rx)

    if isinstance(ods,tuple): # if there is only one od provided, convert to list
        ods = [ods]

    # converts ods from osm node ids to nodeindeces used in rustworkx
    ods = [(node_to_idx[start],node_to_idx[end],year) for start, end, year in ods]
    # find the shortest path for each od given the year
    results_rx = [rx.dijkstra_shortest_paths(G_rx_dict[year],start,end,weight_fn=float) for start, end, year in ods]
    # coerces the data into a list of the route impedances and the edge list for the route taken
    shortest_paths = [[idx_to_node[x] for x in i[1:-1] if isinstance(idx_to_node[x],tuple)] for sublist in results_rx for i in sublist.values()]
    path_lengths = [np.sum([G_rx_dict[year[2]].get_edge_data(xy[0],xy[1]) for xy,year in zip(list(zip(unpacked,unpacked[1:])),ods)]) for path in results_rx for unpacked in [list(z) for z in path.values()]]
    
    return path_lengths, shortest_paths # these are lists of the results

def create_year_networks(
        betas,
        betas_tup,
        starts,
        ends,
        turn_G,
        links,
        turns,
        set_to_zero,
        set_to_inf,
        years,
        link_impedance_function,
        turn_impedance_function,
        base_impedance_col,
        base_link_col
    ):
    '''
    Creates a dict of PyDigraphs based on the year
    '''

    year_networks = {}
    for year in years:
        # create a copy of the network to modify
        turn_G_copy = turn_G.copy()
        
        # if infra is on street (i.e., the link is still traversable but the impedance doesn't apply)
        links.loc[links['year'] > year,set_to_zero] = 0 
        # if it's off-street then assign it a very high cost
        links.loc[(links['year'] > year) & (links.loc[:,set_to_inf]==1).any(axis=1),'link_cost_override'] = True

        # run network update
        updated_edge_costs = impedance_update(betas,betas_tup,
                link_impedance_function,
                base_impedance_col,
                base_link_col,
                turn_impedance_function,
                links,turns,turn_G_copy)
        # just incase a negative link cost appears as a result
        if isinstance(updated_edge_costs,bool):
            return None
        
        # re-add virtual links
        add_virtual_edges(starts,ends,links,turns,turn_G_copy)
        
        # add the network to the dict
        year_networks[year] = turn_G_copy
    
    return year_networks

def impedance_update(betas:np.array,betas_tup:tuple,
                     link_impedance_function,
                     base_impedance_col:str,
                     base_link_col:str,
                     turn_impedance_function,
                     links:pd.DataFrame,turns:pd.DataFrame,
                     turn_G:rx.PyDiGraph):
    '''
    This function takes in the betas, impedance functions, and network objects
    and updates the network graph accordingly.

    Need to think about how to incorporate infrastructure availability into this
    '''
    
    #update link costs using the link impedance function
    link_impedance_function(betas, betas_tup, links, base_impedance_col, base_link_col) # would get an optional base impedance col
    
    # override the cost with 9e9 if future off-street facility
    # this effectively prevents routing w/o messing around with the network structure
    links.loc[links['link_cost_override']==True,'link_cost'] = 9e9

    #create cost dict
    tuple_index = tuple(zip(links['linkid'],links['reverse_link']))
    cost_dict = dict(zip(tuple_index,links['link_cost']))
    
    #costs are stored in the turn graph (only target matters, initial link cost is added during routing)
    turns['target_link_cost'] = turns[['target_linkid','target_reverse_link']].apply(lambda x: cost_dict.get(tuple(x.values),False),axis=1)

    #update turn costs
    turn_impedance_function(betas, betas_tup, turns)

    #cacluate new total cost
    # TODO round these values because they don't need this many decimal points (usually in travel time)
    turns['total_cost'] = (turns['target_link_cost'] + turns['turn_cost'])

    if turns['total_cost'].isna().any():
        raise Exception("There are nan edge costs, exiting")

    #check for negative link impedance
    if (links['link_cost'] < 0).any() | (turns['total_cost'] < 0).any():
        return False

    node_to_idx, _ = rx_conversion_helpers(turn_G)

    #update turn network graph with final cost
    cols = ['source_linkid','source_reverse_link','target_linkid','target_reverse_link','total_cost']
    updated_edge_costs = [((row[0],row[1]),(row[2],row[3]),row[4]) for row in turns[cols].values]
    updated_edge_costs = [(node_to_idx[x[0]],node_to_idx[x[1]],x[2]) for x in updated_edge_costs] 

    # updates the edges
    _ = [turn_G.update_edge(*x) for x in updated_edge_costs]

    # TODO add virtual links

    return updated_edge_costs

def back_to_base_impedance(link_impedance_col,links,turns,turn_G):
    '''
    This function reverts the network graph back to base impedance (distance or travel time)
    with all turns as 0 cost
    '''
    #update link costs
    links['link_cost'] = links[link_impedance_col]
    cost_dict = dict(zip(links['linkid'],links['link_cost']))
    turns['target_link_cost'] = turns['target_linkid'].map(cost_dict)
    
    #cacluate new total cost
    turns['total_cost'] =  turns['target_link_cost']

    node_to_idx, _ = rx_conversion_helpers(turn_G)

    #update turn network graph with final cost
    cols = ['source_linkid','source_reverse_link','target_linkid','target_reverse_link','total_cost']
    updated_edge_costs = [((row[0],row[1]),(row[2],row[3]),row[4]) for row in turns[cols].values]
    updated_edge_costs = [(node_to_idx[x[0]],node_to_idx[x[1]],x[2]) for x in updated_edge_costs] 

#figure out how to use the node indeces to update the costs for the starting connecting links

