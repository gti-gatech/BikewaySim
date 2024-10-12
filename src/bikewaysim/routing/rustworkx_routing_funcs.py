import rustworkx as rx
import numpy as np
import pandas as pd
import pickle

from bikewaysim.paths import config

def rx_import_calibration_network(config):
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
    turns1.columns = ['linkid','source_reverse_link']
    turns2.columns = ['linkid','target_reverse_link']
    to_add = pd.concat([turns1,turns2],ignore_index=True).drop_duplicates()
    
    edge_weights = 1 # sets an initial standard weight
    turn_G.add_nodes_from([tuple(x) for x in to_add.values]) # add the nodes
    node_to_idx, _ = rx_conversion_helpers(turn_G) # get the conversion to node indeces
    turn_G.add_edges_from([(node_to_idx[(x[0],x[1])],node_to_idx[(x[2],x[4])],edge_weights) for x in edges_to_add])

    return turn_G

def get_ods_from_match_dict(match_dict,links):
    starts = []
    ends = []
    for tripid, _ in match_dict.items():
        #get start and end linkid
        start = tuple(match_dict[tripid]['edges'].iloc[0,:].values)
        end = tuple(match_dict[tripid]['edges'].iloc[-1,:].values)

        #get start and end node for shortest and impedance routing
        #TODO change this to be live so we don't run into errors when the matching network is different
        start = links.loc[start,'A']
        end = links.loc[end,'B']
        starts.append(int(start))
        ends.append(int(end))

    return starts, ends

def rx_conversion_helpers(G_rx):
    '''
    Caution, this only works if there aren't nodes with duplicate "payloads"
    '''
    # create a dict for mapping back and forth (only valid if we're positive that each node value is unique)
    node_to_idx = dict(zip(G_rx.nodes(),G_rx.node_indices()))
    idx_to_node = dict(zip(G_rx.node_indices(),G_rx.nodes()))

    return node_to_idx, idx_to_node

def add_virtual_edges(starts,ends,link_costs,turns,G_rx):

    # first get the connecting links
    starting_edges = turns.loc[turns['source_A'].isin(set(starts)),['source_A','source_linkid','source_reverse_link']].drop_duplicates().values
    ending_edges = turns.loc[turns['target_B'].isin(set(ends)),['target_linkid','target_reverse_link','target_B']].drop_duplicates().values

    # get tuples of the edges we need to add
    starting_edges = [(int(x[0]),(int(x[1]),bool(x[2])),link_costs.get((x[1],x[2]))) for x in starting_edges]
    ending_edges = [((int(x[0]),bool(x[1])),int(x[2]),0) for x in ending_edges]

    # this makes sure we don't add any duplicate nodes to the graph
    new_nodes = list(set(starts + ends) - set(G_rx.nodes()))
    added_nodes = G_rx.add_nodes_from(new_nodes)

    node_to_idx, idx_to_node = rx_conversion_helpers(G_rx)

    # get idx edges
    starting_virtual_edges = [(node_to_idx[a],node_to_idx[b],weight) for a, b, weight in starting_edges] 
    ending_virtual_edges = [(node_to_idx[a],node_to_idx[b],weight) for a, b, weight in ending_edges] 

    # add these to graph
    added_edges = G_rx.add_edges_from(starting_virtual_edges + ending_virtual_edges)

    return added_nodes, added_edges

def rx_shortest_paths(ods,G_rx):
    '''
    Takes in a list of origin and destination nodes and find the shortest path according
    to the set impedance.

    rustworkx version needs two dicts for converting back and forth between the node indces of the network graph
    and the (linkid,reverse_link) structure used in the 
    '''
    node_to_idx, idx_to_node = rx_conversion_helpers(G_rx)
    ods = [(node_to_idx[start],node_to_idx[end]) for start, end in ods]
    results_rx = [rx.dijkstra_shortest_paths(G_rx,start,end,weight_fn=lambda x: x['weight']) for start, end in ods]
    shortest_paths = [[idx_to_node[x] for x in i[1:-1]] for sublist in results_rx for i in sublist.values()]
    path_lengths = [np.sum([G_rx.get_edge_data(x,y)['weight'] for x,y in list(zip(unpacked,unpacked[1:]))]) for path in results_rx for unpacked in [list(z) for z in path.values()]]
    
    return path_lengths, shortest_paths

def impedance_update(betas:np.array,betas_tup:tuple,
                     link_impedance_function:function,
                     base_impedance_col:str,
                     turn_impedance_function:function,
                     links:pd.DataFrame,turns:pd.DataFrame,
                     turn_G:rx.PyDiGraph,added_nodes):
    '''
    This function takes in the betas, impedance functions, and network objects
    and updates the network graph accordingly.

    Need to think about how to incorporate infrastructure availability into this
    '''
    
    #update link costs
    link_impedance_function(betas, betas_tup, links, base_impedance_col)
    
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
    updated_edge_costs = [(node_to_idx[x[0]],node_to_idx[x[1]],{'weight':x[2]}) for x in updated_edge_costs] 

    # update the added 

    # updates the edges
    _ = [turn_G.update_edge(*x) for x in updated_edge_costs]
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
    updated_edge_costs = [(node_to_idx[x[0]],node_to_idx[x[1]],{'weight':x[2]}) for x in updated_edge_costs] 

#figure out how to use the node indeces to update the costs for the starting connecting links