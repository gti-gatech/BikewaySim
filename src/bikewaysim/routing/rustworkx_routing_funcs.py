import rustworkx as rx
import numpy as np


def make_turn_graph(turns):
    '''
    Takes in the turns dataframe and returns a rustworkx network
    '''

    # TODO later, focus on getting things up
    turn_G = rx.PyDiGraph()
    
    turns = turns[['source_linkid','source_reverse_link','target_linkid','target_reverse_link']].drop_duplicates().values
    edge_weights = 1 # sets an initial standard weight

    start_nodes = [tuple(x) for x in turns[['source_linkid','source_reverse_link']].drop_duplicates().values]
    end_nodes = [turns[['source_linkid','source_reverse_link']].drop_duplicates().values



    added_nodes = G_rx.add_nodes_from(new_nodes)

    node_to_idx, idx_to_node = rx_conversion_helpers(G_rx)

    # get idx edges
    starting_virtual_edges = [(node_to_idx[a],node_to_idx[b],{'weight':weight}) for a, b, weight in starting_edges] 
    ending_virtual_edges = [(node_to_idx[a],node_to_idx[b],{'weight':weight}) for a, b, weight in ending_edges] 




    turn_G.

    turn_G.add_edges_from([((row[0],row[1]),(row[2],row[3]),edge_weights) for row in turns]) 

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
    Caution, this as
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
    starting_virtual_edges = [(node_to_idx[a],node_to_idx[b],{'weight':weight}) for a, b, weight in starting_edges] 
    ending_virtual_edges = [(node_to_idx[a],node_to_idx[b],{'weight':weight}) for a, b, weight in ending_edges] 

    # add these to graph
    G_rx.add_edges_from(starting_virtual_edges + ending_virtual_edges)

    return node_to_idx, idx_to_node, added_nodes



def rx_shortest_paths(ods,node_to_idx:dict,idx_to_node:dict,G_rx):
    '''
    Takes in a list of origin and destination nodes and find the shortest path according
    to the set impedance.

    rustworkx version needs two dicts for converting back and forth between the node indces of the network graph
    and the (linkid,reverse_link) structure used in the 
    '''

    ods = [(node_to_idx[start],node_to_idx[end]) for start, end in ods]
    results_rx = [rx.dijkstra_shortest_paths(G_rx,start,end,weight_fn=lambda x: x['weight']) for start, end in ods]
    shortest_paths = [[idx_to_node[x] for x in i[1:-1]] for sublist in results_rx for i in sublist.values()]
    path_lengths = [np.sum([G_rx.get_edge_data(x,y)['weight'] for x,y in list(zip(unpacked,unpacked[1:]))]) for path in results_rx for unpacked in [list(z) for z in path.values()]]
    
    return path_lengths, shortest_paths

