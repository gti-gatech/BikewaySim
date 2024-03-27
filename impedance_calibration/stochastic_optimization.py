#https://keurfonluu.github.io/stochopy/api/optimize.html

from pathlib import Path
from stochopy.optimize import minimize
from shapely.ops import nearest_points, Point, MultiLineString
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from tqdm import tqdm
import time
from datetime import timedelta
import ast
import shapely
import sys

sys.path.insert(0,str(Path.cwd().parent))
from network.src import modeling_turns
#import stochastic_optimization.stochastic_optimization as stochastic_optimization

'''
This module is for deriving link costs from matched GPS traces
through the stochastic optimization method used in Schweizer et al. 2020

Required Files:
Network links for shortest path routing
Matched traces

Pre-Algo:
- Create network graph with lengths as original weights
- Create dict with link lengths
- set parameter search range

Algo Steps:

Todo:
- Change the beta array structure to allow for flexible impedance functions
that can be archived after optimization (not sure if this is possible)

- Dict gives the attribute names


'''

# #customize this function to change impedance formula
# #TODO streamline process of trying out new impedance functions
# def link_impedance_function(betas,links):
#     #prevent mutating the original links gdf
#     links = links.copy()
    
#     attr_multiplier = (betas[0]*~links['bike']) + betas[1]*links['up_grade']
#     links['link_cost'] = links['length_ft'] * attr_multiplier
#     return links

# def turn_impedance_function(betas,pseudo_links):
#     #use beta coefficient to calculate turn cost
#     base_turn_cost = 30 # from Lowry et al 2016 DOI: http://dx.doi.org/10.1016/j.tra.2016.02.003
#     turn_costs = {
#         'left': betas[1] * base_turn_cost,
#         'right': betas[1] * base_turn_cost,
#         'straight': betas[1] * base_turn_cost
#     }
#     pseudo_links['turn_cost'] = pseudo_links['turn_type'].map(turn_costs)
#     return pseudo_links

# def impedance_update(df_links,pseudo_links,pseudo_G):
#     '''
#     This function updates the network graph with the correct weights
#     '''

#     return pseudo_G

def objective_function(betas,betas_links,betas_turns,links,
                       pseudo_links,pseudo_G,
                       matched_traces,link_impedance_function,
                       turn_impedance_function,exact,follow_up):

    #prevent negative link weights
    if (betas < 0).any():
        val = 0
        return val

    #TODO bring most of this out as functions
    #use initial/updated betas to calculate link costs
    print('setting link costs')
    links = link_impedance_function(betas, betas_links, links)
    cost_dict = dict(zip(links['linkid'],links['link_cost']))
    
    #TODO is this the best way to accomplish this?
    #get_geo = links.loc[links.groupby(['source','target'])['link_cost'].idxmin(),['source','target','geometry']]

    #add link costs to pseudo_links
    pseudo_links['source_link_cost'] = pseudo_links['source_linkid'].map(cost_dict)
    pseudo_links['target_link_cost'] = pseudo_links['target_linkid'].map(cost_dict)

    #use initial/updated betas to calculate turn costs
    print('setting turn costs')
    pseudo_links = turn_impedance_function(betas, betas_turns, pseudo_links)

    #assign na turns a cost of 0
    #pseudo_links.loc[pseudo_links['turn_cost'].isna(),'turn_cost'] = 0

    #add links and multiply by turn cost
    pseudo_links['total_cost'] = pseudo_links['source_link_cost'] + pseudo_links['target_link_cost'] + pseudo_links['turn_cost']

    #check for negative costs (set them to zero for now?)
    #pseudo_links.loc[pseudo_links['total_cost'] < 0, 'total_cost'] = 1

    #round values
    #pseudo_links['total_cost'] = pseudo_links['total_cost'].round(1)

    #only keep link with the lowest cost
    print('finding lowest cost')
    costs = pseudo_links.set_index(['source','target'])['total_cost']#pseudo_links.groupby(['source','target'])['total_cost'].min()

    # #get linkids used to retreive the correct link geometries for overlap function?
    # #we don't care about direction in this case
    # print('finding link id for lowest cost')
    # source_cols = ['source','source_linkid','source_reverse_link']
    # target_cols = ['target','target_linkid','target_reverse_link']
    # min_links = pseudo_links.loc[pseudo_links.groupby(['source','target'])['total_cost'].idxmin()]
    # source_links = min_links[source_cols]
    # target_links = min_links[target_cols]
    # source_links.columns = ['A_B','linkid','reverse_link']
    # target_links.columns = ['A_B','linkid','reverse_link']
    # #what keeps a_b from not being duplicated here?
    # linkids = pd.concat([source_links,target_links],ignore_index=True).drop_duplicates().set_index('A_B')

    #update edge weights
    print('updating edge weights')
    nx.set_edge_attributes(pseudo_G,values=costs,name='weight')

    #update edge ids (what was this for?)
    
    #do shortest path routing
    shortest_paths = {}
    print(f'Shortest path routing with coefficients: {betas}')    
    for source, targets in matched_traces.groupby('start')['end'].unique().items():

        #add virtual links to pseudo_G
        pseudo_G, virtual_edges = modeling_turns.add_virtual_links(pseudo_links,pseudo_G,source,targets)
        
        #perform shortest path routing for all target nodes from source node
        #(from one to all until target node has been visited)
        for target in targets:  
            #cant be numpy int64 or throws an error
            target = int(target)
            
            try:
                #TODO every result is only a start node, middle node, then end node
                length, node_list = nx.single_source_dijkstra(pseudo_G,source,target,weight='weight')
            except:
                print(source,target)
                length, node_list = nx.single_source_dijkstra(pseudo_G,source,target,weight='weight')

            #get edge list
            edge_list = node_list[1:-1]

            #get geometry from edges
            modeled_edges = links.set_index(['source','target']).loc[edge_list]

            # modeled_edges = links.merge(linkids.loc[edge_list],on=['linkid','reverse_link'],how='inner')
            # modeled_edges = gpd.GeoDataFrame(modeled_edges,geometry='geometry')

            shortest_paths[(source,target)] = {
                'edges': set(modeled_edges['linkid'].tolist()),
                'geometry':MultiLineString(modeled_edges['geometry'].tolist()),#modeled_edges.dissolve()['geometry'].item(),
                'length':MultiLineString(modeled_edges['geometry'].tolist()).length
                }

        #remove virtual links
        pseudo_G = modeling_turns.remove_virtual_edges(pseudo_G,virtual_edges)
    
    print('calculating objective function')

    #turn shortest paths dict to dataframe
    shortest_paths = pd.DataFrame.from_dict(shortest_paths,orient='index')
    shortest_paths.reset_index(inplace=True)
    shortest_paths.columns = ['start','end','linkids','geometry','length']
    #shortest_paths[['start','end']] = shortest_paths['index'].apply(lambda x: pd.Series(x))
    #shortest_paths.drop(columns=['index'],inplace=True)

    #add modeled paths to matched_traces dataframe
    merged = matched_traces.merge(shortest_paths,on=['start','end'],suffixes=(None,'_modeled'))

    if exact:
        sum_all = merged['length'].sum() * 5280
        all_overlap = 0

        for idx, row in merged.iterrows():
            #find shared edges
            chosen_and_shortest = row['linkids_modeled'] & row['linkids']
            #get the lengths of those links
            overlap_length = links.set_index('linkid').loc[list(chosen_and_shortest)]['length_ft'].sum()
            #overlap_length = np.sum([link_lengths.get(link_tup,'error') for link_tup in chosen_and_shortest])
            all_overlap += overlap_length

        #calculate objective function value
        val = all_overlap / sum_all
        print('Exact overlap percent is:',np.round(val*100,1),'%')
    
    #calculate approximate overlap (new approach)
    else:
        #buffer and dissolve generated route and matched route
        buffer_ft = 500

        merged.set_geometry('geometry',inplace=True)
        merged['buffered_geometry'] = merged.buffer(buffer_ft)
        merged.set_geometry('buffered_geometry',inplace=True)
        merged['area'] = merged.area

        merged.set_geometry('geometry_modeled',inplace=True)
        merged['buffered_geometry_modeled'] = merged.buffer(buffer_ft)
        merged.set_geometry('buffered_geometry_modeled',inplace=True)
        merged['area_modeled'] = merged.area

        #for each row find intersection between buffered features
        merged['intersection'] = merged.apply(lambda row: row['buffered_geometry'].intersection(row['buffered_geometry_modeled']), axis=1)

        # merged['intersection'] = merged.apply(
        #     lambda row: shapely.intersection(row['buffered_geometry'],row['buffered_geometry_modeled']))
        merged.set_geometry('intersection',inplace=True)
        merged['intersection_area'] = merged.area

        #find the overlap with the total area (not including intersections)
        #if the modeled/chosen links are different, then overlap decreases
        #punishes cirquitious modeled routes that utilize every link in the chosen one but include extraneous ones
        merged['overlap'] = merged['intersection_area'] / (merged['area_modeled'] + merged['area'] - merged['intersection_area'])

        #find average overlap (using median to reduce impact of outliers?)
        val = merged['overlap'].median()
        print('Median overlap percent is:',np.round(val*100,1),'%')
    
    if follow_up:
        return merged

    return -val#, merged

def follow_up(betas,links,pseudo_links,pseudo_G,matched_traces,exact=False):

    #
    modeled_trips = {}
    
    #use initial/updated betas to calculate link costs
    links = link_impedance_function(betas, links)
    cost_dict = dict(zip(links['linkid'],links['link_cost']))
    
    #TODO is this the best way to accomplish this?
    #get_geo = links.loc[links.groupby(['source','target'])['link_cost'].idxmin(),['source','target','geometry']]

    #add link costs to pseudo_links
    pseudo_links['source_link_cost'] = pseudo_links['source_linkid'].map(cost_dict)
    pseudo_links['target_link_cost'] = pseudo_links['target_linkid'].map(cost_dict)

    #use initial/updated betas to calculate turn costs
    pseudo_links = turn_impedance_function(betas, pseudo_links)

    #assign na turns a cost of 0
    pseudo_links.loc[pseudo_links['turn_cost'].isna(),'turn_cost'] = 0

    #add links and multiply by turn cost
    pseudo_links['total_cost'] = pseudo_links['source_link_cost'] + pseudo_links['target_link_cost'] + pseudo_links['turn_cost']

    #only keep link with the lowest cost
    costs = pseudo_links.groupby(['source','target'])['total_cost'].min()

    #get linkids used
    source_cols = ['source','source_linkid','source_reverse_link']
    target_cols = ['target','target_linkid','target_reverse_link']
    min_links = pseudo_links.loc[pseudo_links.groupby(['source','target'])['total_cost'].idxmin()]
    source_links = min_links[source_cols]
    target_links = min_links[target_cols]
    source_links.columns = ['A_B','linkid','reverse_link']
    target_links.columns = ['A_B','linkid','reverse_link']
    linkids = pd.concat([source_links,target_links],ignore_index=True).drop_duplicates().set_index('A_B')

    #update edge weights
    nx.set_edge_attributes(pseudo_G,values=costs,name='weight')
    
    #do shortest path routing
    shortest_paths = {}
    print(f'Shortest path routing with coefficients: {betas}')    
    for source, targets in matched_traces.groupby('start')['end'].unique().items():

        #add virtual links to pseudo_G
        pseudo_G, virtual_edges = modeling_turns.add_virtual_links(pseudo_links,pseudo_G,source,targets)
        
        #perform shortest path routing for all target nodes from source node
        #(from one to all until target node has been visited)
        for target in targets:  
            #cant be numpy int64 or throws an error
            target = int(target)
            
            try:
                #TODO every result is only a start node, middle node, then end node
                length, node_list = nx.single_source_dijkstra(pseudo_G,source,target,weight='weight')
            except:
                print(source,target)
                length, node_list = nx.single_source_dijkstra(pseudo_G,source,target,weight='weight')

            #get edge list
            edge_list = node_list[1:-1]

            #get geometry from edges
            modeled_edges = links.merge(linkids.loc[edge_list],on=['linkid','reverse_link'],how='inner')
            modeled_edges = gpd.GeoDataFrame(modeled_edges,geometry='geometry')
            
            #
            shortest_paths[(source,target)] = {
                'edges': set(modeled_edges['linkid'].tolist()),
                'geometry':MultiLineString(modeled_edges['geometry'].tolist()),
                'length':modeled_edges.length.sum()
                }

        #remove virtual links
        pseudo_G = modeling_turns.remove_virtual_edges(pseudo_G,virtual_edges)
    
    #turn shortest paths dict to dataframe
    shortest_paths = pd.DataFrame.from_dict(shortest_paths,orient='index')
    shortest_paths.reset_index(inplace=True)
    shortest_paths.columns = ['start','end','linkids','geometry','length']
    #shortest_paths[['start','end']] = shortest_paths['index'].apply(lambda x: pd.Series(x))
    #shortest_paths.drop(columns=['index'],inplace=True)

    #add modeled paths to matched_traces dataframe
    merged = matched_traces.merge(shortest_paths,on=['start','end'],suffixes=(None,'_modeled'))

    if exact:
        sum_all = merged['length'].sum() * 5280
        all_overlap = 0

        for idx, row in merged.iterrows():
            #find shared edges
            chosen_and_shortest = row['linkids_modeled'] & row['linkids']
            #get the lengths of those links
            overlap_length = links.set_index('linkid').loc[list(chosen_and_shortest)]['length_ft'].sum()
            #overlap_length = np.sum([link_lengths.get(link_tup,'error') for link_tup in chosen_and_shortest])
            all_overlap += overlap_length

        #calculate objective function value
        val = all_overlap / sum_all
        print('Exact overlap percent is:',np.round(val*100,1),'%')
    
    #calculate approximate overlap (new approach)
    else:
        #buffer and dissolve generated route and matched route
        buffer_ft = 500

        merged.set_geometry('geometry',inplace=True)
        merged['buffered_geometry'] = merged.buffer(buffer_ft)
        merged.set_geometry('buffered_geometry',inplace=True)
        merged['area'] = merged.area

        merged.set_geometry('geometry_modeled',inplace=True)
        merged['buffered_geometry_modeled'] = merged.buffer(buffer_ft)
        merged.set_geometry('buffered_geometry_modeled',inplace=True)
        merged['area_modeled'] = merged.area

        #for each row find intersection between buffered features
        merged['intersection'] = merged.apply(lambda row: row['buffered_geometry'].intersection(row['buffered_geometry_modeled']), axis=1)

        # merged['intersection'] = merged.apply(
        #     lambda row: shapely.intersection(row['buffered_geometry'],row['buffered_geometry_modeled']))
        merged.set_geometry('intersection',inplace=True)
        merged['intersection_area'] = merged.area

        #find the overlap with the total area (not including intersections)
        #if the modeled/chosen links are different, then overlap decreases
        #punishes cirquitious modeled routes that utilize every link in the chosen one but include extraneous ones
        merged['overlap'] = merged['intersection_area'] / (merged['area_modeled'] + merged['area'] - merged['intersection_area'])

        #find average overlap (using median to reduce impact of outliers?)
        val = merged['overlap'].median()
        print('Median overlap percent is:',np.round(val*100,1),'%')
    
    return -val#, merged         



        # #calculate approximate overlap
        # all_overlap = 0
        # for idx, row in trips_df.iterrows():

        #     #buffer and dissolve generated route and matched route
        #     modeled_edges = shortest_paths[row['od']]['edge_list']
        #     chosen_edges = matched_traces[row['tripid']]['matched_trip']
            
        #     #grab links
        #     links.index = list(zip(links['A'],links['B']))
        #     modeled_edges = links.loc[modeled_edges]

        #     #caluclate absolute difference
        #     difference_ft = (modeled_edges.length.sum() - chosen_edges.legnth.sum()).abs()

        #     #buffer edges and dissolve
        #     buffer_ft = 500
        #     modeled_edges_dissolved = modeled_edges.buffer(buffer_ft).dissolve()
        #     chosen_edges_dissovled = chosen_edges.buffer(buffer_ft).dissolve()

        #     #intersect
        #     intersect = gpd.overlay(modeled_edges_dissolved,chosen_edges_dissovled,how='intersection')
        #     overlap = intersect.area / chosen_edges.area
            
        #     #exponent for difference in lengths between generated route and matched route
        #     #as absolute difference in length increases, overlap gets smaller
        #     all_overlap += overlap ** difference_ft

        # duration = timedelta(seconds=time.perf_counter()-start_time)
        # durations.append(duration)
        # start_time = time.perf_counter()



    # print('Overlap =', val)




# #import segment to use
# segment_filepaths = list((fp/'segments').glob('*'))
# results = {}

# def replace_missing_node(row,trips_df,pseudo_G,nodes):                
#     source = row['od'][0]
#     target = row['od'][1]
    
#     #source column
#     if ~pseudo_G.has_node(source):
#         start_coord = row[['start_lat','start_lon']]
#         start_coord['geometry'] = Point(start_coord.iloc[0,1],start_coord.iloc[0,0])
#         start_coord = gpd.GeoDataFrame(start_coord,geometry='geometry',crs='epsg:4326')
#         start_coord.to_crs(nodes.crs)
#         source = gpd.sjoin_nearest(start_start_coord, nodes)['N'].item()
        
#     if ~pseudo_G.has_node(target):
#         end_coord = trips_df.loc[trips_df['od']==(source,target),['end_lat','end_lon']].iloc[0]
#         end_coord['geometry'] = Point(end_coord.iloc[0,1],end_coord.iloc[0,0])
#         end_coord = gpd.GeoDataFrame(end_coord,geometry='geometry',crs='epsg:4326')
#         end_coord.to_crs(nodes.crs)
#         target = gpd.sjoin_nearest(end_coord, nodes)['N'].item()
        
#     return (source,target)

# for segment_filepath in segment_filepaths:
#     trips_df = pd.read_csv(segment_filepath)
#     #turn to tuple
#     trips_df['od'] = trips_df['od'].apply(lambda row: ast.literal_eval(row))
#     trips_df['od'] = trips_df.apply(lambda row: replace_missing_node(row, trips_df, pseudo_G, nodes))
    
#     #inputs
#     sum_all = trips_df['chosen_length'].sum() * 5280
#     links['tup'] = list(zip(links['A'],links['B']))
#     link_lengths = dict(zip(links['tup'],links['length_ft']))
#     durations = []
#     ods = list(set(trips_df['od'].tolist()))
    
#     start = time.time()
#     bounds = [[-5,5],[-5,5],[-5,5]]
#     x = minimize(objective_function, bounds, args=(links,G,ods,trips_df,matched_traces,durations), method='pso')
#     end = time.time()
#     print(f'Took {(end-start)/60/60} hours')
#     results[segment_filepath] = (x.x,x.fun)

# #%%
# timestr = time.strftime("%Y-%m-%d-%H-%M")
# with (fp/f"calibration_results_{timestr}.pkl").open('wb') as fh:
#     pickle.dump(results,fh)


# new_results = {key.parts[-1].split('.csv')[0]:items for key, items in results.items()}
# new_results = pd.DataFrame.from_dict(new_results,orient='index',columns=['coefs','overlap'])
# new_results[['not_beltline','not_infra','2lanes','30mph']] = new_results['coefs'].apply(pd.Series)