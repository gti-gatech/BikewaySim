# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 23:40:05 2021

@author: tpassmore6
"""

#%% Import Links and Nodes

#filepaths
abm_road, abm_road_nodes = import_links_and_nodes('abm','study_area','road')
#osm_road, osm_road_nodes = import_links_and_nodes('osm','study_area','road')
osm_bike, osm_bike_nodes = import_links_and_nodes('osm','study_area','bike')
here_road, here_road_nodes = import_links_and_nodes('here', 'study_area','road')
#here_bike, here_bike_nodes = import_links_and_nodes('here','bike')


#%% matching abm to here

#these are the matched ABM/here nodes
matched_abm_here, bikewaysim_nodes_v1 = match_intersections(abm_road_nodes, 'abm', here_road_nodes, 'here', 25)

#these are the unmatched nodes
unmatched_abm_nodes, unmatched_here_nodes  = remaining_intersections(matched_abm_here, abm_road_nodes, 'abm', here_road_nodes, 'here')

#%% splitting abm links by here nodes
    
# find unmatched here nodes that lie on abm, find correspoding interpolated abm node, find corresponding abm link        
unmatched_nav_on, corre_abm_point, corre_abm_link = point_on_line(unmatched_here_nodes, "here", abm_road, "abm", tolerance_ft = 26)

print(f'{len(corre_abm_point)} split points found.')

abm_split_multi_lines = split_by_node_to_multilinestring(corre_abm_link, "abm_ip_line", corre_abm_point, "abm_ip_point", "abm")
abm_split_lines, abm_rest_of_lines = transform_multilinestring_to_segment(abm_split_multi_lines, abm_road, "abm")
print(f'{len(abm_split_lines)} abm split lines were added.')
 
#%% Rest of steps

bikewaysim_nodes_v2 = new_ids(corre_abm_point, bikewaysim_nodes_v1,'abm','here')
 
bikewaysim_links_v1 = add_split_links_into_network(abm_road, abm_split_lines, bikewaysim_nodes_v2, 'abm', 'here')

#%% Deal wiht overlapping links and add rest of links in

bikewaysim_links_v2 = add_in_other_links(bikewaysim_links_v1, here_road, bikewaysim_nodes_v2, 'abm', 'here')

#%% create nodes

bikewaysim_nodes_v3 = node_create_bikewaysim(bikewaysim_links_v2, here_road_nodes, bikewaysim_nodes_v2, 'abm', 'here')


#%%add in bike nodes
bikewaysim_nodes_v3 = bikewaysim_nodes_v3.rename(columns={'new_geometry':'bikewaysim_coords'}).set_geometry('bikewaysim_coords')

bikewaysim_nodes_v4 = add_bike_nodes(bikewaysim_nodes_v3, 'bikewaysim', osm_bike_nodes, 'osm', 35)

#%%add in bike links

bikewaysim_links_v3 = add_bike_links(bikewaysim_nodes_v4, bikewaysim_links_v2, 'bikewaysim', osm_bike, 'osm')
    



#%%OSM
#filepaths
abm_road, abm_road_nodes = import_links_and_nodes('abm','study_area','road')
osm_road, osm_road_nodes = import_links_and_nodes('osm','study_area','road')
osm_bike, osm_bike_nodes = import_links_and_nodes('osm','study_area','bike')


#%% matching abm to osm

#these are the matched ABM/here nodes
matched_abm_osm, bikewaysim_nodes_v1 = match_intersections(abm_road_nodes, 'abm', osm_road_nodes, 'osm', 45)

#these are the unmatched nodes
unmatched_abm_nodes, unmatched_osm_nodes  = remaining_intersections(matched_abm_osm, abm_road_nodes, 'abm', osm_road_nodes, 'osm')

#%% splitting abm links by here nodes
    
# find unmatched here nodes that lie on abm, find correspoding interpolated abm node, find corresponding abm link        
unmatched_nav_on, corre_abm_point, corre_abm_link = point_on_line(unmatched_osm_nodes, "osm", abm_road, "abm", tolerance_ft = 26)

abm_split_multi_lines = split_by_node_to_multilinestring(corre_abm_link, "abm_ip_line", corre_abm_point, "abm_ip_point", "abm")
abm_split_lines, abm_rest_of_lines = transform_multilinestring_to_segment(abm_split_multi_lines, abm_road, "abm")

 
#%% Rest of steps

bikewaysim_nodes_v2 = new_ids(corre_abm_point, bikewaysim_nodes_v1,'abm','osm')
 
bikewaysim_links_v1 = add_split_links_into_network(abm_road, abm_split_lines, bikewaysim_nodes_v2, 'abm', 'osm')

#%% Deal wiht overlapping links and add rest of links in

bikewaysim_links_v2 = add_in_other_links(bikewaysim_links_v1, osm_road, bikewaysim_nodes_v2, 'abm', 'osm')

#%% create nodes

bikewaysim_nodes_v3 = node_create_bikewaysim(bikewaysim_links_v2, osm_road_nodes, bikewaysim_nodes_v2, 'abm', 'osm')


#%%add in bike nodes
bikewaysim_nodes_v3 = bikewaysim_nodes_v3.rename(columns={'new_geometry':'bikewaysim_coords'}).set_geometry('bikewaysim_coords')

bikewaysim_nodes_v4 = add_bike_nodes(bikewaysim_nodes_v3, 'bikewaysim', osm_bike_nodes, 'osm', 35)

#%%add in bike links

bikewaysim_links_v3 = add_bike_links(bikewaysim_nodes_v4, bikewaysim_links_v2, 'bikewaysim', osm_bike, 'osm')