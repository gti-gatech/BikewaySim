# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 16:54:13 2021

@author: tpassmore6
"""
import network_filter
import node_ids
import network_summary_stats
import os
from pathlib import Path
import time
import geopandas as gpd

tot_time_start = time.time()

#make directory/pathing more intuitive later
user_directory = os.fspath(Path.home()) #get home directory and convert to path string
file_directory = "/Documents/GitHub/BikewaySim_Network_Processing" #directory of bikewaysim network processing code

#change this to where you stored this folder
os.chdir(user_directory+file_directory)

#%% Study Area File Paths

bikewaysim_study_areafp = r'Base_Shapefiles/bikewaysim_study_area/bikewaysim_study_area.shp'
city_of_atlantafp = r'Base_Shapefiles/coa/Atlanta_City_Limits.shp'
atlanta_regional_commissionfp = r'Base_Shapefiles/arc/arc_bounds.shp'

#add new study areas if desired

#%% Network Data Filepaths

#links
abmfp = r'Base_Shapefiles/arc/ABM2020-TIP20-2020-150kShapefiles-outputs.gdb'
herefp = r'Base_Shapefiles/here/Streets.shp'
osmfp = r'Base_Shapefiles/osm/osm_cleaning/final_osm_links.geojson'
#rc_routesfp = r'Base_Shapefiles/gdot/rc_routes.geojson'

#nodes
osm_nodesfp = r'Base_Shapefiles/osm/osm_nodes_arc.geojson'

#%% Run Network Import Settings


#defined networks
networks = ["abm","here","osm"]

#desired study area
studyareafp = bikewaysim_study_areafp
studyarea_name = "study_area"

#abm inputs
abm = {
       "studyareafp": studyareafp,
       "studyarea_name": studyarea_name,
       "networkfp": abmfp,
       "network_name": 'abm',
       "link_node": False,
       "A": "A",
       "B": "B",
       "layer": "DAILY_LINK",
       }

#here inputs
here = {
       "studyareafp": studyareafp,
       "studyarea_name": studyarea_name,
       "networkfp": herefp,
       "network_name": 'here',
       "link_node": False,
       "A": "REF_IN_ID",
       "B": "NREF_IN_ID",
       "layer": 0,
       }

osm = {
      "studyareafp": studyareafp,
       "studyarea_name": studyarea_name,
       "networkfp": osmfp,
       "network_name": 'osm',
       "link_node": True,
       "A": None,
       "B": None,
       "layer": 0,
       }

#%% run network filter

network_filter.filter_networks(**abm)
network_filter.filter_networks(**here)
network_filter.filter_networks(**osm)



#%% run node_id creation

node_ids.add_nodes_link_types(abm['studyarea_name'],abm['network_name'],abm['A'],abm['B'])
node_ids.add_nodes_link_types(here['studyarea_name'],here['network_name'],here['A'],here['B'])
node_ids.add_nodes_link_types(osm['studyarea_name'],osm['network_name'],osm['A'],osm['B'], osm_nodesfp)


#%% run network summaries
#sum_table = network_summary_stats.summurize_all_networks(studyarea_name, networks)

#%% run conflation
from network_conflation import *

#%% Conflating HERE to OSM
# Import Links and Nodes

import matplotlib.pyplot as plt

#filepaths
abm_road, abm_road_nodes = import_links_and_nodes('abm','study_area','road')
osm_road, osm_road_nodes = import_links_and_nodes('osm','study_area','road')

here_road, here_road_nodes = import_links_and_nodes('here', 'study_area','road')


#%% filter out nodes with two connecting links

#remove nodes with only 2 connecting links
osm_filt = osm_road_nodes[osm_road_nodes[f'osm_num_links'] != 2 ]
here_filt = here_road_nodes[here_road_nodes[f'here_num_links'] != 2 ]

#intersection matching
#tolerances
tolerances = list(range(5,36,10))

#use for iterating through tolerances
matched_intersections_here = {}
matches_here = []
dup_here = []
rem_here = []
rem_2_link_here = []

matched_intersections_osm = {}
matches_osm = []
dup_osm = []
rem_osm = []
rem_2_link_osm = []

for x in tolerances:
    matched_intersections_here[x], dup_matched, matched = match_intersections(abm_road_nodes, 'abm', here_filt, 'here', x)
    matches_here.append(matched)
    dup_here.append(dup_matched)
    unmatched_abm_nodes, unmatched_here_nodes = remaining_intersections(matched_intersections_here[x], abm_road_nodes, 'abm', here_road_nodes, 'here')
    rem_here.append(len(unmatched_abm_nodes))
    rem_2_link_here.append(len(unmatched_abm_nodes[unmatched_abm_nodes['abm_num_links'] != 2]))
    
    matched_intersections_osm[x], dup_matched, matched = match_intersections(abm_road_nodes, 'abm', osm_filt, 'osm', x)
    matches_osm.append(matched)
    dup_osm.append(dup_matched)
    unmatched_abm_nodes, unmatched_osm_nodes = remaining_intersections(matched_intersections_osm[x], abm_road_nodes, 'abm', osm_road_nodes, 'osm')
    rem_osm.append(len(unmatched_abm_nodes))
    rem_2_link_osm.append(len(unmatched_abm_nodes[unmatched_abm_nodes['abm_num_links'] != 2]))

df = pd.DataFrame(list(zip(tolerances,matches_here,dup_here,rem_here,rem_2_link_here,matches_osm,dup_osm,rem_osm,rem_2_link_osm)), 
                  columns = ['tolerance', 'here match', 'here dup','here_rem','here_rem_2', 'osm match', 'osm dup', 'osm rem', 'osm rem 2'])

#plot relationship between tolerance and matches and dup_matches
plt.plot(tolerances, matches_osm) 
plt.plot(tolerances, matches_here)

#find remaining nodes and match these
unmatched_abm_nodes, unmatched_osm_nodes  = remaining_intersections(matched_intersections_here[35], abm_road_nodes, 'abm', here_road_nodes, 'here')

#%%

matched_intersections, dup_matched, matched = match_intersections(here_filt, 'here', osm_filt, 'osm', 30)
matched_intersections.to_file(r'Processed_Shapefiles/conflation_redux/matched_intersection.geojson', driver = 'GeoJSON')

#find remaining nodes and match these
unmatched_here_nodes, unmatched_osm_nodes  = remaining_intersections(matched_intersections, here_road_nodes, 'here', osm_road_nodes, 'osm')

matched_intersections_2, dup_matched, matched = match_intersections(unmatched_here_nodes, 'here', unmatched_osm_nodes, 'osm', 20)
matched_intersections_2.to_file('Processed_Shapefiles/conflation_redux/other_matched_nodes.geojson', driver = 'GeoJSON')

# add matched nodes to intersection matches
bikewaysim_nodes_v1 = matched_intersections.append(matched_intersections_2).reset_index()

# splitting link
unmatched_here_nodes, unmatched_osm_nodes  = remaining_intersections(bikewaysim_nodes_v1, here_road_nodes, 'here', osm_road_nodes, 'osm')

# find unmatched here nodes that lie on abm, find correspoding interpolated abm node, find corresponding abm link        
unmatched_osm_on, corre_here_point, corre_here_link, osm_nodes_remaining = point_on_line(unmatched_osm_nodes, "osm", here_road, "here", tolerance_ft = 20)


print(f'{len(corre_here_point)} split points found.')

here_split_multi_lines = split_by_node_to_multilinestring(corre_here_link, "here_ip_line", corre_here_point, "here_ip_point", "here")
here_split_lines, here_rest_of_lines = transform_multilinestring_to_segment(here_split_multi_lines, here_road, "here")
print(f'{len(here_split_lines)} here split lines were added.')
 
#it seems like the remaining OSM nodes either are just links that are represented in the here service network,
# dead end streets, feature difference, or a result of how the boundary was clipped





# add these split nodes and links into the conflated network
#create new ids for split nodes
new_nodes = new_ids(corre_here_point, bikewaysim_nodes_v1, 'here', 'osm')
#add split links into conflated network
bikewaysim_links_v1 = add_split_links_into_network(here_road, here_split_lines, new_nodes, 'here', 'osm')

#%%
#transfer attributes

#get ones with ref node pairs
test = (bikewaysim_links_v1['osm_A'].isna() == False) & (bikewaysim_links_v1['osm_B'].isna() == False)

#[ for x in joining_node_ids]














#%%

# https://www.geeksforgeeks.org/python-program-to-get-all-unique-combinations-of-two-lists/
# python program to demonstrate
# unique combination of two lists
# using zip() and permutation of itertools
 
# import itertools package
import itertools
from itertools import permutations
 
# initialize lists
list_of_nodes = list(new_nodes['osm_ID'])

 
# create empty list to store the
# combinations
unique_combinations = []
 
# Getting all permutations of list_1
# with length of list_2
permut = itertools.permutations(list_of_nodes, len(list_of_nodes))
 
# zip() is called to pair each permutation
# and shorter list element into combination
for comb in permut:
    zipped = zip(comb, list_of_nodes)
    unique_combinations.append(list(zipped))
 
# printing unique_combination list
print(unique_combinations.head())


  

#%%

# bring in A_B id for joining network for overlapping links so that overlapping links can be brought in
# bring in any links that don't overlap with base networ
bikewaysim_links_v2 = add_in_other_links(bikewaysim_links_v1, osm_road, new_nodes, 'here', 'osm')

#future method
#transfer attribute
#test = attribute_transfer(osm_road, 'osm', new_nodes, bikewaysim_links_v1)


# add in nodes
bikewaysim_nodes_v3 = node_create_bikewaysim(bikewaysim_links_v2, osm_road_nodes, bikewaysim_nodes_v2, 'here', 'osm')

#add in bike nodes
osm_bike, osm_bike_nodes = import_links_and_nodes('osm','study_area','bike')

bikewaysim_nodes_v4 = add_bike_nodes(bikewaysim_nodes_v3, 'bikewaysim', osm_bike_nodes, 'osm', 35)

#add in bike links (this one isn't working yet)
bikewaysim_links_v3 = add_bike_links(bikewaysim_nodes_v4, bikewaysim_links_v2, 'bikewaysim', osm_bike, 'osm')



#%% run convert for bikewaysim

#%% processing time







#%% everything below is deprecated


#use for iterating through tolerances
matched_intersections_2 = {}
matches = []
dup_matches = []

for x in tolerances:
    matched_intersections_2[x], dup_matched, matched = match_intersections(unmatched_here_nodes, 'here', unmatched_osm_nodes, 'osm', x)
    matches.append(matched)
    dup_matches.append(dup_matched)

#plot relationship between tolerance and matches and dup_matches
plt.plot(tolerances, matches) 
plt.plot(tolerances, dup_matches)
#export and browse in QGIS to check
matched_intersections_2[20].to_file('Processed_Shapefiles/conflation_redux/other_matched_nodes.geojson', driver = 'GeoJSON')

add matched nodes to intersection matches
bikewaysim_nodes_v1 = matched_intersections[30].append(matched_intersections_2[20]).reset_index()

splitting link
unmatched_here_nodes, unmatched_osm_nodes  = remaining_intersections(bikewaysim_nodes_v1, here_road_nodes, 'here', osm_road_nodes, 'osm')

    
matched_intersections = {}
matches = []
dup_matches = [] 
match OSM and HERE intersections
for x in tolerances:
    matched_intersections[x], dup_matched, matched = match_intersections(here_filt, 'here', osm_filt, 'osm', x)
    matches.append(matched)
    dup_matches.append(dup_matched)
#plot relationship between tolerance and matches and dup_matches
plt.plot(tolerances, matches) 
plt.plot(tolerances, dup_matches)
#looks like number of matches platous around 25ft
matched_intersections[30].to_file('Processed_Shapefiles/conflation_redux/matched_intersection.geojson', driver = 'GeoJSON')
unmatched_here_nodes, unmatched_osm_nodes  = remaining_intersections(matched_intersections[30], here_road_nodes, 'here', osm_road_nodes, 'osm')


#export and browse in QGIS to check


#%% run network import
#net_imp.run_network_import(**abm)
#net_imp.run_network_import(**here)
#net_imp.run_network_import(**osm)

#%% COA

#net_imp.run_network_import(city_of_atlantafp, 'city_of_atlanta', abmfp, 'abm', False, True, A = 'A', B = 'B', layer = 'DAILY_LINK'  )
#net_imp.run_network_import(city_of_atlantafp, 'city_of_atlanta', herefp, 'here', False, True, 'REF_IN_ID', 'NREF_IN_ID' )
#net_imp.run_network_import(city_of_atlantafp, 'city_of_atlanta', osmfp, 'osm', True, True)

#ARC
#net_imp.run_network_import(atlanta_regional_commissionfp, 'atlanta_regional_commission', abmfp, 'abm', False, True, A = 'A', B = 'B', layer = 'DAILY_LINK'  )
#net_imp.run_network_import(atlanta_regional_commissionfp, 'atlanta_regional_commission', herefp, 'here', False, True, 'REF_IN_ID', 'NREF_IN_ID' )
#net_imp.run_network_import(atlanta_regional_commissionfp, 'atlanta_regional_commission', osmfp, 'osm', True, True)

#%% Create Attribute Tables

#net_imp.create_attribute_definitions(abmfp, 'abm')
#net_imp.create_attribute_definitions(herefp, 'here')
#net_imp.create_attribute_definitions(osmfp, 'osm')

#%% Import OSM ARC
#net_imp.run_network_import(atlanta_regional_commissionfp, 'arc_bounds', osmfp, 'osm', True, True)

#%% fizzy test
#nodes = net_imp.run_network_import(atlanta_regional_commissionfp, 'arc_bound', osmfp, 'osm', False)

#nodes.to_file(r'Processed_Shapefiles/osm/arc_nodes.geojson', driver = 'GeoJSON')

