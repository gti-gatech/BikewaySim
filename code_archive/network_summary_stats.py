# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 09:15:35 2021

@author: tpassmore6
"""

import network_filter
import node_ids
import geopandas as gpd
import pandas as pd
import time
import os

def summurize_all_networks(studyarea_name,networks):
    
    #record the starting time
    tot_time_start = time.time()
    
    #summary table
    summary_table = pd.DataFrame(columns=['network name','link_type','num_links','num_nodes','tot_link_length','avg_link_length','avg_'])
    
    for i in networks:
        summary_table = run_summurize_function(summary_table, studyarea_name, i)
    
# =============================================================================
#     #other networks
#     #osmnx
#     if os.path.exists(r'Base_Shapefiles/osm/osm_links_osmnx.geojson'): #put in conditional for road network
#         links = gpd.read_feather(r'Base_Shapefiles/osm/osm_links_osmnx.geojson')
#         nodes = gpd.read_feather(rf'Base_Shapefiles/osm/osm_nodes_osmnx.geojson')
#         summary_table = summurize_network(summary_table, links, nodes, 'osmnx', 'studyarea', 'base')  
#     
#     #osm no splitting
#     if os.path.exists(rf'Base_Shapefiles/osm/osm_cleaning/osm_links_base.geojson'):
#         links = gpd.read_file(rf'Base_Shapefiles/osm/osm_cleaning/osm_links_base.geojson')
#         nodes = gpd.read_file(rf'')
#         
# # =============================================================================
# #     #abm double links
# #     if os.path.exists(rf'Base_Shapefiles/arc/ABM2020-TIP20-2020-150kShapefiles-outputs.gdb'):
# #         links = gpd.read_file()
# # =============================================================================
# =============================================================================
        
    
    #export summary table
    summary_table.to_csv(r"Processed_Shapefiles/network_summary.csv")
    
    #print the total time it took to run the code
    print(f'Networks summurized... took {round(((time.time() - tot_time_start)/60), 2)} minutes')

def run_summurize_function(summary_table, studyarea_name, network_name):



	#certain link type shapefile exists, then add node ids to it and create node shapefile
    if os.path.exists(rf'Processed_Shapefiles/{network_name}/{network_name}_{studyarea_name}_base.geojson'):
        links, nodes = import_links_and_nodes(network_name, studyarea_name, "base")
        summary_table = summurize_network(summary_table, links, nodes, network_name, studyarea_name, "base")

    if os.path.exists(rf'Processed_Shapefiles/{network_name}/{network_name}_{studyarea_name}_road.geojson'):
        links, nodes = import_links_and_nodes(network_name, studyarea_name, "road")
        summary_table = summurize_network(summary_table, links, nodes, network_name, studyarea_name, "road")

    if os.path.exists(rf'Processed_Shapefiles/{network_name}/{network_name}_{studyarea_name}_bike.geojson'):
        links, nodes = import_links_and_nodes(network_name, studyarea_name, "bike")
        summary_table = summurize_network(summary_table, links, nodes, network_name, studyarea_name, "bike")

    if os.path.exists(rf'Processed_Shapefiles/{network_name}/{network_name}_{studyarea_name}_service.geojson'):
        links, nodes = import_links_and_nodes(network_name, studyarea_name, "service")
        summary_table = summurize_network(summary_table, links, nodes, network_name, studyarea_name, "service")
    
    
    return summary_table


#%% Summurize Function

def import_links_and_nodes(network_name, studyarea_name, link_type):
    links = gpd.read_file(rf'Processed_Shapefiles/{network_name}/{network_name}_{studyarea_name}_{link_type}.geojson').set_crs(epsg=2240, allow_override = True)
    nodes = gpd.read_file(rf'Processed_Shapefiles/{network_name}/{network_name}_{studyarea_name}_{link_type}_nodes.geojson').set_crs(epsg=2240, allow_override = True)
    return links, nodes

def summurize_network(summary_table, links, nodes, network_name, studyarea_name, link_type):
    
    #how many links
    num_links = len(links)
    
    #how many nodes
    num_nodes = len(nodes)
    
    #total mileage of links
    length_mi = links.geometry.length / 5280 # create a new distance column and calculate mileage of each link
    sum_miles = round(length_mi.sum(),0)
    
    #average link length
    avg_len = round(links.geometry.length.mean(),1)
    
    #average number of connecting nodes
    avg_connect_nodes = round(nodes[f'{network_name}_num_links'].mean(),2)
    
    #add to summary table
    summary_table.loc[len(summary_table.index)] = [network_name, link_type, num_links, num_nodes, sum_miles, avg_len, avg_connect_nodes]
    
    #Print Stats
    print(f'There are {num_links} links, {num_nodes} nodes, {sum_miles} miles of links, and average link length of {avg_len} miles in {network_name}')
    
    return summary_table
