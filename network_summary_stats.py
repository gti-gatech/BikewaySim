# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 09:15:35 2021

@author: tpassmore6
"""

import geopandas as gpd
import pandas as pd
import time
import os

def sum_all_networks(networks,studyarea_name):
    
    #summary table
    summary_table = pd.DataFrame(columns=['network name','link_type','num_links','num_nodes','tot_link_length','avg_link_length','avg_'])
    
    for i in networks:
        summary_table = run_sum_function(summary_table, studyarea_name, i)
    
    #export summary table
    summary_table.to_csv(r"network_summary.csv")
   
    print(summary_table)

def run_sum_function(summary_table, studyarea_name, network_name):
	#certain link type shapefile exists, then add node ids to it and create node shapefile
    if os.path.exists(rf'processed_shapefiles/{network_name}/{network_name}_{studyarea_name}_base_links.geojson'):
        links, nodes = import_links_and_nodes(network_name, studyarea_name, "base")
        summary_table = summurize_network(summary_table, links, nodes, network_name, studyarea_name, "base")

    if os.path.exists(rf'processed_shapefiles/{network_name}/{network_name}_{studyarea_name}_road_links.geojson'):
        links, nodes = import_links_and_nodes(network_name, studyarea_name, "road")
        summary_table = summurize_network(summary_table, links, nodes, network_name, studyarea_name, "road")

    if os.path.exists(rf'processed_shapefiles/{network_name}/{network_name}_{studyarea_name}_bike_links.geojson'):
        links, nodes = import_links_and_nodes(network_name, studyarea_name, "bike")
        summary_table = summurize_network(summary_table, links, nodes, network_name, studyarea_name, "bike")

    if os.path.exists(rf'processed_shapefiles/{network_name}/{network_name}_{studyarea_name}_service_links.geojson'):
        links, nodes = import_links_and_nodes(network_name, studyarea_name, "service")
        summary_table = summurize_network(summary_table, links, nodes, network_name, studyarea_name, "service")
    
    return summary_table


#%% Summurize Function

def import_links_and_nodes(network_name, studyarea_name, link_type):
    links = gpd.read_file(rf'processed_shapefiles/{network_name}/{network_name}_{studyarea_name}_{link_type}_links.geojson').set_crs(epsg=2240, allow_override = True)
    nodes = gpd.read_file(rf'processed_shapefiles/{network_name}/{network_name}_{studyarea_name}_{link_type}_nodes.geojson').set_crs(epsg=2240, allow_override = True)
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
    #print(f'There are {num_links} links, {num_nodes} nodes, {sum_miles} miles of links, and average link length of {avg_len} miles in {network_name}')
    
    return summary_table
