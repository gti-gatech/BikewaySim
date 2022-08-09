# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 09:15:35 2021

@author: tpassmore6
"""

import geopandas as gpd
import pandas as pd
import os
import fiona


def sum_all_networks(networks,studyarea_name):
    
    #summary table
    #can add other metrics of interest in the future
    summary_table = pd.DataFrame(columns=['network','link_type','num_links','num_nodes','tot_link_length','avg_link_length'])
    
    #expected link types
    link_types = ['base','road','bike','service']

    #go through each network
    for network in networks:
        #expected filepath
        fp = rf'processed_shapefiles/{network}/{network}_{studyarea_name}_network.gpkg'
        #check if gpkg file exists
        if os.path.exists(fp):
            #go through each link type
            for link_type in link_types:
                #check if network has link type
                if link_type + '_links' in fiona.listlayers(fp):
                    #get summary stats and append them to dataframe
                    summary_table = summurize_network(summary_table, fp, link_type, network)
    
    #export summary table
    summary_table.to_csv(r"network_summary.csv",index=False)
   
    print(summary_table)


#%% Summurize Function


def summurize_network(summary_table, fp, link_type, network):
    
    #import network
    links = gpd.read_file(fp,layer=f'{link_type}_links').set_crs(epsg=2240, allow_override = True)
    nodes = gpd.read_file(fp,layer=f'{link_type}_nodes').set_crs(epsg=2240, allow_override = True)
    
    #how many links
    num_links = len(links)
    
    #how many nodes
    num_nodes = len(nodes)
    
    #total mileage of links
    length_mi = links.geometry.length / 5280 # create a new distance column and calculate mileage of each link
    tot_link_length = round(length_mi.sum(),0)
    
    #average link length
    avg_link_length = round(links.geometry.length.mean(),1)
    
    #create data frame
    summary_table.loc[len(summary_table)] = [network,link_type,num_links,num_nodes,tot_link_length,avg_link_length]
    
    #average number of connecting nodes
    #avg_connect_nodes = round(nodes[f'{network_name}_num_links'].mean(),2)
    
    #add to summary table
    #summary_table.loc[len(summary_table.index)] = [network_name, link_type, num_links, num_nodes, sum_miles, avg_len]
    
    #Print Stats
    #print(f'There are {num_links} links, {num_nodes} nodes, {sum_miles} miles of links, and average link length of {avg_len} miles in {network_name}')
    
    return summary_table

#%% run


#%%
# import os
# from pathlib import Path
# import time
# import geopandas as gpd
# import pickle

# user_directory = os.fspath(Path.home()) #get home directory and convert to path string
# file_directory = r"/Documents/BikewaySimData" #directory of bikewaysim outputs
# os.chdir(user_directory+file_directory)

# #network names to look for, will search your directory for network name
# networks = ["abm","here","osm"]
# studyarea_name = "bikewaysim"

# #summurize networks and export summary as "network_summary.csv in the working directory
# summary_table = sum_all_networks(networks, studyarea_name)


# import os

# #make directory/pathing more intuitive later
# file_dir = r"C:\Users\tpassmore6\Documents\BikewaySimData" #directory of bikewaysim network processing code

# #change this to where you stored this folder
# os.chdir(file_dir)

# networks = ['abm','here','osm']
# studyarea_name = 'bikewaysim'

# sum_all_networks(networks, studyarea_name)
