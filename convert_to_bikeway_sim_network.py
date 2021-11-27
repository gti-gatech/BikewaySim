# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 15:34:15 2021

@author: tpassmore6
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import os
from pathlib import Path

#make directory/pathing more intuitive later
user_directory = os.fspath(Path.home()) #get home directory and convert to path string
file_directory = "\Documents\GitHub\BikewaySim_Network_Processing" #directory of bikewaysim network processing code

#change this to where you stored this folder
os.chdir(user_directory+file_directory)


#%% Network Import Function
def import_attributes(network_name, study_area, link_type, nodes_attr = False):
    
    linksfp = rf'Processed_Shapefiles/{network_name}/{network_name}_{study_area}_{link_type}_id.geojson' 
    links = gpd.read_file(linksfp, ignore_geometry = True)
    
    #if nodes_attr == True:
        #implement later
        #nodesfp = rf'Processed_Shapefiles/{network_name}/{network_name}_{link_type}_nodes.geojson'
        #nodes = gpd.read_file(nodesfp, ignore_geometry = True)
    
    return links #, nodes


def import_conflated_network():
    
    linksfp = r'Processed_Shapefiles/bikewaysim/bikewaysim_links_v4.geojson'
    nodesfp = r'Processed_Shapefiles/bikewaysim/bikewaysim_nodes_v4.geojson'
    
    links = gpd.read_file(linksfp).set_crs(epsg=2240, allow_override = True) #import network links and project
 
    nodes = gpd.read_file(nodesfp).set_crs(epsg=2240, allow_override = True)
    nodes = nodes.rename(columns={'geometry':'bikwaysim_coords'}).set_geometry('bikwaysim_coords')
    
    return links, nodes
    

#%% Add back in attributes

#look at saving attr information as pickle in the first network import file?

def merge_network_and_attributes(conflated_network_name, study_area, network_name1, network_name2, network_name3):
    
    #conflated_network_name = 'bikewaysim'
    #study_area = 'study_area'
    #network_name1 = 'abm'
    #network_name2 = 'navstreets'
    #network_name3 = 'osm'
    #link_type = 'road'
    
    
    #import conflated network
    conflated_network, conflated_network_nodes = import_conflated_network()
    
    #no bike conflated network
    #conflated_network, conflated_network_nodes = import_conflated_network_nobike()
    
    
    conflated_network = conflated_network.drop_duplicates()
    
    #import first attr
    links_attr1 = import_attributes(network_name1, study_area, 'road')
    #merge based on network 1 name
    merged_network1 = pd.merge(conflated_network, links_attr1, how='left', on=['abm_A','abm_B'])
    
    del links_attr1
    
    #import second attr
    links_attr2 = import_attributes(network_name2, study_area, 'road')
    #merge based on network 2 name
    merged_network2 = pd.merge(merged_network1, links_attr2, how='left', on=['navstreets_A','navstreets_B'])
    
    del links_attr2
    
    #import third attr
    links_attr3 = import_attributes(network_name3, study_area, 'bike')
    #merge based on network 2 name
    merged_network3 = pd.merge(merged_network2, links_attr3, how='left', on=['osm_A','osm_B'])
    
    #export?
    #merged_network3.to_file(rf'Processed_Shapefiles/bikewaysim/bikewaysim_links_final.geojson', driver='GeoJSON')
    
    return merged_network3, conflated_network_nodes
    

#%% Run

bikewaysim_links, bikewaysim_nodes = merge_network_and_attributes('bikewaysim', 'study_area', 'abm', 'navstreets', 'osm')


#%% Attribute Cleaning

# #need these fields for links
# A, B, NAME, SPEED_LIMI, SHAPELNGT, FACTYPE

bikewaysim_links['NAME'] = bikewaysim_links.apply(lambda row: row['NAME'] if row['NAME'] == row['NAME'] else row['ST_NAME'], axis=1)
bikewaysim_links['NAME'] = bikewaysim_links.apply(lambda row: row['NAME'] if row['NAME'] == row['NAME'] else 'placeholder', axis=1)

bikewaysim_links['DISTANCE'] = bikewaysim_links.length
bikewaysim_links['SPEEDLIMIT'] = bikewaysim_links['SPEEDLIMIT'].apply(lambda row: row if row == row else 30)
bikewaysim_links['FACTYPE'] = bikewaysim_links['FACTYPE'].apply(lambda row: row if row == row else 11)

#%% Facil Filters

# =============================================================================
# bike_lanes_coa = [
#     'Bike Lane',
#     'Buffered Bike Lane',
#     'Buffered Bike Lane / Bike Lane', 
#     'Buffered Bike Lanes',
#     'Buffered Contraflow Bike Lane',
#     'Uphill Bike Lane / Downhill Sharrows',
#     'Uphill Bike Lane | Downhill Sharrows',
#     'Uphill bike lane downhill shared lane markings',
#     'Uphill bike lane Downhill Sharrows',
#     'Uphill Bike Lane, Downhill SLMS',
#     'Uphill Bike Lanes / Downhill Sharrows',
#     'Uphill Buffered Bike Lane | Downhill Sharrows',
#     'Uphill Buffered Bike Lanes | Downhill Sharrows'
#     ]
# 
# bike_routes_coa = [
#     'Enhanced shared Roadway',
#     'Enhanced Shared Roadway',
#     'Neghborhood Greenway',
#     'Neighborhood Greenway',
#     'Neighborhood greenway',
#     'Sharrows'
#     ]
# 
# bike_paths_coa = [
#     'Curbless Shared Bike/Ped Street (no cars)',
#     'Hard Surface Multi-Use Path',
#     'Multi-Use Path',
#     'Protected Bike Lane',
#     'Raised Bike Lane',
#     'Seperated Bike Lane',
#     'Seperated Bike Lanes',
#     'Seperated WB Lane/Buffered EB Lane',
#     'Shared Use Path',
#     'Two Way Cycle Track'
# ]
# 
# status = [
#     'Existing',
#     'Funded',
#     'Planned'
#     ]
# =============================================================================


#%% Required for utilities


#need these attributes

#distance on facility related
bikewaysim_links['ORDINARY'] = 0 #TRUE if all other types false
bikewaysim_links['BIKE_LANE'] = 0
bikewaysim_links['BIKE_ROUTE'] = 0
bikewaysim_links['ARTERIAL'] = 0
bikewaysim_links['BIKE_PATH'] = 0
bikewaysim_links['WRONG_WAY'] = 0
bikewaysim_links['CYCLE_TRACK'] = 0




#%%starting here

#re-do this with np.select?

#bike route
#bikewaysim_links.loc[
  #  (bikewaysim_links['facil_arc'].isin(bike_routes_coa)) & (bikewaysim_links['STATUS_coa'] == 'Existing') , 'BIKE_ROUTE'] = 1

#bike lane
bikewaysim_links.loc[
   (bikewaysim_links['facil_arc'] == 'Bike Lane'), 'BIKE_LANE'] = 1

#cycletrack
bikewaysim_links.loc[
   (bikewaysim_links['facil_arc'] == 'Protected Bike Lane'), 'CYCLE_TRACK'] = 1
 
#bike path
osm_filter_method = ['cycleway','footway','path','pedestrian','steps']
bikewaysim_links.loc[
   (bikewaysim_links['highway'].isin(osm_filter_method)), 'BIKE_PATH'] = 1

#set everything else to ordinary = 1
test = ['BIKE_ROUTE', 'BIKE_LANE', 'BIKE_PATH', 'CYCLE_TRACK']
bikewaysim_links.loc[bikewaysim_links[test].sum(axis=1) < 1, 'ORDINARY'] = 1

#more advanced, not sure how to implement yet
bikewaysim_links['PATH_SIZE'] = 0
bikewaysim_links['TURNS'] = 0 #look at previous road name?
bikewaysim_links['ELEVATION_GAIN'] = 0 #ask to get this at some point


bikewaysim_links = bikewaysim_links.rename(columns={'bikewaysim_A':'A', 'bikewaysim_B':'B', 'SPEEDLIMIT':'SPEED_LIMI'})

#%% Utilities

coef = {
    'ORDINARY': 1,
    'BIKE_PATH': 0.289,
    'BIKE_LANE': 0.634,
    'distance_bike_routes': 0.901,
    'distance_arterial_no_bike': 2.224,
    'CYCLE_TRACK': 0.494,
    'distance_bike_boulevards': 0.400,
    'distance_wrong_way': 5.015,
    'elevation_gain': 0.012,
    'num_of_turns': 0.097,
    'traffic_signals': 0.047,
    'unsig_left_prin_arterial': 0.420,
    'unsig_left_min_arterial': 0.175,
    'unsig_left_onto_prin_arterial': 0.559,
    'unsig_left_onto_min_arterial': 0.117,
    'log_path_size': 1.000,
    }


bikewaysim_links['DISTANCE'] = bikewaysim_links['DISTANCE'] * (bikewaysim_links['ORDINARY']*coef['ORDINARY']+bikewaysim_links['BIKE_LANE']*coef['BIKE_LANE']+bikewaysim_links['BIKE_PATH']*coef['BIKE_PATH']+bikewaysim_links['CYCLE_TRACK']*coef['CYCLE_TRACK'])

#%% create reverse links
bikewaysim_links_rev = bikewaysim_links.copy().rename(columns={'A':'B','B':'A'})

#filter to those that are two way
bikewaysim_links_rev = bikewaysim_links_rev[(bikewaysim_links_rev['two_way'] != False) &
                                            (bikewaysim_links_rev['DIR_TRAVEL'] != 'F') &
                                            (bikewaysim_links_rev['DIR_TRAVEL'] != 'T')                            
                                            ]

bikewaysim_links = bikewaysim_links.append(bikewaysim_links_rev).reset_index()

bikewaysim_links['A_B'] = bikewaysim_links['A'] + '_' + bikewaysim_links['B']

#%% Filter down to relevant fields

bikewaysim_links = bikewaysim_links[['A','B','A_B','NAME','SPEED_LIMI','DISTANCE','FACTYPE','ORDINARY','BIKE_LANE','BIKE_ROUTE',
                                     'ARTERIAL','BIKE_PATH','CYCLE_TRACK','WRONG_WAY','PATH_SIZE','TURNS','ELEVATION_GAIN','geometry']]

#%%for nodes
# N, X, Y, lat, lon
bikewaysim_nodes = bikewaysim_nodes.rename(columns={'bikewaysim_ID':'N'})

bikewaysim_nodes['X'] = bikewaysim_nodes.geometry.x
bikewaysim_nodes['Y'] = bikewaysim_nodes.geometry.y

bikewaysim_nodes = bikewaysim_nodes.to_crs(epsg=4326)
bikewaysim_nodes['lon'] = bikewaysim_nodes.geometry.x
bikewaysim_nodes['lat'] = bikewaysim_nodes.geometry.y

bikewaysim_nodes = bikewaysim_nodes[['N','X','Y','lon','lat','bikwaysim_coords']]

#%% Export?
# need to set this environmental path for network data and query data at separate locations
user_directory = os.fspath(Path.home()) #get home directory and convert to path string

# set path variable for SidewalkSim
transportsim_dir = user_directory + "/Documents/GitHub/BikewaySim/TransportSim"

bikewaysim_links.to_file(transportsim_dir + r'/bikewaysim_network/2020 links/2020_links.geojson', driver = 'GeoJSON')
bikewaysim_nodes.to_file(transportsim_dir + r'/bikewaysim_network/2020 nodes with latlon/2020_nodes_latlon.geojson', driver = 'GeoJSON')

bikewaysim_nodes.to_csv(transportsim_dir + r'/bikewaysim_network/2020 nodes with latlon/2020_nodes_latlon.csv')

#%% Base ABM Network

#links
abm_linksfp = r'Processed_Shapefiles/abm/abm_study_area_base_id.geojson'

abm_links = gpd.read_file(abm_linksfp).set_crs(epsg=2240, allow_override = True) #import network links and project    

abm_links = abm_links.rename(columns={"abm_A":"A","abm_B":"B",'SPEEDLIMIT':'SPEED_LIMI'})

#reverse links
abm_links_rev = abm_links.copy().rename(columns={'A':'B','B':'A'})

#filter to those that are two way
abm_links_rev = abm_links_rev[abm_links_rev['two_way'] == True]

abm_links = abm_links.append(abm_links_rev).reset_index()

abm_links['A_B'] = abm_links['A'] + '_' + abm_links['B']

#rename speed limit
#abm_links['SPEED_LIMI'] = abm_links['SPEED_LIMIT']

#re-calculate distance
abm_links['DISTANCE'] = abm_links.length

#create empty columns
abm_links['ORDINARY'] = 0 #TRUE if all other types false
abm_links['BIKE_LANE'] = 0
abm_links['BIKE_ROUTE'] = 0
abm_links['ARTERIAL'] = 0
abm_links['BIKE_PATH'] = 0
abm_links['WRONG_WAY'] = 0
abm_links['PATH_SIZE'] = 0
abm_links['TURNS'] = 0 #look at previous road name?
abm_links['ELEVATION_GAIN'] = 0 #ask to get this at some point

#filter links
abm_links = abm_links[['A','B','A_B','NAME','SPEED_LIMI','DISTANCE','FACTYPE','ORDINARY','BIKE_LANE','BIKE_ROUTE',
                                     'ARTERIAL','BIKE_PATH','WRONG_WAY','PATH_SIZE','TURNS','ELEVATION_GAIN','geometry']]

#nodes
abm_nodesfp = r'Processed_Shapefiles/abm/abm_study_area_base_nodes.geojson'

abm_nodes = gpd.read_file(abm_nodesfp).set_crs(epsg=2240, allow_override = True)    

abm_nodes = abm_nodes.rename(columns={'abm_ID':'N'})

abm_nodes['X'] = abm_nodes.geometry.x
abm_nodes['Y'] = abm_nodes.geometry.y

abm_nodes = abm_nodes.to_crs(epsg=4326)
abm_nodes['lon'] = abm_nodes.geometry.x
abm_nodes['lat'] = abm_nodes.geometry.y

abm_nodes = abm_nodes[['N','X','Y','lon','lat','geometry']]

#export
abm_links.to_file(transportsim_dir + r'/abm/2020 links/2020_links.geojson', driver = 'GeoJSON')
abm_nodes.to_file(transportsim_dir + r'/abm/2020 nodes with latlon/2020_nodes_latlon.geojson', driver = 'GeoJSON')

abm_nodes.to_csv(transportsim_dir + r'/abm/2020 nodes with latlon/2020_nodes_latlon.csv')

#%% column

#print(bikewaysim_links.columns)


#%% Edit to match BikewaySim code format

#%% Add reverse links if neccessary 