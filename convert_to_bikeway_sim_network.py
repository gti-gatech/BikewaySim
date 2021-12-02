# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 15:34:15 2021

@author: tpassmore6
"""

import geopandas as gpd
import pandas as pd
import numpy as np


#%% Add back in attributes

# add attributes to network
def merge_network_and_attributes(conflated_network_links, attr_networks):

    #loop through to import base networks to add attributes back to conflated network
    for attr_network in attr_networks:

        #get the network name
        base_cols = list(attr_networks[attr_network].columns)
        base_id_cols = [base_cols for base_cols in base_cols if "_A_B" in base_cols]
        network_name = [base_id_cols.split('_')[0] for base_id_cols in base_id_cols]

        #add suffix in case there are duplicate column names
        attr_networks[attr_network] = attr_networks[attr_network.add_suffix()]

        #see if conflated network has that network
        if conflated_network_links.columns.isin(network_name) == True:
            #if it does then do a merge
            conflated_network_links = pd.merge(conflated_network_links,attr_networks[attr_network],on=network_name,how='left')
        else:
            print(f'{network_name} not in conflated network')
    
    return conflated_network_links
  

#now that conflated network has all the attributes, will need to deal with conflicting information in future version
#these need to be manually specified, this code might be better run in Spyder than in Jupyter Notebook

#streetnames
#if abm name is present use that, else use the HERE name
bikewaysim_links['streetname'] = bikewaysim_links.apply(lambda row: row['NAME'] if row['NAME'].isnull() == True else row['ST_NAME'], axis=1)
#if no streetname exists then put in "placeholder" as the streetname
bikewaysim_links['streetname'] = bikewaysim_links.apply(lambda row: row['streetname'] if row['streetname'].isnull() == True else 'placeholder', axis=1)

#speed limits
#use the ABM speed limit, if none present assume 30mph
bikewaysim_links['SPEEDLIMIT'] = bikewaysim_links['SPEEDLIMIT'].apply(lambda row: row if row == row else 30)

#facility type

#number of lanes


#check on bikewaysim code

#calculate the initial route cost
bikewaysim_links['distance'] = bikewaysim_links.length

#for streets without a name, put in a placeholder name




bikewaysim_links['FACTYPE'] = bikewaysim_links['FACTYPE'].apply(lambda row: row if row == row else 11)



#%% deprecated below








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