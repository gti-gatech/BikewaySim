# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 13:22:22 2021

@author: tpassmore6
"""

#%%import cell
import geopandas as gpd
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' #get rid of copy warning
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  
import os
import time
import pickle
from shapely.geometry import Point, LineString
from collections import Counter

#import custom module
from node_ids import *

def filter_networks(studyarea, studyarea_name, networkfp, network_name, network_mapper, A = None, B = None, layer = 0):
	
    #record the starting time
    tot_time_start = time.time()
    
    #create a new folder for the network if it doesn't already exist
    if not os.path.exists(f'Processed_Shapefiles/{network_name}'):
        os.makedirs(f'Processed_Shapefiles/{network_name}') 

    #import the network
    network = import_network(networkfp, network_name, studyarea, studyarea_name, network_mapper, A, B, layer)  

    #apply filtering methods and create nodes
    filter_to_roads(network, network_name, studyarea_name)
    filter_to_bike(network, network_name, studyarea_name)
    filter_to_service(network, network_name, studyarea_name)
    
    #print the total time it took to run the code
    print(f'{network_name} imported... took {round(((time.time() - tot_time_start)/60), 2)} minutes')

# Import Study Area Function
#imports spatial data for use as a mask and projects it to desired projection
def import_study_area(studyareafp, studyarea_name, desired_crs):
 
    gdf = gpd.read_file(studyareafp)
    
    if gdf.crs != desired_crs:
        gdf = gdf.to_crs(desired_crs)
    
    # calculate the area of the study area in square miles
    sqmi = round(float(gdf['geometry'].area / 5280**2),2) 
    print(f"Done... The area of the {studyarea_name} boundary is {sqmi} square miles.")
    gdf.plot()
    
    return gdf

# Import Network Data for Selected Study Area Function

def import_network(networkfp, network_name, studyarea, studyarea_name, network_mapper, A, B, layer):
    
    #explode and drop level to get rid of multi-index in abm layer
    links = gpd.read_file(networkfp, mask = studyarea, layer = layer).explode().droplevel(level=1).to_crs(epsg=2240) 
    
    #add in general cleaning measures here based on network_name
    if network_name == 'osm':
        print(f'Cleaning measures applied for {network_name}...')
        
    elif network_name == 'abm':
        print(f'Cleaning measures applied for {network_name}...')
        
        #write unmodified version to file for examining
        #links.to_file(f'Processed_Shapefiles/{network_name}/{network_name}_{studyarea_name}_w_dup_links.geojson', driver = 'GeoJSON')
        
        # delete duplicate links in the abm network
        # Since ABM has reversible links, we delete B_A rows if A_B rows already exists 
        # this would help with further identification of whether a point lands on a line
        df_dup = pd.DataFrame(np.sort(links[["A","B"]], axis=1), columns=["A","B"])
        df_dup['two_way'] = df_dup.duplicated(keep=False)
        df_dup = df_dup.drop_duplicates()

        links = pd.merge(links,df_dup,how='inner', left_index = True, right_index = True, suffixes=(None,'_drop')).drop(columns={'A_drop','B_drop'})

    else:
        print(f'No cleaning measures applied for {network_name}...')
   

    #create nodes and add node ids to links
    nodes, links = add_nodes(links, studyarea_name, network_name, 'base', network_mapper, A , B)  
    
    #export links and nodes
    links.to_file(rf'Processed_Shapefiles/{network_name}/{network_name}_{studyarea_name}_base_links.geojson', driver = 'GeoJSON')
    nodes.to_file(rf"Processed_Shapefiles/{network_name}/{network_name}_{studyarea_name}_base_nodes.geojson", driver = 'GeoJSON')
           
    return links

# Filtering Network to Road Links

def filter_to_roads(links, network_name, studyarea_name):  
    filter_specified = True

    #filtering logic
    if network_name == "osm": # osm network
        print(f'{network_name} road filter applied...')    
        
        #find service links that still have a name
        service_links_with_name = links[ (links['highway'] == 'service') & (links['name'].isnull() == False) ]
        
        osm_filter_method = ['primary','primary_link','residential','secondary','secondary_link',
                            'tertiary','tertiary_link','trunk','trunk_link'] 
        
        road_links = links[links["highway"].isin(osm_filter_method)]
        
        #add back in service links with a name
        road_links = road_links.append(service_links_with_name)
        
    elif network_name == "abm": # abm network
        print(f'{network_name} road filter applied...')
        abm_road = [10,11,14]
        road_links = links[links["FACTYPE"].isin(abm_road)]
        
    elif network_name == "here": # here network ## in future if there are more layers just modify this if condition
        print(f'{network_name} road filter applied...')    
        road_links = links[(links['CONTRACC'].str.contains('N'))&
                          (links['SPEED_CAT'].str.contains('8') == False)&
                          (links['AR_AUTO'].str.contains('Y'))&
                          (links['RAMP'].str.contains('N'))]  
    else:
        print(f"No road filtering method available for {network_name}...")
        filter_specified = False

    if filter_specified == True:
        #export links
        road_links.to_file(rf'Processed_Shapefiles/{network_name}/{network_name}_{studyarea_name}_road_links.geojson', driver = 'GeoJSON')
        
        #create nodes
        nodes = make_nodes(road_links, network_name, 'road', studyarea_name)
        #export nodes
        nodes.to_file(rf"Processed_Shapefiles/{network_name}/{network_name}_{studyarea_name}_road_nodes.geojson", driver = 'GeoJSON')

#Filtering Network to Bike Links

def filter_to_bike(links, network_name, studyarea_name):     
    filter_specified = True

    #filtering logic
    if network_name == "osm": # osm network
        print(f'{network_name} bike filter applied...') 
        osm_filter_method = ['cycleway','footway','path','pedestrian','steps']
        bike_links = links[links["highway"].isin(osm_filter_method)]
        bike_links = bike_links[bike_links['footway'] != 'sidewalk']
        bike_links = bike_links[bike_links['footway'] != 'crossing']
        
    elif network_name == "abm": # abm network
        print(f'No bike links present for {network_name}')
        filter_specified = False
        
    elif network_name == "here": # here network ## in future if there are more layers just modify this if condition
        print(f'{network_name} bike filter applied...') 
        bike_links = links[ links['AR_AUTO'].str.contains('N') ]
    else:
        print(f"No bike filtering method available for {network_name}.")
        filter_specified = False

    if filter_specified == True:
        #export links
        bike_links.to_file(rf'Processed_Shapefiles/{network_name}/{network_name}_{studyarea_name}_bike_links.geojson', driver = 'GeoJSON')
        #create nodes
        nodes = make_nodes(bike_links, network_name, 'bike', studyarea_name)
        #export nodes
        nodes.to_file(rf"Processed_Shapefiles/{network_name}/{network_name}_{studyarea_name}_bike_nodes.geojson", driver = 'GeoJSON')
    
    return

#filtering to service links
def filter_to_service(links, network_name, studyarea_name):
    #if this variable remains true, then there are service links
    filter_specified = True
      
    #filtering logic, need to specify for new networks
    if network_name == "osm": # osm network
        print(f'{network_name} service filter applied...') 
        osm_filter_method = ['service']
        service_links = links[links["highway"].isin(osm_filter_method)]  
    
    elif network_name == "abm": # abm network
        print(f'No service links present for {network_name}')
        filter_specified = False
   
    elif network_name == "here": #here network
        print(f'{network_name} service filter applied...') 
        service_links = links[ (links['AR_AUTO'].str.contains('Y')) & (links['SPEED_CAT'].str.contains('8')) ]
    
    else:
        print(f"No sevice filtering method available for {network_name}.")
        filter_specified = False
    
    
    if filter_specified == True:
        #export links
        service_links.to_file(rf'Processed_Shapefiles/{network_name}/{network_name}_{studyarea_name}_service_links.geojson', driver = 'GeoJSON')
        #create nodes
        nodes = make_nodes(service_links, network_name, 'serivce', studyarea_name)
        #export nodes
        nodes.to_file(rf"Processed_Shapefiles/{network_name}/{network_name}_{studyarea_name}_service_nodes.geojson", driver = 'GeoJSON')

    return

   