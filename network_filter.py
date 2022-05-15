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

#master function used to run all the filter functions
def filter_networks(studyarea, studyarea_name, networkfp,
                    network_name, network_mapper, layer,
                    desired_crs, nodesfp, nodes_layer = None,
                    node_id = None, A = None, B = None):

    #record the starting time
    tot_time_start = time.time()
    
    #create a new folder for the network if it doesn't already exist
    if not os.path.exists(f'processed_shapefiles/{network_name}'):
        os.makedirs(f'processed_shapefiles/{network_name}') 

    #import the network
    links, nodes = filter_to_general(studyarea, studyarea_name, networkfp,
                          network_name, network_mapper, layer,
                          desired_crs, nodesfp, nodes_layer,
                          node_id, A, B)  

    #apply filtering methods and create nodes
    filter_to_roads(links, nodes, network_name, studyarea_name)
    filter_to_bike(links, nodes, network_name, studyarea_name)
    filter_to_service(links, nodes, network_name, studyarea_name)
    
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
    print(f"The area of the {studyarea_name} study area is {sqmi} square miles.")
    gdf.plot(color='grey')
    
    return gdf

#use this to create a complete clean network
def filter_to_general(studyarea, studyarea_name, networkfp,
                      network_name, network_mapper, layer,
                      desired_crs, nodesfp, nodes_layer,
                      node_id, A, B):
    
    if layer == None:
        #explode and drop level to get rid of multi-index in abm layer
        links = gpd.read_file(networkfp, mask = studyarea).explode().droplevel(level=1).to_crs(desired_crs) 
    else:
        #explode and drop level to get rid of multi-index in abm layer
        links = gpd.read_file(networkfp, mask = studyarea, layer = layer).explode().droplevel(level=1).to_crs(desired_crs) 
     
    #add in general cleaning measures here based on network_name
    #we want to just drop all links that don't allow bikes (highways/sidewalks)
    if network_name == 'osm':
        print(f'Cleaning measures applied for {network_name}...')
        
        #remove restricted access roads + sidewalks
        restr_access = links['highway'].isin(['motorway','motorway_link'])
        links = links[-restr_access]
        
        #remove sidewalks unless bikes explicitly allowed
        remove_sidewalks = (links['footway'].isin(['sidewalk','crossing'])) & (links['bicycle'] != 'yes')
        links = links[-remove_sidewalks]
    
    elif network_name == 'abm':
        print(f'Cleaning measures applied for {network_name}...')
                
        # delete duplicate links in the abm network
        # Since ABM has reversible links, we delete B_A rows if A_B rows already exists 
        # this would help with further identification of whether a point lands on a line
        df_dup = pd.DataFrame(np.sort(links[["A","B"]], axis=1), columns=["A","B"])
        df_dup['two_way'] = df_dup.duplicated(keep=False)
        df_dup = df_dup.drop_duplicates()
        links = pd.merge(links,df_dup,how='inner', left_index = True, right_index = True, suffixes=(None,'_drop')).drop(columns={'A_drop','B_drop'})

        #remove interstates and centroid connectors
        abm_road = [10,11,14]
        links = links[links["FACTYPE"].isin(abm_road)]
    
    elif network_name == 'here':
        #remove controlled access roads and ramps
        links = links[(links['CONTRACC'].str.contains('N'))& 
                          (links['RAMP'].str.contains('N'))
                          ]
    else:
        print(f'No cleaning measures applied for {network_name}...')
   
    #nodes
    #scenario 1, links have reference ids and there is a seperate nodes layer
    #so we can just import the nodes
    if (nodesfp is not None) & ((A is not None) & (B is not None)):   
        #import nodes
        if nodes_layer == None:
            nodes = gpd.read_file(nodesfp).to_crs(desired_crs)
        else:
            nodes = gpd.read_file(nodesfp,layer=nodes_layer).to_crs(desired_crs)
        
        #rename reference id columns
        links = rename_refcol(links, network_name, A, B, network_mapper)
        nodes = rename_nodes(nodes, network_name, node_id, network_mapper)
        
        #filter out nodes that aren't in links
        nodes = filter_nodes(links, nodes, network_name)
    
    #scenario 2, links have reference ids but there is no nodes layer (HERE)
    #need to make new nodes
    elif (A is not None) & (B is not None):
        #rename reference id columns
        links = rename_refcol(links, network_name, A, B, network_mapper)
        
        #create nodes
        nodes = make_nodes(links, network_name)
    
    #scenario 3, links don't have reference ids but there is a nodes layer
    # need to use nodes layer to create reference id columns
    elif nodesfp is not None:
        #do this later
        print("not implemented yet")    
        
    #scenario 4, links don't have reference ids and there is NO nodes layer
    else:
        #create nodes and add node ids to links
        print("not implemented yet")  
        #links, nodes = create_node_ids(links, network_name, network_mapper)
        #nodes, links = make_nodes(links, network_name)
    
    #export links and nodes
    links.to_file(rf'processed_shapefiles/{network_name}/{network_name}_{studyarea_name}_base_links.geojson', driver = 'GeoJSON')
    nodes.to_file(rf"processed_shapefiles/{network_name}/{network_name}_{studyarea_name}_base_nodes.geojson", driver = 'GeoJSON')
           
    return links, nodes

# Filtering Network to Road Links

def filter_to_roads(links, nodes, network_name, studyarea_name):  
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
        print(f'No further filter needed for abm')
        road_links = links
    
    elif network_name == "here": # here network
        print(f'{network_name} road filter applied...')    
        #only allow links that allow cars and dont have speed of < 6 mph b/c those are service links
        road_links = links[(links['AR_AUTO'].str.contains('Y')) & 
                           (links['SPEED_CAT'].str.contains('8') == False)
                           ]
    else:
        print(f"No road filtering method available for {network_name}...")
        filter_specified = False

    if filter_specified == True:
        #export links
        road_links.to_file(rf'processed_shapefiles/{network_name}/{network_name}_{studyarea_name}_road_links.geojson', driver = 'GeoJSON')      
        #create nodes
        road_nodes = filter_nodes(road_links, nodes, network_name)
        #export nodes
        road_nodes.to_file(rf"processed_shapefiles/{network_name}/{network_name}_{studyarea_name}_road_nodes.geojson", driver = 'GeoJSON')

#Filtering Network to Bike Links

def filter_to_bike(links, nodes, network_name, studyarea_name):     
    filter_specified = True
    
    #filtering logic
    if network_name == "osm": # osm network
        print(f'{network_name} bike filter applied...') 
        osm_filter_method = ['cycleway','footway','path','pedestrian','steps']
        bike_links = links[links["highway"].isin(osm_filter_method)]
        
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
        bike_links.to_file(rf'processed_shapefiles/{network_name}/{network_name}_{studyarea_name}_bike_links.geojson', driver = 'GeoJSON')
        #create nodes
        nodes = filter_nodes(bike_links, nodes, network_name)
        #export nodes
        nodes.to_file(rf"processed_shapefiles/{network_name}/{network_name}_{studyarea_name}_bike_nodes.geojson", driver = 'GeoJSON')
    
    return

#filtering to service links
def filter_to_service(links, nodes, network_name, studyarea_name):
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
        service_links.to_file(rf'processed_shapefiles/{network_name}/{network_name}_{studyarea_name}_service_links.geojson', driver = 'GeoJSON')
        #create nodes
        nodes = filter_nodes(service_links, nodes, network_name)
        #export nodes
        nodes.to_file(rf"processed_shapefiles/{network_name}/{network_name}_{studyarea_name}_service_nodes.geojson", driver = 'GeoJSON')

    return

#quick matching network


# Import Network Data for Selected Study Area Function

# def import_network(networkfp, network_name, studyarea, studyarea_name, network_mapper, A, B, layer=None):
    
#     if layer==None:
        
#     #explode and drop level to get rid of multi-index in abm layer
#     links = gpd.read_file(networkfp, mask = studyarea, layer = layer).explode().droplevel(level=1).to_crs(epsg=2240) 
    
#     #add in general cleaning measures here based on network_name
#     if network_name == 'osm':
#         print(f'Cleaning measures applied for {network_name}...')
        
#     elif network_name == 'abm':
#         print(f'Cleaning measures applied for {network_name}...')
        
#         #write unmodified version to file for examining
#         #links.to_file(f'processed_shapefiles/{network_name}/{network_name}_{studyarea_name}_w_dup_links.geojson', driver = 'GeoJSON')
        
#         # delete duplicate links in the abm network
#         # Since ABM has reversible links, we delete B_A rows if A_B rows already exists 
#         # this would help with further identification of whether a point lands on a line
#         df_dup = pd.DataFrame(np.sort(links[["A","B"]], axis=1), columns=["A","B"])
#         df_dup['two_way'] = df_dup.duplicated(keep=False)
#         df_dup = df_dup.drop_duplicates()

#         links = pd.merge(links,df_dup,how='inner', left_index = True, right_index = True, suffixes=(None,'_drop')).drop(columns={'A_drop','B_drop'})

#     else:
#         print(f'No cleaning measures applied for {network_name}...')
   

#     #create nodes and add node ids to links
#     nodes, links = add_nodes(links, studyarea_name, network_name, 'base', network_mapper, A , B)  
    
#     #export links and nodes
#     links.to_file(rf'processed_shapefiles/{network_name}/{network_name}_{studyarea_name}_base_links.geojson', driver = 'GeoJSON')
#     nodes.to_file(rf"processed_shapefiles/{network_name}/{network_name}_{studyarea_name}_base_nodes.geojson", driver = 'GeoJSON')
           
#     return links


  