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
import time
from shapely.geometry import Point
from pathlib import Path
import contextily as cx
import fiona

from helper_functions import *

def import_study_area(settings):
    if settings['studyarea_layer'] is None:
        studyarea = gpd.read_file(settings['studyarea_fp'])
    else:
        studyarea = gpd.read_file(settings['studyarea_fp'],layer=settings['studyarea_layer'])
    
    #dissolve
    studyarea = studyarea.dissolve()

    #re-project if neccessary
    if studyarea.crs != settings['crs']:
        studyarea.to_crs(settings['crs'],inplace=True)

    # calculate the area of the study area in square miles
    sqmi = round(float(studyarea['geometry'].area / 5280**2),2) 
    studyarea_name = settings['studyarea_name']
    print(f"The area of the {studyarea_name} study area is {sqmi} square miles.")

    # print to examine
    ax = studyarea.plot(figsize=(10,10),alpha=0.5,edgecolor='k')
    cx.add_basemap(ax, crs=studyarea.crs)

    #create directory
    if not (settings['output_fp'] / Path(settings['studyarea_name'])).exists():
        (settings['output_fp'] / Path(settings['studyarea_name'])).mkdir()

    #export
    studyarea.to_file((settings['output_fp'] / Path(settings['studyarea_name'] / Path('network_filtering.gpkg'))),layer='studyarea')

    return studyarea

#master function used to run all the filter functions
def filter_networks(settings:dict,network_dict:dict):
    '''
    This function runs all the other functions in this file. It takes in two settings dictionaries.

    settings:


    network_dict:
    
    "studyarea": studyarea, #geodataframe of the study area
    "studyarea_name": studyarea_name, #name for the study area
    "networkfp": networkfp, #filepath for the network, specified earlier
    "network_name": 'abm', #name for the network being evaluated
    "network_mapper": network_mapper, #leave this, edit in the block above
    "layer": 0 #if network has layers, then specify which layer to look at; if no layers then leave as 0 
    "desired_crs": "desired_crs", # leave this, edit in the CRS block
    "nodesfp":None, # specify the nodes path if available, otherwise leave as None
    "node_id": None, # specify the column in the nodes file that has the node id information, if not available leave as 0
    "A": "A", #column with the starting node id; replace with None if there isn't a column
    "B": "B", #column with the ending node id; replace with None if there isn't a column
    "bbox": True, #use the bounding box of the study area as the mask for bringing in features instead of the polygon boundaries
    

    '''
    #record the starting time
    tot_time_start = time.time()
    
    network_name = network_dict['network_name']
    print(f'Filtering the {network_name} network.')

    #import the network
    links, nodes = filter_to_general(settings,network_dict)  

    #apply filtering methods and create nodes
    filter_to_roadbike(links, nodes, settings, network_name)
    filter_to_roads(links, nodes, settings, network_name)
    filter_to_bike(links, nodes, settings, network_name)
    filter_to_service(links, nodes, settings, network_name) 
    
    #print the total time it took to run the code
    print(f'{network_name} imported... took {round(((time.time() - tot_time_start)/60), 2)} minutes')


#use this to create a complete clean network
def filter_to_general(settings:dict,network_dict:dict):
    
    '''
    This first filter function removes links
    that are not legally traversable by bike
    (e.g., Interstates, sidewalks, private drives)
    '''
    studyarea = settings['studyarea']
    studyarea_name = settings['studyarea_name']
    links_fp = network_dict['links_fp']
    network_name = network_dict['network_name']

    # use bounding box to mask instead of polygon boundaries
    if settings['use_bbox'] == True:
        studyarea.geometry = studyarea.envelope
    
    if network_dict['links_layer'] is None:
        links = gpd.read_file(links_fp, mask = studyarea)
    else:
        links = gpd.read_file(links_fp, mask = studyarea, layer = network_dict['links_layer'])
    
    if links.crs != settings['crs']:
        links.to_crs(settings['crs'],inplace=True)

    #bring in or create nodes and add reference ids to links
    links, nodes = creating_nodes(links,settings,network_dict)

    #make sure ID, A, and B columns are str
    links[f'{network_name}_A'] = links[f'{network_name}_A'].astype(str)
    links[f'{network_name}_B'] = links[f'{network_name}_B'].astype(str)
    nodes[f'{network_name}_N'] = nodes[f'{network_name}_N'].astype(str)

    #create A_B column
    links[f'{network_name}_A_B'] = links[f'{network_name}_A'] + '_' + links[f'{network_name}_B']
    
    #remove directed links
    df_dup = pd.DataFrame(
        np.sort(links[[f"{network_name}_A",f"{network_name}_B"]], axis=1),
            columns=[f"{network_name}_A",f"{network_name}_B"]
            )
    
    #preserve directionality
    df_dup['two_way'] = df_dup.duplicated(keep=False)
    df_dup = df_dup.drop_duplicates()
    links = pd.merge(links,df_dup,how='inner', left_index = True, right_index = True, suffixes=(None,'_drop')).drop(columns={f'{network_name}_A_drop',f'{network_name}_B_drop'})

    #export the attributes
    export_path = settings['output_fp'] / Path(studyarea_name+'/'+network_name+'.pkl')
    links.drop(columns=['geometry']).to_pickle(export_path)

    #export raw links and nodes
    export(links,nodes,'raw',network_name,settings)

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

        #explode and drop level to get rid of multi-index in abm layer
        links = links.explode().reset_index(drop=True)

        #remove interstates and centroid connectors
        abm_road = [10,11,14]
        links = links[links["FACTYPE"].isin(abm_road)]
    
    elif network_name == 'here':
        #remove controlled access roads and ramps
        links = links[(links['CONTRACC'].str.contains('N'))& 
                          (links['RAMP'].str.contains('N'))
                          ]
    else:
        print(f'No cleaning measures defined for {network_name}. Open "filter to general.py" and add.')

    #export the general links
    export(links,nodes,'general',network_name,settings)

    return links, nodes

def creating_nodes(links:gpd.GeoDataFrame,settings:dict,network_dict:dict):
    '''
    This function creates a node layer for the links.
    If a nodes layer is already available, then use that.

    It will also assign reference ids to links if they aren't provided.
    '''
    
    #TODO have cleaner way of extractign variables from dict
    nodes_fp = network_dict['nodes_fp']
    A = network_dict['A']
    B = network_dict['B']
    nodes_layer = network_dict['nodes_layer']
    nodes_id = network_dict['nodes_id']
    network_name = network_dict['network_name']

    #conditions to check
    a_and_b = (A is not None) & (B is not None)
    nodeid_check = nodes_id is not None

    if nodes_fp is not None:
        print('There is a nodes layer...')
        
        if nodes_layer is None:
            nodes = gpd.read_file(nodes_fp)
        else:
            nodes = gpd.read_file(nodes_fp,layer=nodes_layer)

        if nodes.crs != settings['crs']:
            nodes.to_crs(settings['crs'],inplace=True)

        #check if there is a node id column
        if nodeid_check is None:
            print('Setting index as the node IDs.')
            nodes[f'{network_name}_N'] = nodes.index
        else:
            nodes.rename(columns={nodes_id:f'{network_name}_N'},inplace=True)

        if a_and_b:
            print("and links and nodes have reference ids.")
            links.rename(columns={A:f'{network_name}_A',B:f'{network_name}_B'},inplace=True)
        else:
            print('but no reference ids for links.')
            links = add_ref_ids(links,nodes,network_name)

    # no nodes layer
    else:
        print('no nodes layer')
        if a_and_b:
            print('but links have reference ids')
            links.rename(columns={A:f'{network_name}_A',B:f'{network_name}_B'},inplace=True)
            nodes = make_nodes_refid(links,network_name)
        else:
            print('and links dont have reference ids')
            nodes = make_nodes(links,network_name)
            #add node ids to links
            links = add_ref_ids(links,nodes,network_name)

    return links, nodes

#filter to only roads that bikes allowed on and remove service roads
def filter_to_roadbike(links:gpd.GeoDataFrame, nodes:gpd.GeoDataFrame, settings:dict, network_name:str):  
    '''
    This function removes service roads.
    '''
    
    filter_specified = True
    
    #filtering logic
    if network_name == "osm":
        print(f'{network_name} roadbike filter applied...')    
        
        #find service links that still have a name
        service_links_with_name = links[ (links['highway'] == 'service') & (links['name'].isnull() == False) ]
        
        osm_bike_filter_method = ['cycleway','footway','path','pedestrian','steps']
        
        osm_road_filter_method = ['primary','primary_link','residential','secondary','secondary_link',
                            'tertiary','tertiary_link','trunk','trunk_link'] 

        osm_filter_method = osm_bike_filter_method + osm_road_filter_method        

        roadbike_links = links[links["highway"].isin(osm_filter_method)]
        
        #add back in service links with a name
        roadbike_links = roadbike_links.append(service_links_with_name)
        
    elif network_name == "abm": # abm network
        print(f'No further filter needed for {network_name}')
        filter_specified = False
    
    elif network_name == "here": # here network
        print(f'{network_name} roadbike filter applied...')    
        #only allow links that allow cars and dont have speed of < 6 mph b/c those are service links
        roadbike_links = links[(links['SPEED_CAT'].str.contains('8') == False)]
    
    else:
        print(f'No roadbike filter for {network_name}. Open "network_filter.py" and add it to the filter_to_roadbike function.')
        filter_specified = False

    if filter_specified:
        export(roadbike_links,nodes,'roadbike',network_name,settings)
    return
    
def filter_to_roads(links, nodes, settings, network_name):  
    '''
    The function filters the network to only public roads (no bike paths).
    '''
    
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
        filter_specified = False
    
    elif network_name == "here": # here network
        print(f'{network_name} road filter applied...')    
        #only allow links that allow cars and dont have speed of < 6 mph b/c those are service links
        road_links = links[(links['AR_AUTO'].str.contains('Y')) & 
                           (links['SPEED_CAT'].str.contains('8') == False)
                           ]
    else:
        print(f'No road filter for {network_name}. Open "network_filter.py" and add it to the filter_to_road function.')
        filter_specified = False

    if filter_specified == True:
        export(road_links,nodes,'road',network_name,settings)
    
    return
        
def filter_to_bike(links, nodes, settings, network_name):     
    '''
    Filter to network to bike-only links (no public roads or service roads).
    '''
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
        print(f'No bike filter for {network_name}. Open "network_filter.py" and add it to the filter_to_bike function.')
        filter_specified = False

    if filter_specified == True:
        export(bike_links,nodes,'bike',network_name,settings)
    return

def filter_to_service(links, nodes, settings, network_name):
    '''
    Filter to network to service links (driveways, parking lots, alleys).
    '''
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
        print(f'No service filter for {network_name}. Open "network_filter.py" and add it to the filter_to_service function.')
        filter_specified = False
    
    if filter_specified == True:
        export(service_links,nodes,'service',network_name,settings)
    return

# Extract Start and End Points as tuples and round to reduce precision
def start_node(row, geom):
   #basically look at x and then y coord, use apply to do this for every row of a dataframe
   return (round(row[geom].coords.xy[0][0],5), round(row[geom].coords.xy[1][0],5))
def end_node(row, geom):
   return (round(row[geom].coords.xy[0][-1],5), round(row[geom].coords.xy[1][-1],5))

# Extract start and end points but turn them into shapely Points
def start_node_geo(row, geom):
   return (Point(row[geom].coords.xy[0][0], row[geom].coords.xy[1][0])) 
def end_node_geo(row, geom):
   return (Point(row[geom].coords.xy[0][-1], row[geom].coords.xy[1][-1]))


def add_ref_ids(links,nodes,network_name):
    '''
    This function adds reference columns to links.
    '''
    for_matching = links.copy()
    #make the first point the active geometry
    for_matching['geometry'] = for_matching.apply(start_node_geo, geom='geometry', axis=1)
    for_matching.set_geometry('geometry',inplace=True)
    #find nearest node from starting node and add to column
    links[f'{network_name}_A'] = ckdnearest(for_matching,nodes,return_dist=False)[f'{network_name}_N']
    
    #repeat for end point
    for_matching = links.copy()
    #make the first point the active geometry
    for_matching['geometry'] = for_matching.apply(end_node_geo, geom='geometry', axis=1)
    for_matching.set_geometry('geometry',inplace=True)
    #find nearest node from starting node and add to column
    links[f'{network_name}_B'] = ckdnearest(for_matching,nodes,return_dist=False)[f'{network_name}_N']

    #check for missing reference ids
    if links[f'{network_name}_A'].isnull().any() | links[f'{network_name}_B'].isnull().any():
        print("There are missing reference ids")
    else:
        print("Reference IDs successfully added to links.")
    return links

def make_nodes_refid(links, network_name): 
    '''
    This function creates a nodes layer from links with reference ids.
    '''
    #starting point
    nodes_A = links.copy()
    nodes_A.geometry = nodes_A.apply(start_node_geo,geom='geometry',axis=1)
    nodes_A = nodes_A[[f'{network_name}_A','geometry']]
    nodes_A.rename(columns={f'{network_name}_A':f'{network_name}_N'},inplace=True)
    #ending point
    nodes_B = links.copy()
    nodes_B.geometry = nodes_B.apply(end_node_geo,geom='geometry',axis=1)
    nodes_B = nodes_B[[f'{network_name}_B','geometry']]
    nodes_B.rename(columns={f'{network_name}_B':f'{network_name}_N'},inplace=True)

    #append
    nodes = pd.concat([nodes_A,nodes_B],ignore_index=True)

    #drop duplicates
    nodes.drop_duplicates(subset=[f'{network_name}_N'],inplace=True)

    return nodes

def make_nodes(links, network_name):
    links_copy = links.copy()

    #turn to unprojected CRS
    orig_crs = links_copy.crs
    links_copy.to_crs("epsg:4326",inplace=True)

    #extract start and end node, eliminate duplicates, turn into points    
    #add start and end coordinates to each line
    start_nodes = links_copy.apply(start_node, geom='geometry', axis=1)
    end_nodes = links_copy.apply(end_node, geom='geometry', axis=1)

    nodes = pd.concat([start_nodes,end_nodes],ignore_index=True)
    nodes.drop_duplicates(inplace=True)

    #turn into dataframe
    nodes = pd.DataFrame({f'{network_name}_N':range(0,len(nodes)),'geometry':nodes})
    
    #turn the WKT coordinates into points
    nodes['geometry'] = nodes.apply(lambda row: Point([row['geometry']]), axis=1)
        
    #convert to GeoDataFrame and set geo and CRS
    nodes = gpd.GeoDataFrame(nodes,geometry='geometry',crs="epsg:4326").to_crs(orig_crs)
    
    return nodes

def filter_nodes(links,nodes,network_name):
    #remove nodes that aren't in filtered links
    nodes_in = set(links[f'{network_name}_A'].append(links[f'{network_name}_B']))
    nodes_filt = nodes[nodes[f'{network_name}_N'].isin(nodes_in)]
    return nodes_filt

def export(links,nodes,network_type,network_name,settings):
    print(f'Exporting {network_name} {network_type} layer.')
    start_time = time.time()
    studyarea_name = settings['studyarea_name']
    #filter nodes
    nodes = filter_nodes(links,nodes,network_name)
    #remove excess columns
    links = links[[f'{network_name}_A',f'{network_name}_B',f'{network_name}_A_B','geometry']]
    nodes = nodes[[f'{network_name}_N','geometry']]
    #export
    export_fp = settings['output_fp'] / Path(f'{studyarea_name}/filtered.gpkg')
    links.to_file(export_fp,layer=f'{network_name}_links_{network_type}')
    nodes.to_file(export_fp,layer=f'{network_name}_nodes_{network_type}')
    export_time = round(((time.time() - start_time)/60), 2)
    print(f'Took {export_time} minutes to export {network_name} {network_type} layer.')

#TODO fix this function
def summary(settings):
    
    output_fp = settings['output_fp']
    studyarea_name = settings['studyarea_name']
    
    #summary table
    #can add other metrics of interest in the future
    summary_table = pd.DataFrame(columns=['network','link_type','num_links','num_nodes','tot_link_length','avg_link_length'])
    
    #expected link types
    layers = fiona.listlayers(output_fp / Path(f'{studyarea_name}/filtered.gpkg'))

    #remove node layers
    layers = [x for x in layers if 'node' not in x]

    #go through each network
    for network in layers:
        links = gpd.read_file(output_fp / Path(f'{studyarea_name}/filtered.gpkg'),layer=network)
    
        network_name = network.split('_')[0]
    
        #how many links
        num_links = len(links)
    
        #how many nodes
        nodes = pd.concat([links[f'{network_name}_A'],links[f'{network_name}_B']],ignore_index=True)
        num_nodes = len(nodes.unique())

        #total length
        length_mi = links.geometry.length / 5280 # create a new distance column and calculate mileage of each link
        tot_link_length = round(length_mi.sum(),0)

        #average link length
        avg_link_length = round(links.geometry.length.mean(),1)

        #create data frame
        summary_table.loc[network,:] = [num_links,num_nodes,tot_link_length,avg_link_length]

    #export summary table
    summary_table.to_csv(output_fp / Path(f"{studyarea_name}/network_summary.csv"))
   
    print(summary_table)


# working_dir = Path.home() / Path('Documents/NewBikewaySimData')
# settings = {
#     'output_fp': working_dir, #where filtered network files will output
#     'crs': "EPSG:2240", # project all spatial data to this CRS
#     'studyarea_fp': working_dir / Path('Data/Study Areas/bikewaysim_studyarea.geojson'),
#     'studyarea_name': 'bikewaysim',
#     'studyarea_layer': None, #replace with study area layer name if file is gpkg or gdb
#     'use_bbox': False # use bounding box instead of studyarea polygon boundaries as mask for importing links
# }

# summary(settings)
