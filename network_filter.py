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


from scipy.spatial import cKDTree

#take in two geometry columns and find nearest gdB point from each
#point in gdA. Returns the matching distance too.
#MUST BE A PROJECTED COORDINATE SYSTEM
def ckdnearest(gdA, gdB, return_dist=True):  
    
    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].reset_index(drop=True)
    
    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest,
            pd.Series(dist, name='dist')
        ], 
        axis=1)
    
    if return_dist == False:
        gdf = gdf.drop(columns=['dist'])
    
    return gdf


#master function used to run all the filter functions
def filter_networks(studyarea, studyarea_name, networkfp,
                    network_name, network_mapper, layer,
                    desired_crs, nodesfp, nodes_layer = None,
                    node_id = None, A = None, B = None, bbox = False):

    #record the starting time
    tot_time_start = time.time()
    
    #create a new folder for the network if it doesn't already exist
    if not os.path.exists(f'processed_shapefiles/{studyarea_name}'):
        os.makedirs(f'processed_shapefiles/{studyarea_name}') 

    #import the network
    links, nodes = filter_to_general(studyarea, studyarea_name, networkfp,
                          network_name, network_mapper, layer,
                          desired_crs, nodesfp, nodes_layer,
                          node_id, A, B, bbox = bbox)  

    #apply filtering methods and create nodes
    filter_to_roadbike(links, nodes, network_name, studyarea_name)
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

def cleaning_process(links, nodes, network_name):
           
    #use mask to only keep neccessary columns
    links = links[[f'{network_name}_A',f'{network_name}_B',f'{network_name}_A_B','geometry']]
    nodes = nodes[[f'{network_name}_ID','geometry']]

    return links, nodes



#use this to create a complete clean network
def filter_to_general(studyarea, studyarea_name, networkfp,
                      network_name, network_mapper, layer,
                      desired_crs, nodesfp, nodes_layer,
                      node_id, A, B, bbox = False):
    
    # get bounding box instead of polygon boundaries
    if bbox == True:
        studyarea.geometry = studyarea.envelope
    
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
        #read in nodes
        nodes = gpd.read_file(nodesfp)
        
        #add ref ids from nodes
        links = add_ref_ids(nodes,)
        
        
    #scenario 4, links don't have reference ids and there is NO nodes layer
    else:
        #create nodes and add node ids to links
        print("not implemented yet")  
        #links, nodes = create_node_ids(links, network_name, network_mapper)
        #nodes, links = make_nodes(links, network_name)
    
    #export links and nodes
    links.to_file(rf'processed_shapefiles/{studyarea_name}/{network_name}_network.gpkg',layer='base_links',driver = 'GPKG')
    nodes.to_file(rf"processed_shapefiles/{studyarea_name}/{network_name}_network.gpkg",layer='base_nodes',driver = 'GPKG')
    
    return links, nodes

#filter to only roads that bikes allowed on and remove service roads
def filter_to_roadbike(links, nodes, network_name, studyarea_name):  
    filter_specified = True
    
    #filtering logic
    if network_name == "osm": # osm network
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
        roadbike_links = links
    
    elif network_name == "here": # here network
        print(f'{network_name} roadbike filter applied...')    
        #only allow links that allow cars and dont have speed of < 6 mph b/c those are service links
        roadbike_links = links[(links['SPEED_CAT'].str.contains('8') == False)]
    else:
        print(f"No road filtering method available for {network_name}...")
        filter_specified = False

    if filter_specified == True:
        #create nodes
        roadbike_nodes = filter_nodes(roadbike_links, nodes, network_name)
        #cleaning_process
        roadbike_links,roadbike_nodes = cleaning_process(roadbike_links,roadbike_nodes,network_name)
        #export links
        roadbike_links.to_file(rf'processed_shapefiles/{studyarea_name}/{network_name}_network.gpkg',layer='roadbike_links', driver = 'GPKG')   
        #export nodes
        roadbike_nodes.to_file(rf"processed_shapefiles/{studyarea_name}/{network_name}_network.gpkg",layer='roadbike_nodes', driver = 'GPKG')

    
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
        #create nodes
        road_nodes = filter_nodes(road_links, nodes, network_name)
        #cleaning_process
        road_links,road_nodes = cleaning_process(road_links,road_nodes,network_name)
        #export links
        road_links.to_file(rf'processed_shapefiles/{studyarea_name}/{network_name}_network.gpkg',layer='road_links', driver = 'GPKG')   
        #export nodes
        road_nodes.to_file(rf"processed_shapefiles/{studyarea_name}/{network_name}_network.gpkg",layer='road_nodes', driver = 'GPKG')
        
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
        #create nodes
        bike_nodes = filter_nodes(bike_links, nodes, network_name)
        #cleaning_process
        bike_links,bike_nodes = cleaning_process(bike_links,bike_nodes,network_name)
        #export links
        bike_links.to_file(rf'processed_shapefiles/{studyarea_name}/{network_name}_network.gpkg',layer='bike_links', driver = 'GPKG')
        #export nodes
        bike_nodes.to_file(rf"processed_shapefiles/{studyarea_name}/{network_name}_network.gpkg",layer='bike_nodes', driver = 'GPKG')
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
        #create nodes
        service_nodes = filter_nodes(service_links, nodes, network_name)
        #cleaning_process
        service_links,service_nodes = cleaning_process(service_links,service_nodes,network_name)
        #export links
        service_links.to_file(rf'processed_shapefiles/{studyarea_name}/{network_name}_network.gpkg',layer='service_links', driver = 'GPKG')
        #export nodes
        service_nodes.to_file(rf"processed_shapefiles/{studyarea_name}/{network_name}_network.gpkg",layer='service_nodes',driver = 'GPKG')
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


# Creating Reference IDs

def add_ref_ids(links,nodes):
    #get start/end geom of links
    #match id to starting node
    links['start_point_geo'] = links.apply(start_node_geo, geom= links.geometry.name, axis=1)
    #set to active geo
    links = links.set_geometry('start_point_geo')
    #find nearest node from starting node
    links = ckdnearest(links,nodes,return_dist=False)
    #rename id columns to _A
    links.columns = pd.Series(list(links.columns)).str.replace('_ID','_A')
    #remove start node and base_node_geo columns
    links = links.drop(columns=['start_point_geo',nodes.geometry.name])
    #reset geometry
    links = links.set_geometry(links_geo)
 
    #do same for end point
    links['end_point_geo'] = links.apply(end_node_geo, geom= links.geometry.name, axis=1)
    #set active geo
    links = links.set_geometry('end_point_geo')
    #find nearest node from starting node
    links = ckdnearest(links,nodes,return_dist=False)
    #rename id columns to _A
    links.columns = pd.Series(list(links.columns)).str.replace('_ID','_B')
    #remove end point
    links = links.drop(columns=['end_point_geo',nodes.geometry.name])
    #reset geometry   
    links = links.set_geometry(links_geo)
    
    #check for missing ref ids
    cols = list(links.columns)
    a_cols = [cols for cols in cols if "_A" in cols]
    b_cols = [cols for cols in cols if "_B" in cols]
    #first see any As are missing
    a_missing = links[a_cols].apply(lambda row: row.isnull().all(), axis = 1)
    #then see if any Bs are missing
    b_missing = links[b_cols].apply(lambda row: row.isnull().all(), axis = 1)
    if a_missing.any() == True | b_missing.any() == True:
        print("There are missing reference ids")
    else:
        print("Reference IDs successfully added to links.")
    return links

# use this to create node id if no nodes provided
def create_node_ids(links, network_name, network_mapper):
    #need to create new nodes IDs for OSM and TIGER networks
    #extract nodes, find unique nodes, number sequentially, match ids back to links start end nodes
    
    #turn to unprojected and then round
    orig_crs = links.crs
    links = links.to_crs("epsg:4326")
    
    #add start and end coordinates to each line so we can match them
    links['start_node'] = links.apply(start_node, geom= links.geometry.name, axis=1)
    links['end_node'] = links.apply(end_node, geom= links.geometry.name, axis=1)

    #create one long list of nodes to find unique nodes
    nodes_coords = links['start_node'].append(links['end_node'])
    
    #turn series into data frame
    nodes = pd.DataFrame({f'{network_name}_coords':nodes_coords})
    
    #find number of intersecting links
    nodes[f'{network_name}_num_links'] = 1 #this column will be used to count the number of links
    
    #for each unique ID and coordinate, count number of links
    nodes = nodes.groupby([f'{network_name}_coords'], as_index=False).count()
    
    #give nodes row number name
    #CAUTION, node IDs will change based on study area, and/or ordering of the data 
    nodes[f'{network_name}_ID'] = np.arange(nodes.shape[0]).astype(str)
    nodes[f'{network_name}_ID'] = network_mapper[network_name] + network_mapper['generated'] + nodes[f'{network_name}_ID']
    
    #extract only the ID and coords column for joining
    nodes = nodes[[f'{network_name}_ID',f'{network_name}_coords']]
    
    #rename coords to geometry
    nodes = nodes.rename(columns={f'{network_name}_coords':'geometry'})
    
    #turn nodes into gdf
    nodes = gpd.GeoDataframe(nodes,geometry='geometry',crs="epsg:4326").to_crs(orig_crs)
    
    #perform join back to original dataframe
    #rename id to be A, rename coords to match start_node
    joining = nodes.rename(columns={f'{network_name}_ID':f'{network_name}_A',f'{network_name}_coords':'start_node'})
    links = pd.merge(links, joining, how = 'left', on = 'start_node' )

    #rename id to be B, rename coords to match end_node
    joining = nodes.rename(columns={f'{network_name}_ID':f'{network_name}_B',f'{network_name}_coords':'end_node'})
    links = pd.merge(links, joining, how = 'left', on = 'end_node' )
    links = links.rename(columns={f'{network_name}_ID':f'{network_name}_B'})
    
    #drop tuple columns
    links = links.drop(columns=['start_node','end_node'])

    #create an A_B column
    links[f'{network_name}_A_B'] = links[f'{network_name}_A'] + '_' + links[f'{network_name}_B']

    return links, nodes

#use to rename the refid columns in the links
def rename_refcol(links, network_name, A, B, network_mapper):  
    #renames node ID column name to network name _ A/_B
    links = links.rename(columns={A:f'{network_name}_A',B:f'{network_name}_B'})

    #adds numbers to the beginning from network mapper
    links[f'{network_name}_A'] = network_mapper[network_name] + network_mapper['original'] + links[f'{network_name}_A'].astype(str)
    links[f'{network_name}_B'] = network_mapper[network_name] + network_mapper['original'] + links[f'{network_name}_B'].astype(str)
    
    #create an A_B column
    links[f'{network_name}_A_B'] = links[f'{network_name}_A'] + '_' + links[f'{network_name}_B']
    return links

#rename the node id column
def rename_nodes(nodes, network_name, node_id, network_mapper):  
    #renames node ID column name to network name _ A/_B
    nodes = nodes.rename(columns={node_id:f'{network_name}_ID'})
    #adds numbers to the beginning from network mapper
    nodes[f'{network_name}_ID'] = network_mapper[network_name] + network_mapper['original'] + nodes[f'{network_name}_ID'].astype(str)
    return nodes

# create nodes layer from links
def make_nodes(links, network_name): 

    #turn to unprojected and then round
    orig_crs = links.crs
    links = links.to_crs("epsg:4326")

    #extract start and end node, eliminate duplicates, turn into points    
    #add start and end coordinates to each line
    links['start_node'] = links.apply(start_node, geom=links.geometry.name, axis=1)
    links['end_node'] = links.apply(end_node, geom=links.geometry.name, axis=1)

    #stack start/end node coords and IDs on top of each other
    nodes_id = links[f'{network_name}_A'].append(links[f'{network_name}_B'])
    nodes_coords = links['start_node'].append(links['end_node'])
    
    #turn into dataframe
    nodes = pd.DataFrame({f"{network_name}_ID":nodes_id,f"{network_name}_coords":nodes_coords})
    
    #turn the coordinates into points so we can do spatial mapping
    nodes[f'{network_name}_coords'] = nodes.apply(lambda row: Point([row[f'{network_name}_coords']]), axis=1)
    
    #rename geometry column
    nodes = nodes.rename(columns={f'{network_name}_coords':'geometry'})
    
    #convert to GeoDataFrame and set geo and CRS
    nodes = gpd.GeoDataFrame(nodes).set_geometry('geometry').set_crs("epsg:4326").to_crs(orig_crs)
    
    #drop duplicates
    nodes = nodes.drop_duplicates()
    
    return nodes

def filter_nodes(links,nodes,network_name):
    #remove nodes that aren't in filtered links
    nodes_in = links[f'{network_name}_A'].append(links[f'{network_name}_B']).unique()
    nodes_filt = nodes[nodes[f'{network_name}_ID'].isin(nodes_in)]
    return nodes_filt