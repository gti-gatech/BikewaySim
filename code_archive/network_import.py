# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 22:42:19 2021

@author: Daisy, Reid
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

#%% Network mapper
#this is how network node ID's will be identified and coded
#the first letter in the network ID represents its origin network
#the second letter in the network ID represent its link type
#all numbers after that are the original network ID 
network_mapper = {
    "abm": "1",
    "navstreets": "2",
    "osm": "3",
    "base": "0",
    "road": "1",
    "bike": "2",
    "service": "3"
}

#%% run network import function

def run_network_import(studyareafp, studyarea_name, networkfp, network_name, link_node, A = None, B = None, layer = 0, nodesfp=None, road = True, bike = True, service = True):
    
    #studyareafp = file path for the study area spatial file
    #studyarea_name = name for the study area
    #studyarea_crs = coordinate reference system of study area spatial file
    #networkfp = file path for the network spatial file
    #network_name = name of the network
    #network_crs = coordinate reference system of the network shapefile
    #link_node = do a dissolve and multipart to single part GIS operation that seperates intersecting links
    #A = insert name of the A (starting node ID) column in the network links file
    #B = ^^same as A but for B
    #layer = change this if the network shapefile has multiple layers 
    
    #record the starting time
    tot_time_start = time.time()
    
    #create files
    #create a new folder for the network if it doesn't already exist
    if not os.path.exists(f'Processed_Shapefiles/{network_name}'):
        os.makedirs(f'Processed_Shapefiles/{network_name}')  
    
    #import the study area shapefile/geojson/
    studyarea = import_study_area(studyareafp, studyarea_name)
    
    #import the network
    network = import_network(networkfp, network_name, studyarea, studyarea_name, layer)  
    
    #create a pickle of fields to ignore for subsequent runs
    ignore_fields(network, network_name)

    #if there's a nodefp give load in here
    if nodesfp is not None:
        network_nodes = gpd.read_file(nodesfp, mask=studyarea).to_crs(epsg=2240)
        
        #clean the node ids for osm nodes
        if network_name == 'osm':
            trim_node_id = lambda row: row['id'].split("/")[1]
            network_nodes['id'] = network_nodes.apply(trim_node_id, axis = 1)
        
    else:
        network_nodes = None
    
    #export base nodes
    export_nodes_links(network, 'base', nodes, network_name, studyarea_name, A, B)
    
    if road == True:
        #filtering steps
        network_road = filter_to_roads(network, network_name, studyarea_name, link_node)
        #this function gives the network an A and B column and creates a set of nodes and exports both
        export_nodes_links(network_road, 'road', network_nodes, network_name, studyarea_name, A, B)
        del[network_road]
        
    if bike == True:
        network_bike = filter_to_bike(network, network_name, studyarea_name, link_node)
        export_nodes_links(network_bike,'bike',network_nodes,network_name,studyarea_name,A,B)
        del[network_bike]
    if service == True:
        network_service = filter_to_service(network, network_name, studyarea_name, link_node)
        export_nodes_links(network_service,'service',network_nodes,network_name,studyarea_name,A,B)
        del[network_service]
    
    #print out some summary statistics about the network
    #network_sumdata = summurize_network(network, network_name)
    
    #took 6.39 mins before
    #now takes 

    #print the total time it took to run the code
    print(f'{network_name} imported... took {round(((time.time() - tot_time_start)/60), 2)} minutes')

#%% deal with node ids

def export_nodes_links(links,link_type,nodes,network_name,studyarea_name,A,B):

        #find node ids for osm, do this at this step because osm was broken up in filtering process
        if network_name == 'osm': 
            #hardcode for osm
            node_id = 'id'
            
            #need to match osm node ids to osm links
            links = add_nodeids_to_links(links,nodes,node_id,network_name)
            A = 'A'
            B = 'B'
  
        if A and B is not None:
            links = rename_nodes(links, network_name, link_type, A, B)
        else:
            print('A and B is None')
            #don't need create_node_ids code
            links = create_node_ids(links, network_name, link_type)
            
        links.to_file(f'Processed_Shapefiles/{network_name}/{network_name}_{studyarea_name}_{link_type}.geojson', driver = 'GeoJSON')  
        nodes = make_nodes(links,network_name, link_type, studyarea_name)
        
        del[links,nodes]
        
#%% Import Study Area Function
def import_study_area(studyareafp, studyarea_name):
 
    gdf = gpd.read_file(studyareafp).to_crs(epsg=2240) #import shp and project
    
    # calculate the area of the study area in square miles
    sqmi = round(float(gdf['geometry'].area / 5280**2),2) 
    print(f"Done... The area of the {studyarea_name} boundary is {sqmi} square miles.")

    #create a new folder for the network if it doesn't already exist
    if not os.path.exists('Processed_Shapefiles/study_areas'):
        os.makedirs('Processed_Shapefiles/study_areas')  
    gdf.to_file(rf'Processed_Shapefiles/study_areas/{studyarea_name}.geojson', driver='GeoJSON')

    return gdf

#%% Import Network Data for Selected Study Area Function

def import_network(networkfp, network_name, studyarea, studyarea_name, layer = 0):
    
    #ignore fields, use this to ignore fields that are not needed; it can speed up the read/write process
    # if (network_name == 'osm') and (os.path.exists(rf'Ignore_Fields/{network_name}_ignore_empty.pkl') == True):
    #     print(f'Field ignore applied for {network_name}...')
    #     with open('Ignore_Fields/{network_name}_ignore_empty.pkl', 'rb') as f:
    #         cols = pickle.load(f)
    #     links = gpd.read_file(fp, mask = clip_area, layer = layer, ignore_fields = cols).set_crs(f'EPSG:{network_crs}').to_crs(epsg:2240) #import network links and project
    # else:
    #     #import network links and project
    #     #explode to convert multipart geos to singlpart, this is needed for the node extraction to work
    #     #drop level gets rid of the second index
    #     links = gpd.read_file(fp, mask = clip_area, layer = layer).set_crs(f'EPSG:{network_crs}').explode().droplevel(level=1)

    links = gpd.read_file(networkfp, mask = studyarea, layer = layer).explode().droplevel(level=1).to_crs(epsg=2240) 

    #add in general cleaning measures here based on network_name
    if network_name == 'osm':
        print(f'Cleaning measures applied for {network_name}...')
        #if there are features that enclose themselves (usually sidewalk blocs and circular paths in parks), convert them to linestring by taking the exterior
        osm_polygons_to_links = links[links.geometry.type == "Polygon"].boundary.geometry
        links.loc[osm_polygons_to_links.index,'geometry'] = osm_polygons_to_links
        links = links[links.geometry.type == "LineString"] #make sure output is only LineStrings
        
    elif network_name == 'abm':
        print(f'Cleaning measures applied for {network_name}...')
        # delete duplicate links in the abm network
        # Since ABM has reversible links, we delete B_A rows if A_B rows already exists 
        # this would help with further identification of whether a point lands on a line
        df_dup = pd.DataFrame(np.sort(links[["A","B"]], axis=1), columns=["A","B"]).drop_duplicates()   
        links = links.merge(df_dup,how='inner', left_index = True, right_index=True, suffixes=(None,'drop'))
    else:
        print(f'No cleaning measures applied for {network_name}...')

    return links

#%% Extract Start and End Points Code from a LineString as points
def start_node_point(row, geom):
   return (Point(row[geom].coords.xy[0][0], row[geom].coords.xy[1][0])) #basically look at x and then y coord, use apply to do this for every row of a dataframe

def end_node_point(row, geom):
   return (Point(row[geom].coords.xy[0][-1], row[geom].coords.xy[1][-1]))


#this function will match the broken up OSM links with all the possible OSM nodes available

def add_nodeids_to_links(links,nodes,node_id,network_name):
   #this function takes in a set of links and creates an A and B column in the link gdf
   #using the specified id column in the nodes gdf
   #currently used for OSM
   
   #turn link index into column to serve as link id
   links['link_id'] = links.index
   
   #create very small buffer on nodes in nodes gdf and set active geo to buffer
   nodes['buffer'] = nodes.buffer(0.001)
   nodes = nodes.filter([f'{node_id}','buffer']).rename(columns={f'{node_id}':'node_id'})
   nodes = nodes.set_geometry('buffer')

   #start nodes first
   links_start = links.copy()
   links_start['start_node'] = links_start.apply(start_node_point, geom=links_start.geometry.name, axis=1)
   links_start = links_start.set_geometry('start_node').set_crs(epsg=2240)

   #intersect start nodes with nodes
   links_start = gpd.overlay(links_start, nodes, how="intersection")

   #merge with links and rename to A
   links = pd.merge(links, links_start[['link_id','node_id']], on = 'link_id', how = 'left').rename(columns={'node_id':'A'})

   #end nodes next
   links_end = links.copy()
   links_end['end_node'] = links_end.apply(end_node_point, geom= links_end.geometry.name, axis=1)
   links_end = links_end.set_geometry('end_node').set_crs(epsg=2240)
   links_end = gpd.overlay(links_end, nodes, how='intersection')
   links = pd.merge(links, links_end[['link_id','node_id']], on = 'link_id', how = 'left').rename(columns={'node_id':'B'})

   #find max node id
   maximum = nodes['node_id'].int().max()

   #check for nan and replace with new ids
   links['A'] == '32nan'
   links['B'] == '32nan' 


   return links

#this function is meant for OSM. It takes network dataset and will breakup links that are too long
#NOTE: if there isn't supposed to be an intersection between two links, (e.g. bridge over road) it will still create a node
# to minimize this, make sure to only use this on pre filtered OSM data. For instance, in this case it will only be used after the OSM
# data has been reduced to road/service/bike links

def link_node_struct(links, network_name):
    links['dissolve'] = 1 # not sure if neccessary, but dissolve was giving errors without accepting some arguement
    multipart_links = links.dissolve(by='dissolve') #this will combine all the geos of each row by the dissolve column
    singlepart_links = pd.Series(multipart_links.iloc[0].geometry).tolist() # multipart to singleparts
    gdf = gpd.GeoDataFrame({'geometry':singlepart_links}, geometry='geometry').set_crs(epsg=2240)
    
    #now need to join attribute information back in by doing a buffer + intersection with orginal network
    gdf['original_length'] = gdf.length
    
    #create buffer
    links['buffer'] = links.buffer(0.001) #make a small buffer around the original layer
    links = links.set_geometry('buffer') #make the buffer the primary geometry
    
    #perform intersection
    res_intersection = gpd.overlay(gdf, links, how='intersection')
    res_intersection['intersected_length'] = res_intersection.length
    res_intersection['percent_overlap'] =  res_intersection['intersected_length'] / res_intersection['original_length']

    #filter by only retaining matches that had very high overlap
    res_intersection_filt = res_intersection[res_intersection['percent_overlap'] >= 0.99]
    

    #make sure it's singlepart geometry
    res_intersection_filt = res_intersection_filt.explode().droplevel(level=1)


    # gdf = res_intersection_filt

    # #need to create new node ids for splitted lines if ones don't exist
    # if network_name == 'osm':
    #     osm_nodesfp = r'Base_Shapefiles/osm/osm_nodes_split.geojson'
    #     gdf = add_nodeids_to_links(gdf,osm_nodesfp,'id','id','osm')
        
    #     #trim node id down to number and export
           
    #     trim_node_id_a = lambda row: row['A'].split("/")[1]
    #     trim_node_id_b = lambda row: row['B'].split("/")[1]
    #     gdf['A'] = gdf.apply(trim_node_id_a)
    #     gdf['B'] = gdf.apply(trim_node_id_b)
        
    # else:
    #     print('need to create new nodes')
    #     #gdf = create_node_ids(res_intersection_filt, network_name)

    #     #rename old node id columns
    #     # res_intersection_filt = res_intersection_filt.rename(columns={f'{network_name}_A':f'{network_name}_A_orig',
    #     #                                                          f'{network_name}_B':f'{network_name}_B_orig',
    #     #                                                          f'{network_name}_A_B':f'{network_name}_A_B_orig'
    #     #                                                          })
    
    return res_intersection_filt



#%% Filtering Network to Road Links Function

#put in file path, network name, area to clip by, and the name of the clip area, use last parameter to identify desired GDB layer
def filter_to_roads(links, network_name, studyarea_name, link_node):  
    
    #filtering logic
    print(f'Filtering {network_name} to road network...')
    
    #filtering logic
    if network_name == "osm": # osm network
        print(f'{network_name} road filter applied...')    
        osm_filter_method = ['primary','primary_link','residential','secondary','secondary_link',
                            'tertiary','tertiary_link','trunk','trunk_link'] 
        links_road = links[links["highway"].isin(osm_filter_method)]
        
    elif network_name == "abm": # abm network
        print(f'{network_name} road filter applied...')
        abm_road = [10,11,14]
        links_road = links[links["FACTYPE"].isin(abm_road)]
        
    elif network_name == "navstreets": # navstreets network ## in future if there are more layers just modify this if condition
        print(f'{network_name} road filter applied...')    
        links_road = links[(links['CONTRACC'].str.contains('N'))&
                          (links['SPEED_CAT'].str.contains('8') == False)&
                          (links['AR_AUTO'].str.contains('Y'))&
                          (links['RAMP'].str.contains('N'))]  
    else:
        print(f"No road filtering method available for {network_name}...")
        links_road = pd.DataFrame()

    if link_node == True:
        #apply link node function if neccessary and create sub node IDs
        links_road = link_node_struct(links_road, network_name)
    
    return links_road

#%% Filtering Network to Bike Links

def filter_to_bike(links, network_name, studyarea_name, link_node):     
    #filtering logic
    print(f'Filtering {network_name} to bike network...')
    
    if network_name == "osm": # osm network
        print(f'{network_name} filter applied...') 
        osm_filter_method = ['cycleway','footway','path','pedestrian','steps']
        links_bike = links[links["highway"].isin(osm_filter_method)]
        links_bike = links_bike[links_bike['footway'] != 'sidewalk']
        links_bike = links_bike[links_bike['footway'] != 'crossing']
        
    elif network_name == "abm": # abm network
        print(f'No bike links present for {network_name}')
        links_bike = pd.DataFrame()
        
    elif network_name == "navstreets": # navstreets network ## in future if there are more layers just modify this if condition
        print(f'{network_name} filter applied...') 
        links_bike = links[ links['AR_AUTO'].str.contains('N') ]
    else:
        print(f"No filtering method available for {network_name}.")
        links_bike = pd.DataFrame()
    
    if link_node == True:
        links_bike = link_node_struct(links_bike, network_name)
        
    return links_bike

def filter_to_service(links, network_name, studyarea_name, link_node):
    #filtering logic
    print(f'Filtering {network_name} to service network...')
    
    if network_name == "osm": # osm network
        print(f'{network_name} filter applied...') 
        osm_filter_method = ['service']
        links_service = links[links["highway"].isin(osm_filter_method)]
        
    elif network_name == "abm": # abm network
        print(f'No service links present for {network_name}')
        links_service = pd.DataFrame()
        
    elif network_name == "navstreets": # navstreets network ## in future if there are more layers just modify this if condition
        print(f'{network_name} filter applied...') 
        links_service = links[ (links['AR_AUTO'].str.contains('Y')) & (links['SPEED_CAT'].str.contains('8')) ]

    else:
        print(f"No filtering method available for {network_name}.")
        links_service = pd.DataFrame()
    
    if link_node == True:
        links_service = link_node_struct(links_service, network_name)

    return links_service   

#%% Extract Start and End Points Code from a LineString as tuples
def start_node(row, geom):
   return (row[geom].coords.xy[0][0], row[geom].coords.xy[1][0]) #basically look at x and then y coord, use apply to do this for every row of a dataframe

def end_node(row, geom):
   return (row[geom].coords.xy[0][-1], row[geom].coords.xy[1][-1])


#%% create node ID's function
#need to create new nodes IDs for OSM and TIGER
#first will need to extract nodes
#CAUTION, node IDs will change based on study area, and/or ordering of the data 
def create_node_ids(links, network_name, link_type):
    
    #add start and end coordinates to each line
    #this is so we can match them  
    links['start_node'] = links.apply(start_node, geom= links.geometry.name, axis=1)
    links['end_node'] = links.apply(end_node, geom= links.geometry.name, axis=1)

    
    #create one long list of nodes to find duplicates
    nodes_coords = links['start_node'].append(links['end_node'])
    
    #turn series into data frame
    nodes = pd.DataFrame({f'{network_name}_coords':nodes_coords})
    
    #find number of intersecting links
    nodes[f'{network_name}_num_links'] = 1 #this column will be used to count the number of links
    
    #for each unique ID and coordinate, count number of links
    nodes = nodes.groupby([f'{network_name}_coords'], as_index=False).count()
    
    #give nodes row number name
    nodes[f'{network_name}_ID'] = np.arange(nodes.shape[0]).astype(str)
    nodes[f'{network_name}_ID'] = network_mapper{network_name} + network_mapper{link_type} + nodes[f'{network_name}_ID'] + f'_{network_name}'
    
    #extract only the ID and coords column for joining
    nodes = nodes[[f'{network_name}_ID',f'{network_name}_coords']]
    
    #perform join back to original dataframe
    #rename id to be A, rename coords to match start_node
    joining = nodes.rename(columns={f'{network_name}_ID':f'{network_name}_A',f'{network_name}_coords':'start_node'})
    links = pd.merge(links, joining, how = 'left', on = 'start_node' )

    #rename id to be B, rename coords to match end_node
    joining = nodes.rename(columns={f'{network_name}_ID':f'{network_name}_B',f'{network_name}_coords':'end_node'})
    links = pd.merge(links, joining, how = 'left', on = 'end_node' )
    links = links.rename(columns={f'{network_name}_ID':f'{network_name}_B'})
    
    links = links.drop(columns=['start_node','end_node'])
    
    return links

#%% rename node ID's function
def rename_nodes(df, network_name, link_type, A, B):
    df = df.rename(columns={A:f'{network_name}_A',B:f'{network_name}_B'})
    df[f'{network_name}_A'] = network_mapper[network_name] + network_mapper[link_type] + df[f'{network_name}_A'].astype(str)
    df[f'{network_name}_B'] = network_mapper[network_name] + network_mapper[link_type] + df[f'{network_name}_B'].astype(str)
    df[f'{network_name}_A_B'] = df[f'{network_name}_A'] + '_' + df[f'{network_name}_B']
    return df


#%% Create node layer from lines
#do this from the links to make sure that all nodes are included even if they would have been clipped
#by the study area  
def make_nodes(df, network_name, link_type, studyarea_name): 

    #add start and end coordinates to each line
    #this is so we can match them  
    df['start_node'] = df.apply(start_node, geom= df.geometry.name, axis=1)
    df['end_node'] = df.apply(end_node, geom= df.geometry.name, axis=1)

    nodes_id = df[f'{network_name}_A'].append(df[f'{network_name}_B'])
    nodes_coords = df['start_node'].append(df['end_node'])
    
    nodes = pd.DataFrame({f"{network_name}_ID":nodes_id,f"{network_name}_coords":nodes_coords})
    
    #find number of intersecting links
    nodes[f'{network_name}_num_links'] = 1 #this column will be used to count the number of links
    
    #turn nodes from int to str
    #nodes['{network_name}_ID'] = nodes['{network_name}_ID'].astype(str)
    
    #for each unique ID and coordinate, count number of links
    nodes = nodes.groupby([f'{network_name}_ID',f'{network_name}_coords'], as_index=False).count()
    
    #turn the coordinates into points so we can do spatial mapping
    nodes[f'{network_name}_coords'] = nodes.apply(lambda row: Point([row[f'{network_name}_coords']]), axis=1)
    
    #convert to GeoDataFrame and set geo and CRS
    nodes = gpd.GeoDataFrame(nodes).set_geometry(f'{network_name}_coords')
    
    #export nodes as shapefile
    nodes.to_file(rf"Processed_Shapefiles/{network_name}/{network_name}_{studyarea_name}_{link_type}_nodes.geojson", driver = 'GeoJSON')
    
    return nodes

#%% Ignore Fields for Network Conflation Step

def ignore_fields(links, network_name):
    
    #create plain links
    new_links = links.filter([f'{network_name}_A', f'{network_name}_B',f'{network_name}_A_B', 'geometry'])
    
    #create filter fields
    fields = links.columns[~links.columns.isin(new_links.columns)]
    fields = list(links[fields].columns)
    
    with open(f'Ignore_Fields/{network_name}_ignore.pkl', 'wb') as f:
        pickle.dump(fields, f)

#%% Data Cleaning for Future Imports
#removed for now because we want to retain all data fields
# def remove_fields(network,network_name):

#     if('osm'):
#         fields = network.columns[network.isna().all()].tolist() #find all columns with all null values
#         with open(f'Ignore_Fields/{network_name}_ignore_empty.pkl', 'wb') as f:
#             pickle.dump(fields, f)
#     else:
#         print(f'No cleaning parameters saved for {network_name}')
   


#move these to a different file?


#%% Network Data Dictionary/Summary Base

def summurize_network(links, network_name):
    #how many links
    num_links = len(links)
    
    #total mileage of links
    length_mi = links.geometry.length / 5280 # create a new distance column and calculate mileage of each link
    sum_miles = round(length_mi.sum(),0)
    
    #how many attributes
    num_attr = len(links.columns)
    
    #Print Stats
    print(f'There are {num_links} links, {sum_miles} miles of links, and {num_attr} attributes in {network_name}')

def create_attribute_definitions(networkfp, network_name):
    
    #only import the attributes, but leave geometry data out
    gdf = gpd.read_file(networkfp, ignore_geometry = True, bbox=(0,0,0,0))
    
    #data types
    data_types = gdf.dtypes
    
    #number of null values
    #null_values = links.notnull()
    
    sumdata = pd.DataFrame({'data_types': data_types})
    
    #attribute names, number of null values for each attribute, and data type to CSV for data dictionary
    sumdata.to_csv(rf"Processed_Shapefiles/{network_name}/{network_name}.csv")
    
    del[gdf, sumdata]
    

