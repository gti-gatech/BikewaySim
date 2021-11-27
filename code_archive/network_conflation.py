# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 10:36:03 2021

@author: tpassmore6
"""

#%%import cell
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import time
pd.options.display.max_columns = None  # display all columns
pd.options.display.max_rows = None  # display all columns
from shapely import wkt
from shapely.wkt import dumps
from shapely.ops import transform
from shapely.ops import split, snap
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, mapping
import shapely
import pyproj
from itertools import compress
import pickle
from node_ids import start_node, end_node
from collections import Counter
import math
from pathlib import Path

#make directory/pathing more intuitive later
user_directory = os.fspath(Path.home()) #get home directory and convert to path string
file_directory = r"\Documents\GitHub\BikewaySim_Network_Processing" #directory of bikewaysim network processing code

#change this to where you stored this folder
os.chdir(user_directory+file_directory)

#%% Network mapper
#this is how network node ID's will be identified and coded
#the first letter in the network ID represents its origin network
#the second letter in the network ID represent its link type
#all numbers after that are the original network ID 
network_mapper = {
    "abm": "1",
    "here": "2",
    "osm": "3",
    "original": "0",
    "generated": "1",
    "split": "2"
}

#create neccessary folders if they don't exist

#%% ignore fields function
#done in previous script, so not needed here
def ignore_fields(network_name):
    with open(rf'Ignore_Fields/{network_name}_ignore.pkl', 'rb') as f:
        fields_to_ignore = pickle.load(f)
    return fields_to_ignore
    
#%%import links function

def import_links_and_nodes(network_name, studyarea_name, link_type):
    
    #file paths for the links and nodes, should line up with files produced in network processing
    linksfp = rf'Processed_Shapefiles/{network_name}/{network_name}_{studyarea_name}_{link_type}.geojson' 
    nodesfp = rf'Processed_Shapefiles/{network_name}/{network_name}_{studyarea_name}_{link_type}_nodes.geojson'
    
    #only import the necessary fields to reduce memory burden
    fields_to_ignore = ignore_fields(network_name)

    #for some reason ignore fields has stopped working
    links = gpd.read_file(linksfp, driver='GeoJSON').set_crs(epsg=2240) #import network links and project
    #columns
    columns = pd.Series(links.columns)
    links = links[ columns[columns.isin(fields_to_ignore) == False]]
    
    nodes = gpd.read_file(nodesfp).set_crs(epsg=2240) #import nodes and project
    
    #geometry name doesn't seeem to be preserved, so had to rename the geometry column
    nodes = nodes.rename(columns={'geometry':f'{network_name}_coords'}).set_geometry(f'{network_name}_coords')
    
    return links, nodes

#%% Method for finding nearest point
#https://gis.stackexchange.com/questions/222315/geopandas-find-nearest-point-in-other-dataframe

from scipy.spatial import cKDTree

def ckdnearest(gdA, gdB):  
    
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
    
    return gdf

#%%Function for joining join nodes to base nodes, first iteration

def match_intersections(base_nodes, base_name, joining_nodes, joining_name, tolerance_ft, remove_duplicates = True, export_error_lines = False):
    
    #do this step outside of the function
# =============================================================================
#     #Filter joining network
#     #This should make it so that the only joining nodes that the base nodes can join to represent real intersections
#     joining_nodes = joining_nodes[joining_nodes[f'{joining_name}_num_links'] != 2 ].reset_index(drop=True)
# =============================================================================

    #note: if trying to match here to osm or vice versa it would probably be best to use this filter for both

    #from each base node, find the nearest join node
    closest_nodes = ckdnearest(base_nodes, joining_nodes)

    #filter out matched nodes where the match is greater than specified amount aka tolerence, 26ft seemed good
    matched_nodes = closest_nodes[closest_nodes['dist'] <= tolerance_ft]
    
    print(f'{len(matched_nodes)} initial matches')
    
    #if there are one to many matches, then remove_duplicates == True will only keep the match with the smallest match distance
    #set to false if you want to deal with these one to many joins manually
    if remove_duplicates == True:    
        
        #find duplicate matches
        duplicate_matches = matched_nodes[matched_nodes[f'{joining_name}_ID'].duplicated(keep=False)]
        num_with_two_links = len(duplicate_matches[ duplicate_matches[f'{base_name}_num_links'] == 2])
        print(f'{num_with_two_links} duplicate matches only had two connecting links.')
        
        #if two or df_1 nodes match to the same df_2 nodes, then only match the one with the smaller distance
        duplicate_matches_removed = matched_nodes.groupby([f'{joining_name}_ID'], sort=False)['dist'].min()
    
        #used df_2 id to join back to matched nodes
        matched_nodes = pd.merge(matched_nodes, duplicate_matches_removed, how = 'inner', 
                                 on=[f'{joining_name}_ID','dist'], suffixes=(None,'_dup'))
        
        print(f'There were {len(duplicate_matches)} duplicates, now there are {len(matched_nodes)} matches.')
        
    else:
        #check uniqueness this makes sure that it was a one-to-one match
        QAcheck = matched_nodes
        QAcheck['unique'] = 1
        
        #use groupby to check if there are multiple matches to the same joining node
        QAcheck = QAcheck.groupby([f'{joining_name}_ID'], as_index=False).count()
        QAcheck = QAcheck[QAcheck['unique']>1]
        print(f'There are {len(QAcheck)} duplicate matches.')

    if export_error_lines == True:
        #error lines
        #make a line between matched node
        error_lines = matched_nodes.copy()
        error_lines['error_lines'] = error_lines.apply(
            lambda row: LineString([row[f'{base_name}_coords'], row[f'{joining_name}_coords']]), axis=1)
        error_lines = error_lines.set_geometry('error_lines').set_crs(epsg=2240)
        error_lines = error_lines.filter([f'{base_name}_ID', f'{joining_name}_ID', 'error_lines'], axis = 1)
        error_lines.to_file(
            rf'Processed_Shapefiles/matched_nodes/{base_name}_matched_to_{joining_name}_{tolerance_ft}_errorlines.geojson', driver = 'GeoJSON')
        
        #Look at how closely the matched nodes to see if results are reasonable
        print(error_lines.length.describe())
    
    #drop line and here geo and make ABM geo active
    matched_nodes_export = matched_nodes
    matched_nodes_export = matched_nodes_export.filter([f'{base_name}_ID', f'{base_name}_coords', f'{joining_name}_ID'], axis = 1)
    matched_nodes_export = matched_nodes_export.set_geometry(f'{base_name}_coords')
    
    #do the joining process outside of function
# =============================================================================
#     #bikewaysim nodes
#     joining = matched_nodes_export.filter([f'{base_name}_ID',f'{joining_name}_ID']) #only look at the ID columns when joining
#     
#     #join the matched nodes to the base nodes to create the first iteration of the bikewaysim nodes
#     bikewaysim_nodes_v1 = gpd.GeoDataFrame(
#         pd.merge(base_nodes, joining, on=f'{base_name}_ID', how='left'), geometry=f'{base_name}_coords') 
#     
#     #drop the num_links column
#     bikewaysim_nodes_v1 = bikewaysim_nodes_v1.drop(columns={f'{base_name}_num_links'})
#     
#     #export
#     # matched_nodes_export.to_file(
#     #     rf'Processed_Shapefiles/matched_nodes/{base_name}_matched_to_{joining_name}_{tolerance_ft}ft.geojson', driver = 'GeoJSON')
#     bikewaysim_nodes_v1.to_file(
#         rf'Processed_Shapefiles/bikewaysim/bikewaysim_nodes_v1.geojson', driver = 'GeoJSON')
# =============================================================================
    
    
    return matched_nodes_export, len(duplicate_matches), len(matched_nodes_export) #, bikewaysim_nodes_v1

#%% Remaining Nodes Function

def remaining_intersections(matched_nodes, base_nodes, base_network_name, joining_nodes, joining_network_name):
    
    #unmatched base nodes
    unmatched_base_nodes = base_nodes[-base_nodes[f'{base_network_name}_ID'].isin(matched_nodes[f'{base_network_name}_ID'])]#.reset_index(drop = True)
    
    #remove non-intersection base nodes from the remaining nodes
    #unmatched_base_nodes_wo_two_links = len(unmatched_base_nodes[unmatched_base_nodes[f'{base_network_name}_num_links'] != 2])
    
    #unmatched joining nodes
    unmatched_joining_nodes = joining_nodes[-joining_nodes[f'{joining_network_name}_ID'].isin(matched_nodes[f'{joining_network_name}_ID'])]#.reset_index(drop = True)
    #remove non-intersection  nodes from matches
    #unmatched_joining_nodes = unmatched_joining_nodes[unmatched_joining_nodes[f'{joining_network_name}_num_links'] != 2]
    
    print(f'There are {len(unmatched_base_nodes)} {base_network_name} nodes and {len(unmatched_joining_nodes)} {joining_network_name} nodes remaining')
    
    #unmatched_base_nodes.to_file(rf'Processed_Shapefiles/matched_nodes/unmatched_{base_network_name}_with_{joining_network_name}.geojson', driver = 'GeoJSON')
    #unmatched_joining_nodes.to_file(rf'Processed_Shapefiles/matched_nodes/unmatched_{joining_network_name}.geojson', driver = 'GeoJSON')
    
    return unmatched_base_nodes, unmatched_joining_nodes


#%% select unmatched here points which lie on the abm links and find its interpolated point that lie specifically on abm (for precision split)
def point_on_line(unmatched_nodes, unmatched_node_name, base_links, base_name, tolerance_ft):
      
# =============================================================================
#     unmatched_nodes = unmatched_osm_nodes
#     unmatched_node_name = 'osm'
#     base_links = here_road
#     base_name = 'here'
#     tolerance_ft = 20
# =============================================================================


    df_point_on_line_p = pd.DataFrame() # dataframe for storing the corresponding interpolated point information 
    df_point_on_line_l = pd.DataFrame() # dataframe for storing the correspoding base link information
    

    # loop through every unmatched point, as long as the point lies on one link of the whole newtork, it would be identified as lying on the base network
    for index, row in unmatched_nodes.iterrows():
        # check if row in unmatched_nodes distance to all linestrings in base_links
        on_bool_list = base_links["geometry"].distance(row[f"{unmatched_node_name}_coords"]) < tolerance_ft 
        if any(on_bool_list) == True: # if this row matches to base_links feature within the tolerance
            line_idx = list(compress(range(len(on_bool_list)), on_bool_list)) # find the corresponding line
            target_line = base_links.loc[line_idx[0],"geometry"]
            interpolated_point = target_line.interpolate(target_line.project(row[f"{unmatched_node_name}_coords"])) # find the interpolated point on the line
            unmatched_nodes.at[index, f"{base_name}_lie_on"] = "Y"
            df_point_on_line_p.at[index, f"{base_name}_ip_point"] = str(Point(interpolated_point)).strip() 
            df_point_on_line_l.at[index, f"{base_name}_ip_line"] = str(LineString(target_line)).strip()
            df_point_on_line_p.at[index, f"{unmatched_node_name}_cor_ID"] = row[f"{unmatched_node_name}_ID"]
            df_point_on_line_p.at[index, f"{base_name}_A_B"] = base_links.loc[line_idx[0], f"{base_name}_A_B"]
            df_point_on_line_l.at[index, f"{unmatched_node_name}_cor_ID"] = row[f"{unmatched_node_name}_ID"]
            df_point_on_line_l.at[index, f"{base_name}_A_B"] = base_links.loc[line_idx[0], f"{base_name}_A_B"]
        else:
            unmatched_nodes.at[index, f"{base_name}_lie_on"] = "N"
            
    df_point_on = unmatched_nodes[unmatched_nodes[f"{base_name}_lie_on"] == "Y"].reset_index(drop = True)
    df_point_remaining = unmatched_nodes[unmatched_nodes[f"{base_name}_lie_on"] == "N"].reset_index(drop = True)
    df_point_on_line_p = df_point_on_line_p.reset_index(drop = True)
    df_point_on_line_p[f"{base_name}_ip_point"] = df_point_on_line_p[f"{base_name}_ip_point"].apply(wkt.loads) # transform from df to gdf
    gdf_point_on_line_p = gpd.GeoDataFrame(df_point_on_line_p, geometry = f"{base_name}_ip_point").set_crs(epsg=2240)
    df_point_on_line_l = df_point_on_line_l.reset_index(drop = True)
    df_point_on_line_l[f"{base_name}_ip_line"] = df_point_on_line_l[f"{base_name}_ip_line"].apply(wkt.loads)
    gdf_point_on_line_l = gpd.GeoDataFrame(df_point_on_line_l, geometry = f"{base_name}_ip_line").set_crs(epsg=2240) # transform from df to gdf
       
    print(f"There are {len(df_point_on.index)} found to lie on {base_name} network")
    df_point_on.to_file(rf'Processed_Shapefiles/conflation_redux/unmatched_{unmatched_node_name}_lie_on_{base_name}.geojson', driver = 'GeoJSON')
    df_point_remaining.to_file(rf'Processed_Shapefiles/conflation_redux/{unmatched_node_name}_nodes_remaining.geojson', driver = 'GeoJSON')
    print(f'There are {len(df_point_remaining)} osm nodes remaining')
    gdf_point_on_line_p.to_file(rf'Processed_Shapefiles/matched_nodes/unmatched_{unmatched_node_name}_lie_on_{base_name}_ip_point.geojson', driver = 'GeoJSON')
    gdf_point_on_line_l.to_file(rf'Processed_Shapefiles/matched_nodes/unmatched_{unmatched_node_name}_lie_on_{base_name}_ip_line.geojson', driver = 'GeoJSON')
    return df_point_on, gdf_point_on_line_p,gdf_point_on_line_l, df_point_remaining

#%% add node to existing links
# idea behind:
## step 1:return the multistring as a string first (dataframe), since multistring does not split into 
## individual linestring segment, but just add element to list of linestrings

## step 2: expand list of linestring column into several rows, return a dataframe with more rows 

## step 3: turn the dataframe into a geodataframe

def get_linesegments(point, line):  # function to split line into MultiLineString (ATTENTION: not into individual segments, but to MultiLineString)
     return line.difference(point.buffer(1e-6)) #IMPORTANT: add a buffer here make sure it works

def split_by_node_to_multilinestring(gdf_line, line_geom_name, gdf_point, point_geom_name, network_name):
    ab_list = gdf_point[f"{network_name}_A_B"].unique().tolist()
    gdf_line = gdf_line.drop_duplicates(subset = [f"{network_name}_A_B"]) # multiple points could line on the same link, drop duplicates first
    df_split = pd.DataFrame(columns = {f"{network_name}_A_B","geometry"}) # dataframe for storing splitted multistring
    df_split[f"{network_name}_A_B"] = ab_list
    for idx, row in df_split.iterrows():
        ab = row[f"{network_name}_A_B"]
        df_ab = gdf_point[gdf_point[f"{network_name}_A_B"] == ab]
        ab_point = MultiPoint([x for x in df_ab[point_geom_name]])
        #print(ab_point)
        ab_line = gdf_line[gdf_line[f"{network_name}_A_B"] == ab][line_geom_name].values[0]
        #print(df_ab_line)
        split_line = get_linesegments(ab_point, ab_line) # split_line is a geo multilinestring type
        # ATTENTION: format the decimal places to make every row the same, this is important for successfully turning string to geopandas geometry
        split_line = dumps(split_line) # use dump to always get 16 decimal digits irregardles of the total number of digits, dump() change it to MultiLineString to string type
        #print(split_line) # spline_line is MultiLineString type, split_line.wkt is string element
        df_split.at[idx, "geometry"] = split_line
    return df_split
    
def transform_multilinestring_to_segment(df_split, network_link, network_name):
    # ATTENTION: make sure it's a multilinestring element, need to make sure the consistence of data type
    # if the added node coincidentally lie on the end, it still turns a LineString element not a MultiLineString element
    df_split = df_split[df_split["geometry"].str.contains("MULTI")] 
    df_split["geometry"] = df_split["geometry"].str.replace("MULTILINESTRING ","").str.slice(1,-1)
    # explode the list of list columns into separate individual rows
    df_split_line = pd.concat([pd.Series(row[f"{network_name}_A_B"], row['geometry'].split('), (')) for _, row in df_split.iterrows()]).reset_index()
    df_split_line.rename(columns = {"index":"geometry", 0:f"{network_name}_A_B"}, inplace = True)
    # modify each linestring point list into a LineString-like string
    df_split_line["geometry"] = df_split_line["geometry"].str.replace(r"(","").str.replace(")","")
    df_split_line["geometry"] = "LineString (" + df_split_line["geometry"] + ")" 
    gdf_split_line = gpd.GeoDataFrame(columns = {f"{network_name}_A_B","geometry"}).set_crs(epsg=2240) # construct the geo dataframe
    for i in range(len(df_split_line.index)):
        ab = df_split_line.loc[i, f"{network_name}_A_B"]
        geom = df_split_line.loc[i,"geometry"]
        g = shapely.wkt.loads(geom) # the function turns string element into geopandas geometry element
        #print(g)
        gdf_split_line.loc[i,f"{network_name}_A_B"] = ab
        gdf_split_line.loc[i,"geometry"] = g
    
    gdf_split_line.to_file(rf'Processed_Shapefiles/conflation_redux/splitted_{network_name}_link.geojson', driver = 'GeoJSON')
    network_link = network_link[[f"{network_name}_A_B","geometry"]]
    rest_network_line = network_link[~network_link[f"{network_name}_A_B"].isin(df_split[f"{network_name}_A_B"].unique().tolist())]
    rest_network_line.to_file(rf'Processed_Shapefiles/conflation_redux/unsplitted_{network_name}_link.geojson', driver = 'GeoJSON')
    return gdf_split_line, rest_network_line


#%% Create id for split nodes

def new_ids(split_nodes, matched_nodes, base_name, joining_name):
      
    #need to change this naming method later
    #number each match and add 'split' to indicate that it's a split node
    split_nodes[f'{base_name}_ID'] = np.arange(1,split_nodes.shape[0]+1).astype(str)
    split_nodes[f'{base_name}_ID'] = network_mapper[base_name] + network_mapper['split'] + split_nodes[f'{base_name}_ID']

    #rename the columns to match up with the columns of the node gdf for network1
    split_nodes = split_nodes.rename(columns={f'{base_name}_ip_point':f'{base_name}_coords',
                                                                  f'{joining_name}_cor_ID':f'{joining_name}_ID'})
    
    #add in the new nodes
    bikewaysim_nodes_v2 = matched_nodes.append(split_nodes)
    
    #filter any excess columns
    bikewaysim_nodes_v2 = bikewaysim_nodes_v2.filter({f'{base_name}_ID',f'{base_name}_coords',f'{joining_name}_ID'})
    
    #prints how many new nodes were added
    print(f'{len(bikewaysim_nodes_v2)-len(matched_nodes)} more nodes added to {base_name}')
    
    #exports nodes
    bikewaysim_nodes_v2.to_file(rf'Processed_Shapefiles/bikewaysim/bikewaysim_nodes_v2.geojson', driver = 'GeoJSON')
    
    return bikewaysim_nodes_v2

#%% add with split link geo to orignal abm network

def add_split_links_into_network(base_network, split_links, new_nodes, base_name, joining_name):
    
    #merge the base network and new split links, keep all links in base network
    #this is not intended to be a one-to-one merge. It will be a many-to-one merge
    new_network = pd.merge(base_network, split_links, left_on=(f'{base_name}_A_B'), right_on=(f'{base_name}_A_B'), suffixes=(None,'_new'), how='left')
    
    #will have more links because they were added
    print(f'{len(new_network)-len(base_network)} more links added to {base_name}')

    #create new geometry column and add old geometry if new geometry is null
    replace_if_empty = lambda row: row['geometry'] if row['geometry_new'] == None else row['geometry_new']

    #create new column and make that the primary geometry
    new_network['geometry'] = new_network.apply(replace_if_empty, axis = 1)
    new_network = new_network.set_geometry('geometry')

    #remove other two geometry columns
    new_network = new_network.filter([f'{base_name}_A',f'{base_name}_B','geometry'])

    #becauses the start and end coords are slightly off from each other, the typical way of adding new nodes ID associations won't work
    #extract start and end coordinates to each line  
    new_network['start_node'] = new_network.apply(start_node, geom= new_network.geometry.name, axis=1)
    new_network['end_node'] = new_network.apply(end_node, geom= new_network.geometry.name, axis=1)
    
    #filter to just points and rename to same column as nodes
    df_A = new_network.filter(['start_node']).rename(columns={'start_node':f'{base_name}_tuple_coords_new'})
    df_B = new_network.filter(['end_node']).rename(columns={'end_node':f'{base_name}_tuple_coords_new'})
    
    #append end nodes to start nodes to generate list of nodes, can't get rid of duplicates yet
    nodes_coords = df_A.append(df_B)
    
    #turn into points
    nodes_coords[f'{base_name}_coords_new'] = nodes_coords[f'{base_name}_tuple_coords_new'].apply(lambda row: Point(row))
    
    #create dataframe
    nodes = gpd.GeoDataFrame(nodes_coords, geometry = f'{base_name}_coords_new')
    
    #match the nodes using the nearest_neighbor function
    closest_nodes = ckdnearest(nodes, new_nodes)
    
    #filter out matched nodes where the match is greater than specified amount aka tolerence
    matched_nodes = closest_nodes[closest_nodes['dist'] <= 0.00001].drop(columns={'dist',f'{base_name}_coords_new',f'{base_name}_coords'})
    
    #finally drop duplicate nodes
    matched_nodes = matched_nodes.drop_duplicates()
    
    #match these back to the links to get links
    linksA = pd.merge(new_network, matched_nodes, how='left', left_on = 'start_node', right_on = f'{base_name}_tuple_coords_new')
    
    #rename everything so next match can be performed
    linksA = linksA.rename(columns={f'{base_name}_ID':f'{base_name}split_A',f'{joining_name}_ID':f'{joining_name}_A'}).drop(
        columns={f'{base_name}_tuple_coords_new'})
    
    #repeat for B node
    links_fin = pd.merge(linksA, matched_nodes, how='left', left_on = 'end_node', right_on = f'{base_name}_tuple_coords_new')
    
    links_fin = links_fin.rename(columns={f'{base_name}_ID':f'{base_name}split_B',f'{joining_name}_ID':f'{joining_name}_B'}).drop(
        columns={f'{base_name}_tuple_coords_new'})
    
    #desired order of columns
    col_list = [f'{base_name}split_A',f'{base_name}split_B',f'{base_name}_A',f'{base_name}_B',f'{joining_name}_A',f'{joining_name}_B','geometry']
    links_fin = links_fin[col_list]
    
    #export to geojson for examination
    links_fin.to_file(rf'Processed_Shapefiles/bikewaysim/bikewaysim_links_v1.geojson', driver = 'GeoJSON')
    
    return links_fin

#%% Adding in rest of here links and creating nodes file

#implement this new method later
# =============================================================================
# def new_overlapping_links(bikewaysim_links_df, bikewaysim_nodes_df, here_nodes):
#     bikewaysim_links_df = bikewaysim_links_v1.rename(columns={'geometry_new':'geometry'})
#     bikewaysim_links_df[f'{base_network}_A_B'] = bikewaysim_links_df[f'{base_network}_A'] + bikewaysim_links_df[f'{base_name}_B']
#     bikewaysim_nodes_df = bikewaysim_nodes_v2
#     here_nodes = here_road_nodes
#     
#     #find remaining here nodes
#     remaining_here = pd.merge(bikewaysim_nodes_df, here_nodes, on = f'{joining_name}_ID', how="outer", indicator=True)
#     remaining_here = remaining_here[remaining_here['_merge'] == 'right_only']
#     
#     #split abm links but use a finer tolerance
#     # find unmatched here nodes that lie on abm, find correspoding interpolated abm node, find corresponding abm link        
#     unmatched_nav_on_empty, bikewaysim_split, to_be_split_links = point_on_line(remaining_here, "here", bikewaysim_links_df, "abm", tolerance_ft = 26)
# 
#     print(f'{len(corre_abm_point)} split points found.')
# 
#     bikewaysim_split_multi_lines = split_by_node_to_multilinestring(to_be_split_links, "abm_ip_line", bikewaysim_split, "abm_ip_point", "abm")
#     bikewaysim_split_lines, bikewaysim_rest_of_lines = transform_multilinestring_to_segment(bikewaysim_split_multi_lines, bikewaysim_links_df, "abm")
#     print(f'{len(abm_split_lines)} abm split lines were added.')
#     
#     #testing
#     bikewaysim_split_lines.to_file(r'Processed_Shapefiles/testing/split_lines.geojson', driver = 'GeoJSON')
#     
#     bikewaysim_nodes_v3testing = new_ids(bikewaysim_split, bikewaysim_nodes_v2, "abm", "here")
#     bikewaysim_links_v2testing = add_split_links_into_network(bikewaysim_links_df, bikewaysim_split_lines, bikewaysim_nodes_v3testing, "abm", "here")
# 
# =============================================================================

def overlapping_links(bikewaysim_links_df, joining_links_df, base_name, joining_name):

    
    #change name of A_joining andn B_joining
    bikewaysim_links_df = bikewaysim_links_df.rename(columns={f"{joining_name}_A":f"{joining_name}_A_match",f"{joining_name}_B":f"{joining_name}_B_match"})
    
    #buffer the bikewaysim links
    bikewaysim_links_df['buffer_geo'] = bikewaysim_links_df.buffer(30)
    bikewaysim_links_df = bikewaysim_links_df.set_geometry('buffer_geo')
    
    #export buffer for examination
    bikewaysim_links_df.drop(columns={'geometry'}).to_file(r'Processed_Shapefiles/overlapping_links/buffer_bikewaysim.geojson', driver = 'GeoJSON')
    
    #calculate initial length
    joining_links_df['original_length'] = joining_links_df.length
    
    #perform overlay with joining links (here)
    overlapping_links_df = gpd.overlay(joining_links_df, bikewaysim_links_df, how='intersection')
    
    #overlap length
    overlapping_links_df['overlap_length'] = overlapping_links_df.length 
    
    #find percentage overlap and filter
    overlapping_links_df['percent_overlap'] = overlapping_links_df.overlap_length / overlapping_links_df['original_length']
    
    #examine overlap
    #print(overlapping_links_df.head())
    #overlapping_links_df.drop(columns={'geometry_new'}).to_file(r'Processed_Shapefiles/overlapping_links/intersected.geojson', driver = 'GeoJSON')
    
    #filter to only get joining links with high percentage of overlap with bikewaysim links
    overlapping_links_df_filter = overlapping_links_df[overlapping_links_df['percent_overlap'] >= 0.99].drop(columns={
        'geometry','percent_overlap','overlap_length','original_length'}) #needs to be more general

    overlapping_links_df.head()
    
    #need an A_B here for all the overalpping abm links so that we can add attributes
    #groupby A_B_abm 
    idx = overlapping_links_df.groupby([f'{base_name}split_A',f'{base_name}split_B'])['overlap_length'].transform(max) == overlapping_links_df['overlap_length']
    
    #use index to select joining links with the max amount of overlap
    overlapping_links_df = overlapping_links_df[idx]

    #filter columns
    overlapping_links_df = overlapping_links_df.filter([f'{base_name}split_A',f'{base_name}split_B','{base_name}_A',f'{joining_name}_B'])

    #join back to bikewaysim_links   
    bikewaysim_links_df = pd.merge(bikewaysim_links_df, overlapping_links_df, on=[f'{base_name}split_A',f'{base_name}split_B'] )
    
    #check shapefile
    #overlapping_links_df.to_file(r'Processed_Shapefiles/overlapping_links/overlapping_links.geojson', driver = 'GeoJSON')

    #could have second method
    #where we split abm links again
    #or maybe we could just try splitting the first time
    
    return overlapping_links_df_filter, bikewaysim_links_df
    
    

def add_in_other_links(bikewaysim_links_df, joining_links_df, bikewaysim_nodes_df, base_name, joining_name):
    
    
    #find overlapping links
    overlapping_links_df_filter, bikewaysim_links_df = overlapping_links(bikewaysim_links_df, joining_links_df, base_name, joining_name)

    #filter out any overlapping links from joining links
    joining_links_df_filt = joining_links_df[-joining_links_df[f'{joining_name}_A_B'].isin(overlapping_links_df_filter[f'{joining_name}_A_B'])]
    

    #rename geometry column so it can be appended
    #joining_links_df_filt = joining_links_df_filt.rename(columns={'geometry':'geometry_new'})

    #make sure node associations are there
    joining_links_df_filt_A = pd.merge(joining_links_df_filt, bikewaysim_nodes_df, how='left', left_on=f'{joining_name}_A', right_on=f'{joining_name}_ID')
    joining_links_df_filt_A = joining_links_df_filt_A.drop(columns={f'{joining_name}_ID',f'{base_name}_coords'}).rename(columns={f'{base_name}_ID':f'{base_name}split_A'})
    joining_links_df_filt_B = pd.merge(joining_links_df_filt_A, bikewaysim_nodes_df, how='left', left_on=f'{joining_name}_B', right_on=f'{joining_name}_ID')
    joining_links_df_filt_B = joining_links_df_filt_B.drop(columns={f'{joining_name}_ID',f'{base_name}_coords'}).rename(columns={f'{base_name}_ID':f'{base_name}split_B'})

    #append to the links_fin
    links_fin_v2 = bikewaysim_links_df.append(joining_links_df_filt_B)
    
    #filter out columns that aren't neccessary
    links_fin_v2 = links_fin_v2.drop(columns={f'{joining_name}_A_B',f'{joining_name}_A_B','original_length'})
    
    
    ##append {joining_name} at end so we know which network it came from
    links_fin_v2[f'{joining_name}_A_add'] = links_fin_v2[f'{joining_name}_A']
    links_fin_v2[f'{joining_name}_B_add'] = links_fin_v2[f'{joining_name}_B']
    
    #if A_split present, go with that, if not then use {joining_name} ID and append nav at end
    create_active_column_a = lambda row: row[f'{joining_name}_A_add'] if row[f'{base_name}split_A'] is np.nan else row[f'{base_name}split_A']
    create_active_column_b = lambda row: row[f'{joining_name}_B_add'] if row[f'{base_name}split_B'] is np.nan else row[f'{base_name}split_B']
    
    links_fin_v2['bikewaysim_A'] = links_fin_v2.apply(create_active_column_a, axis = 1)
    links_fin_v2['bikewaysim_B'] = links_fin_v2.apply(create_active_column_b, axis = 1)
    
    links_fin_v2 = links_fin_v2.drop(columns={f'{joining_name}_A_add',f'{joining_name}_B_add','buffer_geo'})#.set_geometry('geometry_new')
    
    print(links_fin_v2.head())

    #export
    #links_fin_v2.to_file(r'Processed_Shapefiles/bikewaysim/bikewaysim_links_v2.geojson', driver = 'GeoJSON')
    
    #print how many links were added
    print(f'There were {len(links_fin_v2)-len(bikewaysim_links_df)} links added.')
    
    return links_fin_v2

#%% Node Capture

#this function buffers the joining network and intersects with the conflated networks nodes that possess joining network IDs.
# If two nodes match to a link, then another intersect is done to find which links to add attributes to

def attribute_transfer(joining_network, joining_name, conflated_network_nodes, conflated_network_links):
    
    #need bikewaysim columns first
    
    
    #joining_network = osm_road
    #joining_name = 'osm'
    #conflated_network_nodes = new_nodes
    #conflated_network_links = bikewaysim_links_v1
    
    #search matched nodes for link pairs
    
    
    
    #record original length
    joining_network['original_length'] = joining_network.length

    #buffer and set geo to buffer
    joining_network['buffer'] = joining_network.buffer(10)
    joining_network = joining_network.add_suffix('_attr').set_geometry('buffer_attr')

    #for now just use simple buffer
    intersection = gpd.overlay(conflated_network_links, joining_network, how='intersection')

    #filter by matching length
    intersection['percent_overlap_attr'] = intersection['original_length_attr'] / intersection.length

    intersection = intersection[intersection['percent_overlap_attr'] > 0.99]
    
    #filter to only the joining_network_A_attr and joining_network_B_attr
    #stopped here with this method

    #merge data with original conflated network
    transferred_attr = pd.merge(conflated_network_links, intersection, on=['here_A','here_B'], how = 'left')

    # #intersect with conflated network nodes with buffer
    # intersected_nodes = gpd.overlay(conflated_network_nodes, joining_network, how='intersection')

    # #check result
    # intersected_nodes.head()

    # #see if intersect ID matches one of the A/B refrence ID's
    # id_matches_one = (intersected_nodes[f'{joining_name}_ID'] == intersected_nodes[f'{joining_name}_A']) | (intersected_nodes[f'{joining_name}_ID'] == intersected_nodes[f'{joining_name}_B'])

    # #for intersects with the same link, check to see if both ID's are represented
    # for idx, row in intersected_nodes.iterrows():




    # #intersected_nodes.grouby(by=[f'{joining_network_name}_A',f'{joining_network_name}_B'])

    # test = intersected_nodes

    return transferred_attr




#%% Create new nodes layer

def node_create_bikewaysim(bikewaysim_links_v2, joining_nodes, bikewaysim_nodes_v2, base_name, joining_name):
    
    #bikewaysim_links_v2
    #joining_nodes = here_road_nodes
    #bikewaysim_nodes_v2
    #base_name = 'abm'
    #joining_name = 'here'
    
    #filter bikewaysim links A and B to ID
    node_associations_A = bikewaysim_links_v2.filter({f'{base_name}split_A',f'{joining_name}_A','bikewaysim_A'})
    node_associations_A = node_associations_A.rename(columns=
                                                     {f'{base_name}split_A':f'{base_name}split_ID',
                                                      f'{joining_name}_A':f'{joining_name}_ID',
                                                      'bikewaysim_A':'bikewaysim_ID'})
    
    node_associations_B = bikewaysim_links_v2.filter({f'{base_name}split_B',f'{joining_name}_B','bikewaysim_B'})
    node_associations_B = node_associations_B.rename(columns=
                                                     {f'{base_name}split_B':f'{base_name}split_ID',
                                                      f'{joining_name}_B':f'{joining_name}_ID',
                                                      'bikewaysim_B':'bikewaysim_ID'})
    
    #all the current nodes and associations thus far
    node_associations = node_associations_A.append(node_associations_B)

    #get rid of duplicate ones
    node_associations = node_associations.drop_duplicates()

    #get base and joining node geometries
    node_associations = pd.merge(
        node_associations, bikewaysim_nodes_v2.drop(columns={f'{joining_name}_ID'}), left_on='bikewaysim_ID', right_on=f'{base_name}_ID', how='left')
    
 
    node_associations = pd.merge(
        node_associations, joining_nodes, left_on='bikewaysim_ID', right_on=f'{joining_name}_ID', how='left', suffixes=(None,'_drop')).drop(columns={f'{joining_name}_ID_drop'})

    #create new geometry column
    node_associations['bikewaysim_coords'] = node_associations.apply(
        lambda row: row[f'{joining_name}_coords'] if row[f'{base_name}_coords'] == None else row[f'{base_name}_coords'], axis = 1)
    node_associations = node_associations.drop(columns={f'{base_name}_coords',f'{joining_name}_coords'})
    
    bikewaysim_nodes = gpd.GeoDataFrame(node_associations,geometry='bikewaysim_coords')
    
    bikewaysim_nodes.to_file(rf'Processed_Shapefiles/bikewaysim/bikewaysim_nodes_v3.geojson', driver = 'GeoJSON')
    
    return bikewaysim_nodes


#%%Function for joining join nodes to base nodes, first iteration

def add_bike_nodes(base_nodes, base_name, bike_nodes, bike_name, tolerance_ft, remove_duplicates = True, export_error_lines = False):

# =============================================================================
#     base_nodes = bikewaysim_nodes_v3
#     base_name = 'bikewaysim'
#     bike_nodes = osm_bike_nodes
#     bike_name = 'osm'
#     tolerance_ft = 35
#     remove_duplicates = True
#     export_error_lines = False
# 
# =============================================================================
    #remove excess info
    base_nodes_filt = base_nodes[[f'{base_name}_ID',f'{bike_name}_ID', f'{base_name}_coords']]

    #add suffix to bike nodes so we know what columns are from bike nodes
    bike_nodes = bike_nodes[[f'{bike_name}_ID',f'{bike_name}_coords']].add_suffix('_bike').set_geometry(f'{bike_name}_coords_bike')

    contains_bike_id = True
    if contains_bike_id == True:
        #match bike nodes with osm_id first
        bike_match = pd.merge(base_nodes_filt, bike_nodes, left_on=f'{bike_name}_ID', right_on=f'{bike_name}_ID_bike')

        #remove these from contention
        base_nodes_filt = base_nodes_filt[-base_nodes_filt[f'{bike_name}_ID'].isin(list(bike_match[f'{bike_name}_ID']))]
        bike_nodes = bike_nodes[-bike_nodes[f'{bike_name}_ID_bike'].isin(list(bike_match[f'{bike_name}_ID']))]
        
        #filter
        bike_match = bike_match[[f'{base_name}_ID', f'{bike_name}_ID_bike']]

    #from each base node, find the nearest bike node that has not already been matched
    closest_nodes = ckdnearest(base_nodes_filt, bike_nodes)

    #filter out matched nodes where the match is greater than specified amount aka tolerence, 26ft seemed good
    matched_nodes = closest_nodes[closest_nodes['dist'] <= tolerance_ft]
    
    #if there are one to many matches, then remove_duplicates == True will only keep the match with the smallest match distance
    #set to false if you want to deal with these one to many joins manually
    if remove_duplicates == True:    
        #if two or df_1 nodes match to the same df_2 nodes, then only match the one with the smaller distance
        duplicate_matches_removed = matched_nodes.groupby([f'{bike_name}_ID_bike'], sort=False)['dist'].min()
    
        #used df_2 id to join back to matched nodes
        matched_nodes = pd.merge(matched_nodes, duplicate_matches_removed, how = 'inner', 
                                 on=[f'{bike_name}_ID_bike','dist'], suffixes=(None,'_dup'))

        

# create seperate functions for these to make function cleaner
# =============================================================================
#     else:
#         #check uniqueness this makes sure that it was a one-to-one match
#         QAcheck = matched_nodes
#         QAcheck['unique'] = 1
#         
#         #use groupby to check if there are multiple matches to the same bike node
#         QAcheck = QAcheck.groupby([f'{bike_name}_ID'], as_index=False).count()
#         QAcheck = QAcheck[QAcheck['unique']>1]
#         print(f'There are {len(QAcheck)} duplicate matches.')
# 
#     if export_error_lines == True:
#         #error lines
#         #make a line between matched node
#         error_lines = matched_nodes.copy()
#         error_lines['error_lines'] = error_lines.apply(
#             lambda row: LineString([row[f'{base_name}_coords'], row[f'{bike_name}_coords']]), axis=1)
#         error_lines = error_lines.set_geometry('error_lines').set_crs(epsg=2240)
#         error_lines = error_lines.filter([f'{base_name}_ID', f'{bike_name}_ID', 'error_lines'], axis = 1)
#         error_lines.to_file(
#             rf'Processed_Shapefiles/bikewaysim/bikewaysim_bike_connections.geojson', driver = 'GeoJSON')
#         
#         #Look at how closely the matched nodes to see if results are reasonable
#         print(error_lines.length.describe())
# =============================================================================
    


    #check to see if matched node already has an OSM ID
    if contains_bike_id == True:        
        #remove any matches where OSM ID isn't NA
        matched_nodes = matched_nodes[matched_nodes[f'{bike_name}_ID'].isna()]

        #combine bike match and matched_nodes
        matched_nodes = matched_nodes.append(bike_match)

    #filter matched nodes
    matched_nodes_export = matched_nodes[[f'{base_name}_ID',f'{bike_name}_ID_bike']]
    
    #merge matches back into base nodes
    bikewaysim_nodes_v4 = gpd.GeoDataFrame(
        pd.merge(base_nodes, matched_nodes_export, on=f'{base_name}_ID', how='left'), geometry=f'{base_name}_coords') 
    
    if contains_bike_id == True:
        #add osm id bike to osm id 
        #merge osm id columns
        bikewaysim_nodes_v4[f'{bike_name}_ID'] = bikewaysim_nodes_v4.apply(
            lambda row: row[f'{bike_name}_ID_bike'] if row[f'{bike_name}_ID'] == None else row[f'{bike_name}_ID'], axis = 1)
    else:
        #just rename osmid bike back to osm id
        bikewaysim_nodes_v4 = bikewaysim_nodes_v4.rename(columns={f'{bike_name}_ID_bike':f'{bike_name}_ID'})
    
    #remove nodes that have alraedy been matched
    rest_of_bike_nodes = bike_nodes[-bike_nodes[f'{bike_name}_ID_bike'].isin(bikewaysim_nodes_v4[f'{bike_name}_ID'].dropna())]
    
    #append rest of nodes
    bikewaysim_nodes_v4 = bikewaysim_nodes_v4.append(rest_of_bike_nodes)
    

    #repeat merge osm id columns
    bikewaysim_nodes_v4[f'{bike_name}_ID'] = bikewaysim_nodes_v4.apply(
                lambda row: row[f'{bike_name}_ID_bike'] if row[f'{bike_name}_ID'] == None else row[f'{bike_name}_ID'], axis = 1)
    
    #create new geometry column
    bikewaysim_nodes_v4[f'{base_name}_coords'] = bikewaysim_nodes_v4.apply(
        lambda row: row[f'{bike_name}_coords_bike'] if row[f'{base_name}_coords'] == None else row[f'{base_name}_coords'], axis = 1)
    
    #add bike id to id bikewaysim if empty
    bikewaysim_nodes_v4[f'{base_name}_ID'] = bikewaysim_nodes_v4.apply(
        lambda row: row[f'{bike_name}_ID'] if row[f'{base_name}_ID'] == None else row[f'{base_name}_ID'], axis = 1)
    
    
    #if bikewaysim_ID present, go with that, if not then use {joining_name} ID and append add at end
    #create_active_column_a = lambda row: row[f'{bike_name}_ID'] if row[f'bikewaysim_ID'] is np.nan else row[f'bikewaysim_ID']
    
    #bikewaysim_nodes_v4['bikewaysim_ID'] = bikewaysim_nodes_v4.apply(create_active_column_a, axis = 1)
    
    #filter
    bikewaysim_nodes_v4 = bikewaysim_nodes_v4.drop(columns=[f'{bike_name}_ID_bike',f'{bike_name}_coords_bike'])
    
    bikewaysim_nodes_v4.to_file(
        r'Processed_Shapefiles/bikewaysim/bikewaysim_nodes_v4.geojson', driver = 'GeoJSON')
    
    return bikewaysim_nodes_v4

def add_bike_links(base_nodes, base_links, base_name, bike_links, bike_name):

    base_nodes = bikewaysim_nodes_v4
    base_links = bikewaysim_links_v2
    base_name = 'bikewaysim'
    bike_links = osm_bike
    bike_name = 'osm'

    base_nodes = base_nodes.rename(columns={f'{bike_name}_ID':f'{bike_name}_ID_bike'})
        
    #add in matched bike node points
    dfA = pd.merge(
        base_links, base_nodes.filter([f'{base_name}_ID',f'{bike_name}_ID_bike']), left_on = f'{base_name}_A', right_on= f'{base_name}_ID', how = 'left')
    
    dfA = dfA.drop(columns=[f'{base_name}_ID']).rename(columns={f'{bike_name}_ID_bike':f'{bike_name}_A_bike'})
    
    dfB = pd.merge(
        dfA, base_nodes.filter([f'{base_name}_ID',f'{bike_name}_ID_bike']), left_on = f'{base_name}_B', right_on = f'{base_name}_ID', how = 'left')

    #merge osm id columns
    dfB[f'{base_name}_ID'] = dfB.apply(
        lambda row: row[f'{bike_name}_ID'] if row[f'{base_name}_ID'] == None else row[f'{base_name}_ID'], axis = 1)
    bikewaysim_nodes_v4[f'{base_name}_ID'] = bikewaysim_nodes_v4.apply(
        lambda row: row[f'{bike_name}_ID'] if row[f'{base_name}_ID'] == None else row[f'{base_name}_ID'], axis = 1)
    
    
    
    bikewaysim_links_v3 = dfB.drop(columns=[f'{base_name}_ID']).rename(columns={f'{bike_name}_ID_bike':f'{bike_name}_B_bike'})
    
    #add in node associations to bike links
    dfA = pd.merge(bike_links,base_nodes.drop(columns=[f'{base_name}_coords']), left_on=f'{bike_name}_A', right_on=f'{bike_name}_ID').drop(
        columns = {f'{bike_name}_ID'}).rename(columns={f'{base_name}_ID':f'{base_name}_A',
                                                       f'{bike_name}_ID':f'{base_name}_A',
                                                       f'{base_name}split_ID':f'{base_name}split_A',
                                                       f'{base_name}_ID':f'{base_name}_A',})
    
    dfB = pd.merge(dfA,base_nodes.drop(columns=[f'{base_name}_coords']), left_on=f'{bike_name}_B', right_on=f'{bike_name}_ID').drop(
        columns = {f'{bike_name}_ID'}).rename(columns={f'{base_name}_ID':f'{base_name}_B',
                                                       f'{bike_name}_ID':f'{bike_name}_B',
                                                       f'{base_name}split_ID':f'{base_name}split_B',
                                                       f'{base_name}_ID':f'{base_name}_B',})
    
    #append bike links to bikewaysim links
    bike_links = dfB.drop(columns=['osm_A_B']).rename(columns={'geometry':'geometry_new'})
    
    #add in the rest of the links
    bikewaysim_links_v3 = bikewaysim_links_v3.append(bike_links)
    
    #for some reason there are a lot of duplicates, look at later
    bikewaysim_links_v3 = bikewaysim_links_v3.drop_duplicates()
    
    #if bikewaysim_A present, go with that, if not then use {bike_name} ID and append add at end
    create_active_column_a = lambda row: row[f'{bike_name}_A'] if row[f'bikewaysim_A'] is np.nan else row[f'bikewaysim_A']
    create_active_column_b = lambda row: row[f'{bike_name}_B'] if row[f'bikewaysim_B'] is np.nan else row[f'bikewaysim_B']
    
    bikewaysim_links_v3['bikewaysim_A'] = bikewaysim_links_v3.apply(create_active_column_a, axis = 1)
    bikewaysim_links_v3['bikewaysim_B'] = bikewaysim_links_v3.apply(create_active_column_b, axis = 1)
    
    #export bikewaysim links
    bikewaysim_links_v3.to_file(rf'Processed_Shapefiles/bikewaysim/bikewaysim_links_v3.geojson', driver = 'GeoJSON')    
    
    return bikewaysim_links_v3















































#%% plotting

#world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
#africa = world[world.continent == 'Africa']
#cities = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))

#ax = africa.plot()
#cities.plot(ax=ax, color='r')

#ax = bikewaysim_nodes_v4.set_crs(epsg=2240).to_crs(epsg=4326).plot(markersize = 2)
#minx, miny, maxx, maxy = [-84.383722,33.776262,-84.366884,33.783068]
#ax.set_xlim(minx, maxx)
#ax.set_ylim(miny, maxy)











#%% matching abm to osm

#matched_abm_osm = match_intersections(abm_road_nodes, 'abm', osm_road_nodes, 'osm', 100)

#these are the unmatched nodes
#unmatched_abm_nodes, unmatched_osm_nodes  = remaining_intersections(matched_abm_osm, abm_road_nodes, 'abm', osm_road_nodes, 'osm')

#%% splitting abm links by osm nodes
    
#find unmatched here nodes lie on abm, find correspoding interpolated abm node, find corresponding abm link        
#unmatched_nav_on, corre_abm_point, corre_abm_link = point_on_line(unmatched_osm_nodes, "osm", abm_road, "abm", tolerance_ft = 25)

#df_abm_split = split_by_node_to_multilinestring(corre_abm_link, "ip_abm_line", corre_abm_point, "ip_abm_point", "abm")
#gdf_abm_split, gdf_abm_nosplit = transform_multilinestring_to_segment(df_abm_split, abm_road, "abm")






#%% Future Work

#can simplify the multi to single parts
#simplify

#df_abm_split = split_by_node_to_multilinestring(corre_abm_link, "ip_abm_line", corre_abm_point, "ip_abm_point", "abm")
#gdf_abm_split, gdf_abm_nosplit = transform_multilinestring_to_segment(df_abm_split, abm_road, "abm")

#test = df_abm_split.geometry.apply(shapely.wkt.loads).explode()

#test = pd.merge(test, df_abm_split['A_B_abm'], how='left', left_index=True, right_index=True)
#test = gpd.GeoDataFrame(test, geometry = 'geometry')

#test['count'] = 1
#count = test.groupby('A_B_abm').sum()
#count = count[count['count'] > 1]

#test = pd.merge(test, count, on='A_B_abm', how='inner')

#test.to_file('Processed_Shapefiles/testing/new_split_lines.geojson', driver = 'GeoJSON')

#for index, row in df_abm_split.iterrows():
    
#test3 = pd.merge(gdf_abm_split,test, on = 'A_B_abm', how='right')
