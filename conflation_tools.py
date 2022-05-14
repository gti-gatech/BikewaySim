# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 14:56:07 2021

@author: tpassmore6 and ziyi
"""

#%%
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import wkt
from shapely.wkt import dumps
from itertools import compress
from shapely.ops import LineString, Point, MultiPoint


def cleaning_process(links, nodes, name):
    
    #column names for A, B, and/or linkID
    mask = [f'{name}_A_B','geometry']
    
    #use mask to only keep neccessary columns
    links = links[mask]
    
    #rename geometry collumns
    links = links.rename(columns={'geometry':f'{name}_line_geo'}).set_geometry(f'{name}_line_geo')
    nodes = nodes.rename(columns={'geometry':f'{name}_point_geo'}).set_geometry(f'{name}_point_geo')

    return links, nodes



#%% Match Points Function

#base function
#https://gis.stackexchange.com/questions/222315/geopandas-find-nearest-point-in-other-dataframe

from scipy.spatial import cKDTree

#take in two geometry columns and find nearest gdB point from each
#point in gdA. Returns the matching distance too.
#MUST BE PROJECTED COORDINATE SYSTEM
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

def match_nodes(base_nodes, base_name, join_nodes, join_name, tolerance_ft, prev_matched_nodes = None, remove_duplicates = True, export_error_lines = False, export_unmatched = False):
    
    #if there are previous matched nodes, remove them from the base and join nodes
    if prev_matched_nodes != None:
        base_nodes = base_nodes[-base_nodes[f'{base_name}_ID'].isin(prev_matched_nodes[f'{base_name}_ID'])]
        join_nodes = join_nodes[-join_nodes[f'{join_name}_ID'].isin(prev_matched_nodes[f'{join_name}_ID'])]
    
    #from each base node, find the nearest join node
    closest_nodes = ckdnearest(base_nodes, join_nodes)

    #filter out matched nodes where the match is greater than specified amount aka tolerence, 25ft seemed good
    matched_nodes = closest_nodes[closest_nodes['dist'] <= tolerance_ft]
    
    #print out the initial number of matches
    print(f'{len(matched_nodes)} initial matches')
    
    #if there are one to many matches, then remove_duplicates == True will only keep the match with the smallest match distance
    #set to false if you want to deal with these one to many joins manually
    if remove_duplicates == True:    
        
        #find duplicate matches
        duplicate_matches = matched_nodes[matched_nodes[f'{join_name}_ID'].duplicated(keep=False)]
        
        #if two or base nodes match to the same join nodes, then only match the one with the smaller distance
        duplicate_matches_removed = matched_nodes.groupby([f'{join_name}_ID'], sort=False)['dist'].min()
    
        #used df_2 id to join back to matched nodes
        matched_nodes = pd.merge(matched_nodes, duplicate_matches_removed, how = 'inner', 
                                 on=[f'{join_name}_ID','dist'], suffixes=(None,'_dup'))
        
        print(f'There were {len(duplicate_matches)} duplicates, now there are {len(matched_nodes)} matches.')
        
    else:
        #mark which ones have been duplicated
        duplicate_matches = matched_nodes[matched_nodes[f'{join_name}_ID'].duplicated(keep=False)]
        print(f'There are {len(duplicate_matches)} duplicate matches.')

    #if this is set to true, it will export a geojson of lines between the matched nodes
    #can be useful for visualizing the matching process
    if export_error_lines == True:
        #make the lines
        error_lines_geo = matched_nodes.apply(
            lambda row: LineString([row[f'{base_name}_point_geo'], row[f'{join_name}_point_geo']]), axis=1)
        #create geodataframe
        error_lines = gpd.GeoDataFrame({f"{base_name}_ID":matched_nodes[f"{base_name}_ID"],
                                        f"{join_name}_ID":matched_nodes[f"{join_name}_ID"],
                                        "geometry":error_lines_geo}, geometry = "geometry")
        #export it to file
        error_lines.to_file(
            rf'processed_shapefiles/matched_nodes/{base_name}_matched_to_{join_name}_{tolerance_ft}_errorlines.geojson', driver = 'GeoJSON')
    
    #drop join geometry and make sure base geometry active
    matched_nodes = matched_nodes.filter([f'{base_name}_ID', f'{join_name}_ID'], axis = 1)
    
    # find remaining nodes for both networks to manage which ones have been merged
    #unmatched base nodes
    unmatched_base_nodes = base_nodes[-base_nodes[f'{base_name}_ID'].isin(matched_nodes[f'{base_name}_ID'])]
    
    #unmatched join nodes
    unmatched_join_nodes = join_nodes[-join_nodes[f'{join_name}_ID'].isin(matched_nodes[f'{join_name}_ID'])]
    
    #if there was a previous match process, merge the old matching one with the new one
    if prev_matched_nodes != None:
        matched_nodes = prev_matched_nodes.append(matched_nodes)    
    
    if export_unmatched == True:
        unmatched_base_nodes.to_file(rf'processed_shapefiles/conflation/matched_nodes/unmatched_{base_name}_nodes.geojson', driver = 'GeoJSON')
        unmatched_join_nodes.to_file(rf'processed_shapefiles/conflation/matched_nodes/unmatched_{join_name}_nodes.geojson', driver = 'GeoJSON')
    
    print(f'There are {len(unmatched_base_nodes)} {base_name} nodes and {len(unmatched_join_nodes)} {join_name} nodes remaining')
    print(f'{len(matched_nodes)} node pairs have been matched so far.')
    
    
    return matched_nodes, unmatched_base_nodes, unmatched_join_nodes


#%% splitting base links by nearest joining nodes

#this function finds the nearest point on a base link from every join node
#this interpolated point will then be used to split the base link
def point_on_line(unmatched_join_nodes, join_name, base_links, base_name, tolerance_ft):
    split_points = pd.DataFrame() # dataframe for storing the corresponding interpolated point information 
    line_to_split = pd.DataFrame() # dataframe for storing the correspoding base link information
    
    # loop through every unmatched point, as long as the point lies on one link of the whole newtork, it would be identified as lying on the base network
    for index, row in unmatched_join_nodes.iterrows():
        # check if row in unmatched_join_nodes distance to all linestrings in base_links
        on_bool_list = base_links[f"{base_name}_line_geo"].distance(row[f"{join_name}_point_geo"]) < tolerance_ft 
        if any(on_bool_list) == True: # if this row matches to base_links feature within the tolerance
            line_idx = list(compress(range(len(on_bool_list)), on_bool_list)) # find the corresponding line
            target_line = base_links.loc[line_idx[0],f"{base_name}_line_geo"]
            interpolated_point = target_line.interpolate(target_line.project(row[f"{join_name}_point_geo"])) # find the interpolated point on the line
            unmatched_join_nodes.at[index, f"{base_name}_lie_on"] = "Y"
            split_points.at[index, f"{base_name}_split_point_wkt"] = str(Point(interpolated_point)).strip() 
            line_to_split.at[index, f"{base_name}_split_line_wkt"] = str(LineString(target_line)).strip()
            split_points.at[index, f"{join_name}_ID"] = row[f"{join_name}_ID"]
            split_points.at[index, f"{base_name}_A_B"] = base_links.loc[line_idx[0], f"{base_name}_A_B"]
            line_to_split.at[index, f"{join_name}_ID"] = row[f"{join_name}_ID"]
            line_to_split.at[index, f"{base_name}_A_B"] = base_links.loc[line_idx[0], f"{base_name}_A_B"]
        else:
            unmatched_join_nodes.at[index, f"{base_name}_lie_on"] = "N"

    #update the unmatced nodes
    unmatched_join_nodes = unmatched_join_nodes[unmatched_join_nodes[f"{base_name}_lie_on"] == "N"].reset_index(drop = True)
    unmatched_join_nodes.drop(columns=[f"{base_name}_lie_on"], inplace = True)
    
    #convert everything to a geodataframe but keep wkt column
    split_points = split_points.reset_index(drop = True)
    split_points[f"{base_name}_split_point_geo"] = split_points[f"{base_name}_split_point_wkt"].apply(wkt.loads) # transform from df to gdf
    split_points = gpd.GeoDataFrame(split_points,geometry=f"{base_name}_split_point_geo")

    line_to_split = line_to_split.reset_index(drop = True)
    line_to_split[f"{base_name}_split_line_geo"] = line_to_split[f"{base_name}_split_line_wkt"].apply(wkt.loads)
    line_to_split = gpd.GeoDataFrame(line_to_split,geometry=f"{base_name}_split_line_geo")
    
    return split_points, line_to_split, unmatched_join_nodes


# add node to existing links
# idea behind:
## step 1:return the multistring as a string first (dataframe), since multistring does not split into 
## individual linestring segment, but just add element to list of linestrings

## step 2: expand list of linestring column into several rows, return a dataframe with more rows 

## step 3: turn the dataframe into a geodataframe

def get_linesegments(point, line):  # function to split line into MultiLineString (ATTENTION: not into individual segments, but to MultiLineString)
     return line.difference(point.buffer(1e-6)) #IMPORTANT: add a buffer here make sure it works

def split_by_nodes(line_to_split, split_points, base_name):
    ab_list = split_points[f"{base_name}_A_B"].unique().tolist()
    line_to_split = line_to_split.drop_duplicates(subset = [f"{base_name}_A_B"]) # multiple points could line on the same link, drop duplicates first
    df_split = pd.DataFrame(columns = {f"{base_name}_A_B",f"{base_name}_wkt"}) # dataframe for storing splitted multistring
    df_split[f"{base_name}_A_B"] = ab_list
    
    for idx, row in df_split.iterrows():
        ab = row[f"{base_name}_A_B"]
        df_ab = split_points[split_points[f"{base_name}_A_B"] == ab]
        ab_point = MultiPoint([x for x in df_ab[f"{base_name}_split_point_geo"]])
        ab_line = line_to_split[line_to_split[f"{base_name}_A_B"] == ab][f"{base_name}_split_line_geo"].values[0]
        split_line = get_linesegments(ab_point, ab_line) # split_line is a geo multilinestring type
        # ATTENTION: format the decimal places to make every row the same, this is important for successfully turning string to geopandas geometry
        # use dump to always get 16 decimal digits irregardles of the total number of digits, dump() change it to MultiLineString to string type
        split_line = dumps(split_line) 
        df_split.at[idx, f"{base_name}_wkt"] = split_line
    
    df_split[f'{base_name}_line_geo'] = df_split[f"{base_name}_wkt"].apply(wkt.loads)
    df_split = gpd.GeoDataFrame(df_split,geometry=f'{base_name}_line_geo')
    
    #convert from multilinestring to segments
    df_split = df_split.explode().reset_index(drop=True)
    
    return df_split

def split_lines_create_points(unmatched_join_nodes, join_name, base_links, base_name, tolerance_ft, export = False):
    
    #get CRS information
    desired_crs = base_links.crs
    
    #note that these function use WKT to do the splitting rather than shapely geometry
    split_points, line_to_split, unmatched_join_nodes = point_on_line(unmatched_join_nodes, join_name, base_links, base_name, tolerance_ft) #finds the split points
    print(f"There are {len(split_points.index)} {join_name} points matching to {len(line_to_split[f'{base_name}_A_B'].unique())} {base_name} links")
    print(f'There are {len(unmatched_join_nodes)} {join_name} nodes remaining')
    
    #splits the lines by nodes found in previous function
    split_lines = split_by_nodes(line_to_split, split_points, base_name) 
    print(f'There were {len(split_lines)} new lines created.')
    
    #drop the wkt columns and A_B column for points
    split_points.drop(columns=[f'{base_name}_split_point_wkt',f'{base_name}_A_B'], inplace=True)
    split_lines.drop(columns=[f'{base_name}_wkt'], inplace=True)
    
    #project gdfs
    split_points.set_crs(desired_crs, inplace=True)
    split_lines.set_crs(desired_crs, inplace=True)

    if export == True:
        #write these to file
        split_lines.to_file("processed_shapefiles/conflation/line_splitting/split_lines.geojson", driver = "GeoJSON")
        split_points.to_file("processed_shapefiles/conflation/line_splitting/split_points.geojson", driver = "GeoJSON")

    return split_lines, split_points, unmatched_join_nodes


# function to add new nodes/links
def add_new_links_nodes(base_links, base_nodes, new_links, new_nodes, base_name):

    #remove links that were splitted
    mask = -base_links[f'{base_name}_A_B'].isin(new_links[f'{base_name}_A_B'])
    base_links = base_links[mask]
    
    #add new links
    base_links = base_links.append(new_links)
     
    #rename geo col
    new_nodes = new_nodes.rename(columns={f'{base_name}_split_point_geo':f'{base_name}_point_geo'}).set_geometry(f'{base_name}_point_geo')
    
    #add split nodes to nodes with the here match
    base_nodes = base_nodes.append(new_nodes)
    
    return base_links, base_nodes

def add_attributes(base_links, base_name, join_links, join_name, buffer_ft):
     
    #give base_links a temp column so each row has unique identifyer
    base_links['temp_ID'] = np.arange(base_links.shape[0]).astype(str)
    
    #buffer base links by 30 ft (or whatever the projected coord unit is)
    base_links['buffer_geo'] = base_links.buffer(buffer_ft)
    base_links = base_links.set_geometry('buffer_geo')
    
    #export buffer for examination
    base_links.drop(columns={f'{base_name}_line_geo'}).to_file(rf'Processed_Shapefiles/conflation/add_attributes/{base_name}_buffer.geojson', driver = 'GeoJSON')
    
    #calculate initial length of join links
    join_links['original_length'] = join_links.length
    
    #perform overlay with join links
    overlapping_links = gpd.overlay(join_links, base_links, how='intersection')
    
    #overlap length
    overlapping_links['overlap_length'] = overlapping_links.length 
    
    
    #for each base link find join link with greatest length overlap
    overlapping_links = overlapping_links.loc[overlapping_links.groupby('temp_ID')['overlap_length'].idxmax()]
    
    #merge the join_A_B column to base_links by temp ID
    base_links = pd.merge(base_links, overlapping_links[['temp_ID',f'{join_name}_A_B']], on = 'temp_ID', how = 'left')
    
    #clean up base_links
    base_links.drop(columns=['temp_ID','buffer_geo'], inplace = True)
    
    #reset active geo
    base_links = base_links.set_geometry(f'{base_name}_line_geo')

    #export final result
    base_links.to_file(rf'Processed_Shapefiles/conflation/add_attributes/{base_name}_joined.geojson', driver = 'GeoJSON')

    return base_links


def start_node_geo(row, geom):
   return (Point(row[geom].coords.xy[0][0], row[geom].coords.xy[1][0])) 

def end_node_geo(row, geom):
   return (Point(row[geom].coords.xy[0][-1], row[geom].coords.xy[1][-1]))

def add_rest_of_features(base_links,base_nodes,base_name,join_links,join_nodes,join_name):
    
    #find the nodes that are not present
    unadded_nodes = join_nodes[-join_nodes[f'{join_name}_ID'].isin(base_nodes[f'{join_name}_ID'])]
     
    #add them
    base_nodes = base_nodes.append(unadded_nodes)
    
    #find the links that are not present
    unadded_links = join_links[-join_links[f'{join_name}_A_B'].isin(base_links[f'{join_name}_A_B'])]
    
    #add them
    base_links = base_links.append(unadded_links)
    
    #merge the geometry columns into one called bikewaysim
    base_links[f'{base_name}_line_geo'] = base_links.apply(
           lambda row: row[f'{join_name}_line_geo'] if row[f'{base_name}_line_geo'] is None else row[f'{base_name}_line_geo'], axis = 1)
    
    base_nodes[f'{base_name}_point_geo'] = base_nodes.apply(
           lambda row: row[f'{join_name}_point_geo'] if row[f'{base_name}_point_geo'] is None else row[f'{base_name}_point_geo'], axis = 1)
    
    #drop the excess geo column, make sure base is set to active geometry
    base_links = base_links.drop(columns=[f'{join_name}_line_geo','original_length']).set_geometry(f'{base_name}_line_geo')
    base_nodes = base_nodes.drop(columns=[f'{join_name}_point_geo']).set_geometry(f'{base_name}_point_geo')
    
    return base_links, base_nodes
 
def merge_diff_networks(base_links, base_nodes, base_type, join_links, join_nodes, join_type, tolerance_ft):
    
    #notes
    #merging network could have the same nodes
    #don't mess with ref id till end
    #need to consider how many ids there will be
    
    #first find nodes that are already present and don't add them
    
    #get network names for each network
    base_cols = list(base_nodes.columns)
    base_ids = [base_cols for base_cols in base_cols if "_ID" in base_cols]
    
    join_cols = list(join_nodes.columns)
    join_ids = [join_cols for join_cols in join_cols if "_ID" in join_cols]
    
    #get list of common names between networks
    common_ids = [base_ids for base_ids in base_ids if base_ids in join_ids]
    
    #remove join_nodes that are in base_nodes
    initial_nodes = len(join_nodes)
    
    for name in common_ids:
        join_nodes = join_nodes[-join_nodes[name].isin(base_nodes[name])]
    
    final_nodes = len(join_nodes)
    print(f'{initial_nodes - final_nodes} nodes already in the base network')
    
    #add rest of join nodes to base nodes
    base_nodes = base_nodes.append(join_nodes)
    
    #get geo names
    base_line_geo = base_links.geometry.name
    base_point_geo = base_nodes.geometry.name
    join_line_geo = join_links.geometry.name
    join_point_geo = join_nodes.geometry.name
    
    #call the match nodes function to form connections between nodes
    base_nodes_to_match = base_nodes.add_suffix(f'_{base_type}').set_geometry(f'{base_point_geo}_{base_type}')
    join_nodes_to_match = join_nodes.add_suffix(f'_{join_type}').set_geometry(f'{join_point_geo}_{join_type}')
    
    print(base_nodes_to_match.geometry.isnull().any())
    print(join_nodes_to_match.isnull().any())
    
    #this isn't working
    #connections = ckdnearest(base_nodes_to_match,join_nodes_to_match)
    #connections = connections[connections['dist'] <= tolerance_ft]

    #only keep connections if there is not already a connection in respective column
    #for name in common_ids:
       # rem_cond = connections[f'{name}_{base_type}'] == connections[f'{name}_{join_type}']
       # connections = connections[-rem_cond]
     
    #add all the links
    base_links = base_links.append(join_links)
    
    #merge the geometry columns into one
    base_links[base_links.geometry.name] = base_links.apply(
           lambda row: row[join_links.geometry.name] if row[base_links.geometry.name] is None else row[base_links.geometry.name], axis = 1)
    
    base_nodes[base_nodes.geometry.name] = base_nodes.apply(
           lambda row: row[join_nodes.geometry.name] if row[base_nodes.geometry.name] is None else row[base_nodes.geometry.name], axis = 1)
    
    #drop the excess geo column, make sure base is set to active geometry
    base_links = base_links.drop(columns=[join_line_geo]).set_geometry(base_line_geo)
    base_nodes = base_nodes.drop(columns=[join_point_geo]).set_geometry(base_point_geo)

    return base_links, base_nodes#, connections
   
def add_reference_ids(links, nodes):

    #get network names
    cols = list(nodes.columns)
    id_cols = [cols for cols in cols if "_ID" in cols]
    names = [id_cols.split('_')[0] for id_cols in id_cols]
    
    #filter nodes
    id_cols.append(nodes.geometry.name)
    nodes = nodes[id_cols]
    
    #get name of geo column
    links_geo = links.geometry.name
    
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
            
    return links


#%% for testing move this to jupyter notebook after


# #this is what you edit
# base_name = "abm"
# join_name = "here"

# base_links = gpd.read_file(r"C:/Users/tpassmore6/Documents/GitHub/BikewaySimDev/processed_shapefiles/abm/abm_bikewaysim_road_links.geojson")
# base_nodes = gpd.read_file(r"C:/Users/tpassmore6/Documents/GitHub/BikewaySimDev/processed_shapefiles/abm/abm_bikewaysim_road_nodes.geojson")
# join_links = gpd.read_file(r"C:/Users/tpassmore6/Documents/GitHub/BikewaySimDev/processed_shapefiles/here/here_bikewaysim_road_links.geojson")
# join_nodes = gpd.read_file(r"C:/Users/tpassmore6/Documents/GitHub/BikewaySimDev/processed_shapefiles/here/here_bikewaysim_road_nodes.geojson")

# #%% conflation steps

# #get rid of excess columns
# base_links, base_nodes = cleaning_process(base_links,base_nodes,base_name)
# join_links, join_nodes = cleaning_process(join_links,join_nodes,join_name)

# #first match the nodes, can repeat this by adding in previously matched_nodes
# tolerance_ft = 25
# matched_nodes, unmatched_base_nodes, unmatched_join_nodes = match_nodes(base_nodes, base_name, join_nodes, join_name, tolerance_ft, prev_matched_nodes=None)

# #join the matched nodes to the base nodes once done with matching
# matched_nodes_final = pd.merge(base_nodes, matched_nodes, on = f'{base_name}_ID', how = "left")

# #create new node and lines from the base links by splitting lines can repeat after the add_new_links_nodes function
# tolerance_ft = 25
# split_lines, split_nodes, unmatched_join_nodes = split_lines_create_points(unmatched_join_nodes, join_name, base_links, base_name, tolerance_ft, export = False)

# #add new links and nodes to the base links and nodes created from split_lines_create_points function
# new_links, new_nodes = add_new_links_nodes(base_links, matched_nodes_final, split_lines, split_nodes, base_name)

# #match attribute information with greatest overlap from joining links
# new_base_links_w_attr = add_attributes(new_links, base_name, join_links, join_name)


# #add unrepresented features from joining by looking at the attributes added in prevoius step for links and the list of matched nodes
# added_base_links, added_base_nodes = add_rest_of_features(new_base_links_w_attr,new_nodes,base_name,join_links,join_nodes,join_name)

# #merge other conflated networks into this
# #import a bike layer
# bike_links = gpd.read_file(r'C:/Users/tpassmore6/Documents/GitHub/BikewaySimDev/processed_shapefiles/here/here_bikewaysim_bike_links.geojson')
# bike_nodes = gpd.read_file(r'C:/Users/tpassmore6/Documents/GitHub/BikewaySimDev/processed_shapefiles/here/here_bikewaysim_bike_nodes.geojson')
# bike_name = 'here'

# #clean excess columns
# bike_links, bike_nodes = cleaning_process(bike_links,bike_nodes,bike_name)

# #merge diff netwrks
# tolerance_ft = 25
# merged_links, merged_nodes, connections = merge_diff_networks(added_base_links, added_base_nodes, 'road', bike_links, bike_nodes, 'bike', tolerance_ft)

# # match reference IDs based on all the id in the nodes
# refid_base_links = add_reference_ids(merged_links, merged_nodes)

# #export
# refid_base_links
# merged_nodes





