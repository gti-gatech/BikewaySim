
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 12:06:49 2022

@author: tpassmore6
"""
#%%
import os
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import wkt
from shapely.wkt import dumps
from itertools import compress
from shapely.ops import LineString, Point, MultiPoint

import os
from pathlib import Path
import time
import geopandas as gpd
import pickle

user_directory = os.fspath(Path.home()) #get home directory and convert to path string
file_directory = r"/Documents/BikewaySimData" #directory of bikewaysim outputs
os.chdir(user_directory+file_directory)

#%%


#function for importing networks that haven't been conflated
def import_network(fp,network_name,layer):    
    #import links and nodes
    links = gpd.read_file(fp,layer=layer+'_links')
    nodes = gpd.read_file(fp,layer=layer+'_nodes')
    
    #filter links
    links = links[[f'{network_name}_A',f'{network_name}_B',f'{network_name}_A_B','geometry']]
    
    #calculates the number of connecting links for each node for filtering purposes
    num_links = links[f'{network_name}_A'].append(links[f'{network_name}_B']).value_counts()
    num_links.name = 'num_links'
    nodes = pd.merge(nodes,num_links,left_on=f'{network_name}_ID',right_index=True)
 
    #rename geometry collumns
    links = links.rename(columns={'geometry':f'{network_name}_line_geo'}).set_geometry(f'{network_name}_line_geo')
    nodes = nodes.rename(columns={'geometry':f'{network_name}_point_geo'}).set_geometry(f'{network_name}_point_geo')
        
    return links, nodes

#initialize conflation
#function creates columns in base links/nodes for join links/nodes ids to go when matched
def initialize_base(base_links,base_nodes,join_name):
    #links
    base_links[f'{join_name}_A'] = None
    base_links[f'{join_name}_B'] = None
    base_links[f'{join_name}_A_B'] = None
    
    #nodes
    base_nodes[f'{join_name}_ID'] = None
    return base_links, base_nodes

#%% Match Points Function

#base function
#https://gis.stackexchange.com/questions/222315/geopandas-find-nearest-point-in-other-dataframe

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


#%% new match_nodes that just appends matched nodes to base nodes

# take in base and join nodes, and output just base nodes with added join nodes

def match_nodes(base_links, base_nodes, base_name, join_nodes, join_name, tolerance_ft, export = False, remove_duplicates = True):
     
    #check for nodes that have already been matched and remove them
    check_prev_matches = base_nodes[f'{join_name}_ID'].isnull()
    base_matching = base_nodes[check_prev_matches]
    
    check_prev_matches = join_nodes[f'{join_name}_ID'].isin(base_nodes[f'{join_name}_ID'])
    join_matching = join_nodes[-check_prev_matches]
    
    #drop join id from base
    base_matching = base_matching.drop(columns=[f'{join_name}_ID'])
    
    #append new to join node id name
    join_matching.rename(columns={f'{join_name}_ID':f'{join_name}_ID_new'},inplace=True)
    
    #print current number of matches
    if check_prev_matches.sum() > 0:
        print(f'{check_prev_matches.sum()} previous matches detected.')
    
    #from each base node, find the nearest join node
    closest_nodes = ckdnearest(base_matching,join_matching)

    #filter out matched nodes where the match is greater than specified amount aka tolerence, 25ft seemed okay    
    matched_nodes = closest_nodes[closest_nodes['dist'] <= tolerance_ft]

    #if there are one to many matches, then remove_duplicates == True will only keep the match with the smallest match distance
    #set to false if you want to deal with these one to many joins manually
    if remove_duplicates == True:    
        
        #find duplicate matches
        duplicate_matches = matched_nodes[matched_nodes[f'{join_name}_ID_new'].duplicated(keep=False)]
        
        #if two or base nodes match to the same join nodes, then only match the one with the smaller distance
        duplicate_matches_removed = matched_nodes.groupby([f'{join_name}_ID_new'], sort=False)['dist'].min()
    
        #used df_2 id to join back to matched nodes
        matched_nodes = pd.merge(matched_nodes, duplicate_matches_removed, how = 'inner', 
                                 on=[f'{join_name}_ID_new','dist'], suffixes=(None,'_dup'))
        
    else:
        #mark which ones have been duplicated
        duplicate_matches = matched_nodes[matched_nodes[f'{join_name}_ID_new'].duplicated(keep=False)]
        print(f'There are {len(duplicate_matches)} duplicate matches.')

    #if this is set to true, it will export a geojson of lines between the matched nodes
    #can be useful for visualizing the matching process
    if export == True:
        #make the lines
        error_lines_geo = matched_nodes.apply(
            lambda row: LineString([row[f'{base_name}_point_geo'], row[f'{join_name}_point_geo']]), axis=1)
        #create geodataframe
        error_lines = gpd.GeoDataFrame({f"{base_name}_ID":matched_nodes[f"{base_name}_ID"],
                                        f"{join_name}_ID_new":matched_nodes[f"{join_name}_ID_new"],
                                        "geometry":error_lines_geo}, geometry = "geometry")
        #export it to file
        error_lines.to_file(
            rf'processed_shapefiles/conflation/node_matching/{base_name}_to_{join_name}_{tolerance_ft}ft.gpkg', layer='errorlines', driver = 'GPKG')
    
    #filter matched nodes
    matched_nodes = matched_nodes[[f'{base_name}_ID',f'{join_name}_ID_new']]
    
    #resolve with base nodes
    base_nodes = pd.merge(base_nodes, matched_nodes, on=f'{base_name}_ID', how = 'left')
    
    #if there is no existing join node id AND there is a matched node id then replace the None with the node id
    cond = -(base_nodes[f'{join_name}_ID'].isnull() & -(base_nodes[f'{join_name}_ID_new'].isnull()))
    base_nodes[f'{join_name}_ID'] = base_nodes[f'{join_name}_ID'].where(cond,base_nodes[f'{join_name}_ID_new'])
    
    #drop new node id columns
    base_nodes.drop(columns=[f'{join_name}_ID_new'],inplace=True)

    #find unmatched base nodes
    unmatched_base_nodes = base_nodes[base_nodes[f'{join_name}_ID'].isnull()]
    
    #find unmatched join nodes
    unmatched_join_nodes = join_nodes[-join_nodes[f'{join_name}_ID'].isin(base_nodes[f'{join_name}_ID'])]
    
    if export:
        #export
        unmatched_base_nodes.to_file(rf'processed_shapefiles/conflation/node_matching/{base_name}_to_{join_name}_{tolerance_ft}ft.gpkg', layer='unmatched_base_nodes', driver = 'GPKG')
        unmatched_join_nodes.to_file(rf'processed_shapefiles/conflation/node_matching/{base_name}_to_{join_name}_{tolerance_ft}ft.gpkg', layer='unmatched_join_nodes', driver = 'GPKG')

    #get number of matched nodes
    num_matches = (-(base_nodes[f'{join_name}_ID'].isnull())).sum()
    
    print(f'There are {len(unmatched_base_nodes)} {base_name} nodes and {len(unmatched_join_nodes)} {join_name} nodes remaining')
    print(f'{num_matches - check_prev_matches.sum()} new matches')
    print(f'{num_matches} node pairs have been matched so far.')
    
    #add ref ids
    base_links = add_reference_ids_geo(base_links, base_nodes)
    
    return base_links, base_nodes



#%% split lines

#master function
def split_lines_create_points(base_nodes, base_links, base_name, join_nodes, join_name, tolerance_ft, export = False):
    
    #take out join nodes that have already been matched
    check_prev_matches = join_nodes[f'{join_name}_ID'].isin(base_nodes[f'{join_name}_ID'])
    potential_nodes = join_nodes[-check_prev_matches]
    
    #filter join nodes by num links
    # 1 = dead ends
    # 2 = continue, no intersection
    # 3 = three-way intersection
    # 4 = four-way intersection
    # etc...
    potential_nodes = potential_nodes[potential_nodes['num_links'] > 2]
    
    #get CRS information
    desired_crs = base_links.crs
    
    split_points = pd.DataFrame() # dataframe for storing the corresponding interpolated point information 
    lines_to_split = pd.DataFrame() # dataframe for storing the correspoding base link information
    
    #note that these function use WKT to do the splitting rather than shapely geometry
    split_points, lines_to_split = point_on_line(potential_nodes, join_name, base_links, base_name, tolerance_ft) #finds the split points
    print(f"There are {len(split_points)} {join_name} points matching to {len(lines_to_split[f'{base_name}_A_B'].unique())} {base_name} links")
    
    #splits the lines by nodes found in previous function
    split_lines = split_by_nodes(lines_to_split, split_points, base_name) 
    print(f'There were {len(split_lines)} new lines created.')
    
    #project gdfs
    split_points.set_crs(desired_crs, inplace=True)
    split_lines.set_crs(desired_crs, inplace=True)
    
    if export == True:
        split_lines.to_file(f"processed_shapefiles/conflation/link_splitting/split_{tolerance_ft}ft.gpkg", layer='split_lines', driver = "GPKG")
        split_points.to_file(f"processed_shapefiles/conflation/link_splitting/split_{tolerance_ft}ft.gpkg", layer='split_points', driver="GPKG")
    
    #add to base nodes after conforming column names
    split_points.drop(columns=[f'{base_name}_split_point_wkt',f'{base_name}_A_B'], inplace=True)
    split_points.rename(columns={f'{base_name}_split_point_geo':f'{base_name}_point_geo'}, inplace=True)
    base_nodes = base_nodes.append(split_points)
    
    #add to base links after conforming column names
    split_lines.drop(columns=[f'{base_name}_wkt'], inplace=True)
    split_lines.reset_index(drop=True,inplace=True) 
    #remove links that were splitted
    mask = -base_links[f'{base_name}_A_B'].isin(split_lines[f'{base_name}_A_B'])
    base_links = base_links[mask]
    #add new links
    base_links = base_links.append(split_lines)
    
    check=len(base_links)
    #fix reference nodes
    base_links = add_reference_ids_geo(base_links, base_nodes)
    print(check==len(base_links))
    return base_links, base_nodes


def start_node_geo(row, geom):
   return (Point(row[geom].coords.xy[0][0], row[geom].coords.xy[1][0])) 

def end_node_geo(row, geom):
   return (Point(row[geom].coords.xy[0][-1], row[geom].coords.xy[1][-1]))



def add_reference_ids_geo(links, nodes):
    
    #get network names
    cols = list(nodes.columns)
    id_cols = [cols for cols in cols if "_ID" in cols]
    
    #filter nodes
    id_cols.append(nodes.geometry.name)
    nodes = nodes[id_cols]
    
    #get link columns
    cols = list(links.columns)
    ref_cols = [col for col in cols if ("_A" in col or "_B" in col) and not "A_B" in col]
    
    #get name of geo column
    links_geo = links.geometry.name
    
    #match id to starting node
    links['start_point_geo'] = links.apply(start_node_geo, geom= links.geometry.name, axis=1)
        
    #set to active geo
    links = links.set_geometry('start_point_geo')
        
    #find nearest node from starting node
    links = ckdnearest(links,nodes,return_dist=False)
        
    #rename id columns to _A
    links.columns = pd.Series(list(links.columns)).str.replace('_ID','_Anew')

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
    links.columns = pd.Series(list(links.columns)).str.replace('_ID','_Bnew')
 
    #remove end point
    links = links.drop(columns=['end_point_geo',nodes.geometry.name])
 
    #reset geometry   
    links = links.set_geometry(links_geo)

    #replace empty ref col with the new ones (preserve the original ones though)
    for ref_col in ref_cols:
        links.loc[links[ref_col].isnull(),ref_col] = links.loc[links[ref_col].isnull(),ref_col+'new']
    
    #drop new ones
    cols = links.columns
    new_cols = [cols for cols in cols if "new" in cols]
    links.drop(columns=new_cols,inplace=True)
            
    return links


#this function finds the nearest point on a base link from every join node
#this interpolated point will then be used to split the base link
def point_on_line(potential_nodes, join_name, base_links, base_name, tolerance_ft):
    split_points = pd.DataFrame() # dataframe for storing the corresponding interpolated point information 
    lines_to_split = pd.DataFrame() # dataframe for storing the correspoding base link information
    
    # loop through every unmatched point, as long as the point lies on one link of the whole newtork, it would be identified as lying on the base network
    for index, row in potential_nodes.iterrows():
        
        # check if row in unmatched_join_nodes distance to all linestrings in base_links
        candidates = base_links[f"{base_name}_line_geo"].distance(row[f"{join_name}_point_geo"]) 
        
        if any(candidates < tolerance_ft): # if this row matches to base_links feature within the tolerance
            #find candidate line
            candidate = candidates.idxmin()
            
            #get closest line geometry
            target_line = base_links.loc[candidate,f"{base_name}_line_geo"]
            
            #find the interpolated point on the line
            #project finds the distance along the line
            #interpolate takes this distance and returns the point
            interpolated_point = target_line.interpolate(target_line.project(row[f"{join_name}_point_geo"]))
            
            #add split point geo to split_points df but convert point geo to wkt (well-known text)
            split_points.at[index, f"{base_name}_split_point_wkt"] = str(Point(interpolated_point)).strip() 
            
            #add the line geo to the lines_to_split df but convert line geo to wkt
            lines_to_split.at[index, f"{base_name}_split_line_wkt"] = str(LineString(target_line)).strip()
            
            #add the join node id
            split_points.at[index, f"{join_name}_ID"] = row[f"{join_name}_ID"]
            
            #add the link id
            split_points.at[index, f"{base_name}_A_B"] = base_links.loc[candidate, f"{base_name}_A_B"]
            
            #add the join node id
            lines_to_split.at[index, f"{join_name}_ID"] = row[f"{join_name}_ID"]
            
            #add the link id
            lines_to_split.at[index, f"{base_name}_A_B"] = base_links.loc[candidate, f"{base_name}_A_B"]
    
    #convert everything to a geodataframe but keep wkt column (for exporting if needed)
    split_points = split_points.reset_index(drop = True)
    split_points[f"{base_name}_split_point_geo"] = split_points[f"{base_name}_split_point_wkt"].apply(wkt.loads) # transform from df to gdf
    split_points = gpd.GeoDataFrame(split_points,geometry=f"{base_name}_split_point_geo")

    lines_to_split = lines_to_split.reset_index(drop = True)
    lines_to_split[f"{base_name}_split_line_geo"] = lines_to_split[f"{base_name}_split_line_wkt"].apply(wkt.loads)
    lines_to_split = gpd.GeoDataFrame(lines_to_split,geometry=f"{base_name}_split_line_geo")
    
    return split_points, lines_to_split


# add node to existing links
# idea behind:
## step 1:return the multistring as a string first (dataframe), since multistring does not split into 
## individual linestring segment, but just add element to list of linestrings
## step 2: expand list of linestring column into several rows, return a dataframe with more rows 
## step 3: turn the dataframe into a geodataframe
# function to split line into MultiLineString (ATTENTION: not into individual segments, but to MultiLineString)
def get_linesegments(point, line):  
    #IMPORTANT: add a very small buffer here otherwise point won't touch line
    #difference returns the lines that are not within the multipoint buffer
    return line.difference(point.buffer(1e-6))

def split_by_nodes(lines_to_split, split_points, base_name):
    #get all the unique link ids
    ab_list = split_points[f"{base_name}_A_B"].unique().tolist()
    
    #multiple points could lie on the same link, drop duplicates first
    lines_to_split = lines_to_split.drop_duplicates(subset = [f"{base_name}_A_B"])
    
    #create dataframe for storing splitted multistring
    df_split = pd.DataFrame(columns = {f"{base_name}_A_B",f"{base_name}_wkt"}) # dataframe for storing splitted multistring
    
    #add link ids to dataframe
    df_split[f"{base_name}_A_B"] = ab_list
    
    for idx, row in df_split.iterrows():
        #get link id of line to split
        ab = row[f"{base_name}_A_B"]
        
        #pull out corresponding split points with link id
        df_ab = split_points[split_points[f"{base_name}_A_B"] == ab]
        
        #put all split points for the link to be split into a multipoint object
        ab_point = MultiPoint([x for x in df_ab[f"{base_name}_split_point_geo"]])
        
        #get the line geometry for this link id 
        ab_line = lines_to_split[lines_to_split[f"{base_name}_A_B"] == ab][f"{base_name}_split_line_geo"].values[0]
        
        #split the line using the multipoint and linestring
        split_line = get_linesegments(ab_point, ab_line) # split_line is a geo multilinestring type
        
        # ATTENTION: format the decimal places to make every row the same, this is important for successfully turning string to geopandas geometry
        # use dump to always get 16 decimal digits irregardles of the total number of digits, dump() change it to MultiLineString to string type
        split_line = dumps(split_line) 
        
        #add the multi line string to the dataframe
        df_split.at[idx, f"{base_name}_wkt"] = split_line
     
    #turn WKT's into shapely multi line string    
    df_split[f'{base_name}_line_geo'] = df_split[f"{base_name}_wkt"].apply(wkt.loads)
    
    #turn into geodataframe
    df_split = gpd.GeoDataFrame(df_split,geometry=f'{base_name}_line_geo')
    
    #convert from multilinestring to segments
    df_split = df_split.explode()
    
    return df_split



#%% add attributes in for next step

#need to carve out exceptions for no name streets?
def dissolve_by_attr(buffered_links,join_name,study_area):
    #expected filepath pattern for retrieving attributes
    fp = f'processed_shapefiles/{join_name}/{join_name}_{study_area}_network.gpkg'
    
    #bring in attributes
    #may need to change by link type
    if os.path.exists(fp):
        #import link attributes (geometry not necessary)
        attr = gpd.read_file(fp,layer='base_links',ignore_geometry=True)
        
        #for each network, specify which attributes you want to use to check
        if join_name == 'here':
            #use street name, functional classification, speed category, and lane category
            columns = ['ST_NAME','FUNC_CLASS','SPEED_CAT','LANE_CAT']
        elif join_name == 'osm':
            columns = ['osmid']
        else:
            columns = None
        
        #check if there were attributes specified
        if columns is not None:
            #turn attribute comlumns into tuple
            attr['attr_tup'] = [tuple(x) for x in attr[columns].values.tolist()]
            
            #filter attr df
            attr = attr[[f'{join_name}_A_B','attr_tup']+columns]
            
            #merge with join links
            links = pd.merge(buffered_links,attr,on=f'{join_name}_A_B')
            
            #get list of dissolved ids
            #group = links.groupby('attr_tup')
            
            #disolve geo
            dissolved_buffer = links.dissolve('attr_tup')
            
            #reset index
            dissolved_buffer = dissolved_buffer.reset_index()
            
            #filter columns
            dissolved_buffer = dissolved_buffer[[f'{join_name}_A_B','attr_tup','geometry']+columns]
        
        return dissolved_buffer, columns

#%% attribute transfer and add rest of links

def add_attributes(base_links, join_links, join_name, buffer_ft, study_area, export=True):
    
    export_fp = rf'processed_shapefiles/conflation/attribute_transfer/attribute_transfer_{join_name}.gpkg'
    
    #give base_links a temp column so each row has unique identifier (index basically does this too)
    # A_B doesn't work because there might be duplicates from the split links step
    base_links['temp_ID'] = np.arange(base_links.shape[0]).astype(str)
    
    #calculate original base_links link length
    base_links['length'] = base_links.geometry.length
    
    #create copy of join links to use for bufferring
    buffered_links = join_links.copy()
    
    #buffer the join links by tolerance_ft
    buffered_links['geometry'] = buffered_links.buffer(buffer_ft)
    
    #make sure it's the active geometry
    buffered_links.set_geometry('geometry',inplace=True)
    
    #filter join links to just link id and geometry
    buffered_links = buffered_links[[f'{join_name}_A_B','geometry']]
    
    #export buffered links
    buffered_links.to_file(export_fp,layer='buffered_links',driver='GPKG')

    #intersect with just buffered_links no dissolve 
    #just_buffer = gpd.overlay(base_links, buffered_links, how='intersection')
    
    #dissolve the buffers according to attribute data
    dissolved_buffer, columns = dissolve_by_attr(buffered_links,join_name,study_area)
    
    #dissolved buffer export
    dissolved_buffer.drop(columns=['attr_tup']).to_file(export_fp,layer='dissolved_buffer',driver='GPKG')
    
    #intersect join buffer and base links
    overlapping_links = gpd.overlay(base_links, dissolved_buffer, how='intersection')
    
    #re-calculate the link length to see which join buffers had the greatest overlap of base links
    overlapping_links['percent_overlap'] = (overlapping_links.geometry.length / overlapping_links['length'] )
    #just_buffer['percent_overlap'] = (just_buffer.geometry.length / just_buffer['length'])
    
    ##
    #export overlapping to examine
    overlapping_links.drop(columns=['attr_tup']).to_file(export_fp,layer='buffered_overlap',driver='GPKG')
    #just_buffer.to_file(export_fp,layer='just_buffer_overlap',driver='GPKG')
    ##  
    
    #select matches with greatest link overlap
    max_overlap_idx = overlapping_links.groupby('temp_ID')['percent_overlap'].idxmax().to_list()
    
    #get matching links from overlapping links
    match_links = overlapping_links.loc[max_overlap_idx]
    
    # #get list of here links that are overlapping
    # attr_tup = match_links['attr_tup'].drop_duplicates().to_list()
    # list_of_lists = [group.get_group(x)[f'{join_name}_A_B'].to_list() for x in attr_tup]
    # flattened_list = [item for sublist in list_of_lists for item in sublist]
    
    # #join back with intersected data to get percent overlap
    # #prolly shouldn't use flattened list for this?
    # partial_overlap = overlapping_links[overlapping_links[f'{join_name}_A_B_2'].isin(flattened_list)]
    # partial_overlap = partial_overlap[partial_overlap['percent_overlap'] < 0.8]
    # partial_overlap = join_links[-(join_links[f'{join_name}_A_B'].isin(partial_overlap[f'{join_name}_A_B_2']))] 
    
    # #find links that need to be added
    # rem_join_links = join_links[-(join_links[f'{join_name}_A_B'].isin(flattened_list))] 
    
    # #add in the partial overlap links
    # rem_join_links = rem_join_links.append(partial_overlap)
    
    #join with base links
    base_links = pd.merge(base_links,match_links[['temp_ID',f'{join_name}_A_B_2']+columns],on='temp_ID',)

    #resolve join link id
    base_links.loc[base_links[f'{join_name}_A_B'].isnull(),f'{join_name}_A_B'] = base_links.loc[base_links[f'{join_name}_A_B'].isnull(),f'{join_name}_A_B_2']
    
    #drop added columns
    base_links.drop(columns=['temp_ID','length',f'{join_name}_A_B_2'],inplace=True)
    
    #test_export
    base_links.to_file(export_fp, layer='base_links_w_join_attr', driver='GPKG')
    
    #drop the attr columns
    base_links.drop(columns=columns,inplace=True)
    
    return base_links

def add_rem_links(base_nodes,base_links,base_name,join_links,join_nodes,join_name,buffer_ft):
    
    export_fp = r'processed_shapefiles/conflation/conflated_networks/conflated.gpkg'
    
    #give base_links a temp column so each row has unique identifier (index basically does this too)
    # A_B doesn't work because there might be duplicates from the split links step
    #base_links['temp_ID'] = np.arange(base_links.shape[0]).astype(str)
    
    #get count
    #starting_count = len(base_links)
    
    #calculate original base_links link length
    join_links['length'] = join_links.geometry.length
    
    #create copy of join links to use for bufferring
    buffered_links = base_links.copy()
    
    #buffer the join links by tolerance_ft
    buffered_links['geometry'] = buffered_links.buffer(buffer_ft)
    
    #make sure it's the active geometry
    buffered_links.set_geometry('geometry',inplace=True)
    
    #dissolve buffered links and filter to just geometry
    dissolved_links = buffered_links[['geometry']].dissolve()
    
    #intersect join_links and dissolved_links
    overlapping_links = gpd.overlay(join_links, dissolved_links, how='intersection')
    
    #calculate percent overlap
    overlapping_links['percent_overlap'] = (overlapping_links.geometry.length / overlapping_links['length'] )
    
    #filter by overlap percentage
    dup_links = overlapping_links[overlapping_links['percent_overlap'] > 0.95]
    
    #remove dup_links but add the rest
    rem_links = join_links[-join_links[f'{join_name}_A_B'].isin(dup_links[f'{join_name}_A_B'])]
    
    #merge with join links to get links to add
    #rem_links = pd.merge(join_links,rem_links[[f'{join_name}_A_B']],on=f'{join_name}_A_B')
    
    #append to base links
    base_links = base_links.append(rem_links)
    
    #resolve geometry column
    base_links.loc[base_links[f'{base_name}_line_geo'].isnull(),f'{base_name}_line_geo'] = base_links.loc[base_links[f'{base_name}_line_geo'].isnull(),f'{join_name}_line_geo']
    
    #drop length column
    base_links.drop(columns=['length',f'{join_name}_line_geo'],inplace=True)
    
    #resolve base_A_B columns
    #add ref nodes
    base_links = add_reference_ids(base_links, base_nodes, base_name, join_name)
    
    #add rem new nodes to base nodes
    rem_nodes = base_links[f'{join_name}_A'].append(base_links[f'{join_name}_B']).drop_duplicates()
    rem_nodes = join_nodes[join_nodes[f'{join_name}_ID'].isin(rem_nodes)]
    
    #make sure that node doesn't already exist
    rem_nodes = rem_nodes[-rem_nodes[f'{join_name}_ID'].isin(base_nodes[f'{join_name}_ID'])]
    base_nodes = base_nodes.append(rem_nodes)
    
    #resolve node geo
    base_nodes.loc[base_nodes[f'{base_name}_point_geo'].isnull(),f'{base_name}_point_geo'] = base_nodes.loc[base_nodes[f'{base_name}_point_geo'].isnull(),f'{join_name}_point_geo']
    
    #drop
    base_nodes.drop(columns=[f'{join_name}_point_geo','num_links'],inplace=True)
    
    #create new link and node ids
    base_links, base_nodes = conflation_ids(base_name, base_nodes, base_links, join_nodes, join_links, join_name)
    
    #export
    base_links.to_file(export_fp,layer='conflated_'+base_name+join_name+'_links',driver='GPKG')
    base_nodes.to_file(export_fp,layer='conflated_'+base_name+join_name+'_nodes',driver='GPKG')
    
    return base_links, base_nodes

#take in base nodes and use them to add ref nodes to rem links
def add_reference_ids(links, base_nodes, base_name, join_name):
    
    #create dict
    for_dict = base_nodes[-base_nodes[f'{base_name}_ID'].isnull()]
    node_dict = dict(zip(for_dict[f'{join_name}_ID'],for_dict[f'{base_name}_ID']))

    #use map on a column
    links[f'{base_name}_A'] = links[f'{join_name}_A'].map(node_dict)
    links[f'{base_name}_B'] = links[f'{join_name}_B'].map(node_dict)
    
    return links

#%% resolve so it can be exported

def conflation_ids(base_name,base_nodes,base_links,join_nodes,join_links,join_name):
    #creates new ids for conflated links to streamline conflation in the next step
    conflate_name = base_name + join_name
    
    #new link id
    base_links[conflate_name+'_A_B'] = base_links[f'{base_name}_A_B'].astype(str) + ',' + base_links[f'{join_name}_A_B'].astype(str)

    #new a
    base_links[conflate_name+'_A'] = base_links[f'{base_name}_A'].astype(str) + ',' + base_links[f'{join_name}_A'].astype(str)
    
    #new b
    base_links[conflate_name+'_B'] = base_links[f'{base_name}_B'].astype(str) + ',' + base_links[f'{join_name}_B'].astype(str)
    
    #new id
    base_nodes[conflate_name+'_ID'] = base_nodes[f'{base_name}_ID'].astype(str) + ',' + base_nodes[f'{join_name}_ID'].astype(str)

    return base_links, base_nodes


#%%pickle backup

def backup(base_links,base_nodes,join_links,join_nodes):
    backup_dict = {'base_links':base_links,'base_nodes':base_nodes,
                   'join_links':join_links,'join_nodes':join_nodes}
    
    with open('processed_shapefiles/conflation/backup.pkl', 'wb') as fh:
        pickle.dump(backup_dict,fh)
    
def load_backup():
    with open('processed_shapefiles/conflation/backup.pkl', 'rb') as fh:
        backup_dict = pickle.load(fh)
    
    base_links = backup_dict['base_links']
    base_nodes = backup_dict['base_nodes']
    join_links = backup_dict['join_links']
    join_nodes = backup_dict['join_nodes']
    
    return base_links, base_nodes, join_links, join_nodes

#%% combine bike and road links

# #shortcut being used for now
# def combine_bike_road(network_name):
#     #expected fp
#     fp = f'processed_shapefiles/{network_name}/{network_name}_bikewaysim_network.gpkg'

#     #load links
#     road_links = gpd.read_file(fp,layer='road_links')
#     bike_links = gpd.read_file(fp,layer='bike_links')
    
#     #combine
#     links = road_links.append(bike_links)
#     print(links.duplicated().sum())
#     links.drop_duplicates(inplace=True)
    
#     #load nodes
#     road_nodes = gpd.read_file(fp,layer='road_nodes')
#     bike_nodes = gpd.read_file(fp,layer='bike_nodes')
    
#     nodes = road_nodes.append(bike_nodes)
#     print(nodes.duplicated().sum())
#     nodes.drop_duplicates(inplace=True)
    
#     #add to gpkg
#     links.to_file(fp,layer='roadbike_links',driver='GPKG')
#     nodes.to_file(fp,layer='roadbike_nodes',driver='GPKG')
    
#     return

# combine_bike_road('osm')
# combine_bike_road('here')




#%% abm + here

study_area = 'bikewaysim'

base_name = 'abm'
basefp = 'processed_shapefiles/abm/abm_bikewaysim_network.gpkg'

join_name = 'here'
joinfp = 'processed_shapefiles/here/here_bikewaysim_network.gpkg'

base_links, base_nodes = import_network(basefp,base_name,'road')
join_links, join_nodes = import_network(joinfp,join_name,'roadbike')

#initialize the base network
base_links, base_nodes = initialize_base(base_links, base_nodes, join_name)

#%%match nodes step

tolerance_ft = 25
base_links, base_nodes = match_nodes(base_links, base_nodes, base_name, join_nodes,join_name,tolerance_ft)

#tolerance_ft = 30
#base_links, base_nodes = match_nodes(base_links, base_nodes, base_name, join_nodes,join_name,tolerance_ft)

#%%split lines step
tolerance_ft = 25
base_links, base_nodes = split_lines_create_points(base_nodes, base_links, base_name, join_nodes, join_name, tolerance_ft)





#%% attribute transfer


#match attribute information with greatest overlap from joining links
buffer_ft = 30

base_links = add_attributes(base_links, join_links, join_name, buffer_ft, study_area)


base_links,base_nodes = add_rem_links(base_nodes,base_links,base_name,join_links,join_nodes,join_name,buffer_ft)


#%% repeat for osm

study_area = 'bikewaysim'

base_name = 'abmhere'
basefp = 'processed_shapefiles/conflation/conflated_networks/conflated.gpkg'

join_name = 'osm'
joinfp = 'processed_shapefiles/osm/osm_bikewaysim_network.gpkg'


#base_links, base_nodes, _,_ = load_backup()

base_links, base_nodes = import_network(basefp,base_name,'conflated_'+base_name)
join_links, join_nodes = import_network(joinfp,join_name,'roadbike')

#initialize the base network
base_links, base_nodes = initialize_base(base_links, base_nodes, join_name)

#%%match nodes step

tolerance_ft = 25
base_links, base_nodes = match_nodes(base_links, base_nodes, base_name, join_nodes,join_name,tolerance_ft)

tolerance_ft = 30
base_links, base_nodes = match_nodes(base_links, base_nodes, base_name, join_nodes,join_name,tolerance_ft)

#%%split lines step
tolerance_ft = 25
base_links, base_nodes = split_lines_create_points(base_nodes, base_links, base_name, join_nodes, join_name, tolerance_ft,export=True)

#%% attribute transfer
#match attribute information with greatest overlap from joining links
buffer_ft = 30

base_links = add_attributes(base_links, join_links, join_name, buffer_ft, study_area)


base_links,base_nodes = add_rem_links(base_nodes,base_links,base_name,join_links,join_nodes,join_name,buffer_ft)

#back this up
backup(base_links,base_nodes,join_links,join_nodes)






#%% TRB shortcut code

#base_links, base_nodes, join_links, join_nodes = load_backup()


#figure out how to make this a function and not hard coded

#A_B columns are only good for attributes

#split the abmhere_A, B, and A_B cols (wont work if three columns)
# base_links['abmhere_A'].str.split(',',expand=True)

# base_links['abm_A'] = base_links['abmhere_A'].str.split(',',expand=True)[0]
# base_links['here_A'] = base_links['abmhere_A'].str.split(',',expand=True)[1]

# base_links['abm_B'] = base_links['abmhere_B'].str.split(',',expand=True)[0]
# base_links['here_B'] = base_links['abmhere_B'].str.split(',',expand=True)[1]

# base_links['abm_A_B'] = base_links['abmhere_A_B'].str.split(',',expand=True)[0]
# base_links['here_A_B'] = base_links['abmhere_A_B'].str.split(',',expand=True)[1]

# base_links['trb_A'] = None
# base_links['trb_B'] = None
# base_nodes['trb_ID'] = None

# aCond = base_links['trb_A'].isnull()
# bCond = base_links['trb_B'].isnull()
# idCond = base_nodes['trb_ID'].isnull()

# #first is abm
# base_links.loc[-base_links['abm_A'].isnull() & aCond,'trb_A'] = base_links.loc[-base_links['abm_A'].isnull() & aCond,'abm_A']
# base_links.loc[-base_links['abm_B'].isnull() & bCond,'trb_B'] = base_links.loc[-base_links['abm_B'].isnull() & bCond,'abm_B']

# #second is here
# base_links.loc[-base_links['here_A'].isnull() & aCond,'trb_A'] = base_links.loc[-base_links['here_A'].isnull() & aCond,'here_A']
# base_links.loc[-base_links['here_B'].isnull() & bCond,'trb_B'] = base_links.loc[-base_links['here_B'].isnull() & bCond,'here_B']

# #third is osm
# base_links.loc[-base_links['osm_A'].isnull() & aCond,'trb_A'] = base_links.loc[-base_links['osm_A'].isnull() & aCond,'osm_A']
# base_links.loc[-base_links['osm_B'].isnull() & bCond,'trb_B'] = base_links.loc[-base_links['osm_B'].isnull() & bCond,'osm_B']

# #create trb_a_b col
# base_links['trb_A_B'] = base_links['trb_A'] + '_' + base_links['trb_B']


# #first is abm
# base_nodes.loc[-base_nodes['abm_ID'].isnull() & idCond,'trb_ID'] = base_nodes.loc[-base_nodes['abm_ID'].isnull() & idCond,'abm_ID']

# #second is here
# base_nodes.loc[-base_nodes['here_ID'].isnull() & idCond,'trb_ID'] = base_nodes.loc[-base_nodes['here_ID'].isnull() & idCond,'here_ID']

# #third is osm
# base_nodes.loc[-base_nodes['osm_ID'].isnull() & idCond,'trb_ID'] = base_nodes.loc[-base_nodes['osm_ID'].isnull() & idCond,'osm_ID']

# #filter
# base_links=base_links[['trb_A','trb_B','trb_A_B','abm_A_B','here_A_B','osm_A_B','abmhere_line_geo']]
# base_nodes=base_nodes[['trb_ID','abmhere_point_geo']]

#%%

#export for adding in attribute data
base_links.to_file('processed_shapefiles/conflation/finalized_networks/trb.gpkg',layer='links')
base_nodes.to_file('processed_shapefiles/conflation/finalized_networks/trb.gpkg',layer='nodes')

#make sure every ref id has at least one A and B


    


#%%

# new_base_links_w_attr = add_attributes(new_links, base_name, join_links, join_name, buffer_ft)
# new_base_links_w_attr.head()

# #add unrepresented features from joining by looking at the attributes added in previous step for links and the list of matched nodes
# added_base_links, added_base_nodes = add_rest_of_features(new_base_links_w_attr,new_nodes,base_name,join_links,join_nodes,join_name)

# #create new abmhere column with id and geo
# final_links, final_nodes = fin_subnetwork(added_base_links,added_base_nodes,base_name,join_name)

# final_links.to_file(rf'processed_shapefiles/conflation/{base_name+join_name}_links.geojson')
# final_nodes.to_file(rf'processed_shapefiles/conflation/{base_name+join_name}_nodes.geojson')



#base_links.to_file('processed_shapefiles/conflation/testing.gpkg',layer='links',driver='GPKG')
#base_nodes.to_file('processed_shapefiles/conflation/testing.gpkg',layer='nodes',driver='GPKG')

# def retrieve_attributes(links,network_name,study_area):
#     #expected relative fp of link attributes
#     fp = f'processed_shapefiles/{network_name}/{network_name}_{study_area}_base_links_atrr.geojson'
    
#     #bring in attributes
#     if os.path.exists(fp):
#         #import link attributes (geometry not necessary)
#         attr = gpd.read_file(fp,ignore_geometry=True)
        
#         #for each network, specify which attributes you want to use to check
#         if network_name == 'here':
#             #use street name, functional classification, speed category, and lane category
#             columns = ['ST_NAME','FUNC_CLASS','SPEED_CAT','LANE_CAT']
#         else:
#             columns = None
        
#         #check if there were attributes specified
#         if columns is not None:
#             #turn attribute comlumns into tuple
#             #attr['attr_tup'] = [tuple(x) for x in attr[columns].values.tolist()]
            
#             #filter attr to just link id and columns
#             #attr = attr[[f'{network_name}_A_B','attr_tup']]
            
#             attr = attr[[f'{network_name}_A_B']+columns]
            
#             #merge with join links
#             links = pd.merge(links,attr,left_on=f'{network_name}_A_B_2',right_on=f'{network_name}_A_B')
            
#             links.to_file(r'processed_shapefiles/conflation/test.geojson',driver='GeoJSON')
            
            
#             #get the mode attribute for each match
#             attr_mode = links.groupby('temp_ID')['attr_tup'].agg(pd.Series.mode).rename('attr_mode')
            
#             #add back to links
#             links = pd.merge(links,attr_mode,left_on='temp_ID',right_index=True)
            
#             #compare tuple to mode tuple
#             def compare(x):
#                 #if there are more than one modes then
#                 #compare each mode to attr tup
#                 if len(x['attr_mode']) > 1:
#                     checks=[]
#                     for element in x['attr_mode']:
#                         check = x['attr_tup'] == element
#                         checks.append(check)
#                     check = any(checks)
#                 #otherwise just check tup against mode
#                 else:
#                     check = x['attr_tup'] == x['attr_mode']
#                 return check
            
#             #run above function
#             links['attr_match'] = links.apply(compare, axis=1)
            
#             #find most overlap
#             links = max_overlap(links)
    
#             #find the join link that covers most of the base link
#             #and shares the most common attributes with the other join links
#             matching_conditions = (links['attr_match']) & (links['ismax'])
#             matching_links = links[matching_conditions]
            
#             #check if match for each ABM link
#             print(links['temp_ID'].nunique())
#             print(len(matching_links))
    
#             #these links shouldn't be matched, but should be considered already represented
#             overlap_conditions = (links['attr_match'] == True) & (links['percent_overlap'] > 0.50)
#             overlapping_links = links[overlap_conditions]
            

#     else:
#         print(f'No {network_name} network attributes found, using just percent overlap')
#         #match with link with greatest overlap
#         matching_links = links.groupby('temp_ID')['percent_overlap'].idxmax()
        
#         #check if match for each ABM link
#         print(links.nunique('temp_ID') == len(matching_links))
        
#         #these links shouldn't be matched, but should be considered already represented
#         overlap_conditions = links['percent_overlap'] > 0.50
#         overlapping_links = links[overlap_conditions]
     
#     return matching_links, overlapping_links

# def max_overlap(links):
#     #find most overlap
#     links['maxidx'] = links.groupby('temp_ID')['percent_overlap'].transform('idxmax')

#     #reset index to make it a column
#     links.reset_index(inplace=True)
    
#     #and compare them
#     links['ismax'] = links['index']==links['maxidx']
    
#     return links



# def add_attributes(buffered_links,network_name,study_area):
    
#     #expected filepath pattern
#     fp = f'processed_shapefiles/{join_name}/{join_name}_{study_area}_base_links_atrr.geojson'
    
#     #bring in attributes
#     if os.path.exists(fp):
#         #import link attributes (geometry not necessary)
#         attr = gpd.read_file(fp,ignore_geometry=True)
        
#         #for each network, specify which attributes you want to use to check
#         if join_name == 'here':
#             #use street name, functional classification, speed category, and lane category
#             columns = ['ST_NAME','FUNC_CLASS','SPEED_CAT','LANE_CAT']
#         else:
#             columns = None
        
#         #check if there were attributes specified
#         if columns is not None:
#             #turn attribute comlumns into tuple
#             attr['attr_tup'] = [tuple(x) for x in attr[columns].values.tolist()]
            
#             #filter attr df
#             attr = attr[[f'{network_name}_A_B','attr_tup']+columns]
            
#             #merge with join links
#             links = pd.merge(join_links,attr,on=f'{network_name}_A_B')
            
#             #get list of dissolved ids
#             group = links.groupby('attr_tup')
            
#             #disolve geo
#             dissolved_buffer = links.dissolve('attr_tup')
            
#             #reset index
#             dissolved_buffer = dissolved_buffer.reset_index(drop=True)
            
#             #filter columns
#             dissolved_buffer = dissolved_buffer[[f'{network_name}_A_B','geometry']]
    
#             return dissolved_buffer, group
#         else:
#             print('No attributes available')
#             return links
