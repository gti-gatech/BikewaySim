#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 09:22:21 2021

@author: tannerpassmore
"""


import os
from pathlib import Path
import time
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.wkt import dumps
from shapely.ops import linemerge 
from shapely.geometry import Point

#ignore the feather warnings
import warnings; warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')

tot_time_start = time.time()

#make directory/pathing more intuitive later
user_directory = os.fspath(Path.home()) #get home directory and convert to path string
file_directory = r"/Documents/GitHub/BikewaySim_Network_Processing" #directory of bikewaysim network processing code

#change this to where you stored this folder
os.chdir(user_directory+file_directory)


#desired study area
#options studyarea, coa, arc
study_area = 'studyarea'


#%% read geojson and write to feather
tot_time_start = time.time()
osm_links_full = gpd.read_file(rf'Base_Shapefiles/osm/osm_links_{study_area}.geojson').to_crs(epsg=2240)
osm_links = osm_links_full[['id','geometry']]
print(f'Took {round(((time.time() - tot_time_start)/60), 2)} mins to read links GeoJSON')

#osm_links.to_feather(rf'Base_Shapefiles/osm/osm_links_{study_area}.feather')
#del osm_links

#%% read geojson and write to feather
osm_nodesfp = rf'Base_Shapefiles/osm/osm_nodes_{study_area}.geojson'

tot_time_start = time.time()
osm_nodes = gpd.read_file(osm_nodesfp).to_crs(epsg=2240)
osm_nodes = osm_nodes[['id','geometry']]
print(f'Took {round(((time.time() - tot_time_start)/60), 2)} mins to read nodes GeoJSON')
#osm_nodes.to_feather(rf'Base_Shapefiles/osm/osm_nodes_{study_area}.feather')
#del osm_nodes

#%% read feather

# =============================================================================
# tot_time_start = time.time()
# osm_links_full = gpd.read_feather(rf'Base_Shapefiles/osm/osm_links_{study_area}.feather').to_crs(epsg=2240)
# osm_links = osm_links_full[['id','geometry']]
# osm_nodes = gpd.read_feather(rf'Base_Shapefiles/osm/osm_nodes_{study_area}.feather').to_crs(epsg=2240)
# osm_nodes = osm_nodes[['id','geometry']]
# print(f'Took {round(((time.time() - tot_time_start)/60), 2)} mins to read links and nodes feather')
# =============================================================================

#%% convert polygon to linestring and break to links and nodes

def convert_poly_to_line(gdf):
    #convert poly to line
    polygons_to_links = gdf[gdf.geometry.type == "Polygon"].boundary.geometry
    gdf.loc[polygons_to_links.index,'geometry'] = polygons_to_links
    gdf = gdf[gdf.geometry.type == "LineString"]
    return gdf

def split_links(gdf):
    #break to links and nodes
    gdf['dissolve'] = 1
    multipart = gdf.dissolve(by='dissolve')
    singlepart = pd.Series(multipart.iloc[0].geometry).tolist()
    split_links = gpd.GeoDataFrame({'geometry':singlepart}, geometry = 'geometry')#.set_crs(epsg=2240)
    return split_links

#%% rejoin attributes

def rejoin_attributes(gdf, gdf_w_attr):
    
    #now need to join attribute information back in by doing a buffer + intersection with orginal network
    gdf['original_length'] = gdf.length
    
    #create buffer
    gdf_w_attr['buffer'] = gdf_w_attr.buffer(0.001) #make a small buffer around the original layer
    gdf_w_attr = gdf_w_attr.set_geometry('buffer') #make the buffer the primary geometry
    
    #perform intersection
    res_intersection = gpd.overlay(gdf, gdf_w_attr, how='intersection')
    res_intersection['intersected_length'] = res_intersection.length
    res_intersection['percent_overlap'] =  res_intersection['intersected_length'] / res_intersection['original_length']
    
    #filter by only retaining matches that had very high overlap
    res_intersection_filt = res_intersection[res_intersection['percent_overlap'] >= 0.99]
    
    #make sure it's singlepart geometry
    res_intersection_filt = res_intersection_filt.explode().droplevel(level=1)

    return res_intersection_filt


#%% Extract Start and End Points Code from a LineString as points
def start_node_point(row, geom):
   return (Point(row[geom].coords.xy[0][0], row[geom].coords.xy[1][0])) #basically look at x and then y coord, use apply to do this for every row of a dataframe

def end_node_point(row, geom):
   return (Point(row[geom].coords.xy[0][-1], row[geom].coords.xy[1][-1]))

#%% this function will match the broken up OSM links with all the possible OSM nodes available

def add_nodeids_to_links(links,nodes,link_id,node_id):
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
   links_start = links_start.set_geometry('start_node').set_crs(epsg=2240, allow_override = True)

   #intersect start nodes with nodes
   links_start = gpd.overlay(links_start, nodes, how="intersection")

   #merge with links and rename to A
   links = pd.merge(links, links_start[['link_id','node_id']], on = 'link_id', how = 'left').rename(columns={'node_id':'A'})

   #end nodes next
   links_end = links.copy()
   links_end['end_node'] = links_end.apply(end_node_point, geom= links_end.geometry.name, axis=1)
   links_end = links_end.set_geometry('end_node').set_crs(epsg=2240, allow_override = True)
   links_end = gpd.overlay(links_end, nodes, how='intersection')
   links = pd.merge(links, links_end[['link_id','node_id']], on = 'link_id', how = 'left').rename(columns={'node_id':'B'})

   return links


#%% process base links

#match and export orginal osm file
poly_to_line = convert_poly_to_line(osm_links_full) #convert polygons to lines
osm_links_w_nodes_base = add_nodeids_to_links(poly_to_line, osm_nodes, 'id', 'id') #add ref nodes ids
osm_links_w_nodes_base.to_file(r'Base_Shapefiles/osm/osm_cleaning/osm_links_base.geojson', driver = 'GeoJSON') #write this to file


#%% process broken up links

#first convert polygons to lines and split all the features
poly_to_line = convert_poly_to_line(osm_links)
split_links = split_links(poly_to_line)

#add attributes back
split_links = rejoin_attributes(split_links, osm_links)

#find reference nodes
osm_links_ref_nodes = add_nodeids_to_links(split_links, osm_nodes, 'id', 'id')

#these are the nodes that show where osm should be split (both endpoints do match to an OSM node)
matched = osm_links_ref_nodes[ (osm_links_ref_nodes.A.isnull() | osm_links_ref_nodes.B.isnull()) == False]

# =============================================================================
# # Non matched links
# =============================================================================
#these are the links that show where osm shouldn't be split (either endpoint doesn't match to an OSM node)
non_matched = osm_links_ref_nodes[ (osm_links_ref_nodes.A.isnull() | osm_links_ref_nodes.B.isnull())]
non_matched.to_file(r'Base_Shapefiles/osm/osm_cleaning/dont_match.geojson', driver = 'GeoJSON')

#dissolve links that don't need to be added by osmid
non_matched = non_matched.dissolve(by='id')

#multilinestring to linestrings using line merge
non_matched['geometry'] = non_matched.geometry.apply(linemerge)
non_matched = non_matched.set_geometry('geometry')

#reset index to not be id
non_matched = non_matched.reset_index()

#filter to only osmid and geo column
non_matched = non_matched[['id','geometry']]
non_matched.to_file(r'Base_Shapefiles/osm/osm_cleaning/not_matched_fixed.geojson', driver = 'GeoJSON')

#filter to only osmid and geo column
matched = matched[['id','geometry']]

# =============================================================================
# #join matched and unmatched links back together
# =============================================================================

#add split links and non-split links
final_links = matched.append(non_matched)

#rename id column
final_links = final_links.rename(columns={'id':'id_del'})

#add attributes
final_links = rejoin_attributes(final_links, osm_links_full)

#what multilinestrings remain?
mulit_lines = final_links[final_links.geometry.type == 'MultiLineString']

#make reference node a and b col
final_links = add_nodeids_to_links(final_links, osm_nodes, 'id', 'id')


#QA Check
final_links = final_links[ (final_links.A.isnull() | final_links.B.isnull()) == False]

non_matched = final_links[ (final_links.A.isnull() | final_links.B.isnull())]
print(len(non_matched))

#remove uneccessary columns
final_links = final_links.drop(columns={'id_del','original_length','intersected_length','percent_overlap','link_id'})

#make sure final crs is right
final_links = final_links.set_crs(epsg=2240, allow_override=True)

#write final to file
final_links.to_file(r'Base_Shapefiles/osm/osm_cleaning/final_osm_links.geojson', driver = 'GeoJSON')




