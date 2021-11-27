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
import numpy as np

#make directory/pathing more intuitive later
user_directory = os.fspath(Path.home()) #get home directory and convert to path string
file_directory = r"/Documents/GitHub/BikewaySimDev" #directory of bikewaysim network processing code

#change this to where you stored this folder
os.chdir(user_directory+file_directory)

# function for importing data downloaded from 'download_from_overpassAPI.py'

def import_osm(study_area):    
    #Links
    tot_time_start = time.time()
    osm_links = gpd.read_file(rf'base_shapefiles/osm/osm_links_{study_area}_raw.geojson').to_crs(epsg=2240)
    osm_links = osm_links.rename(columns={'id':'osm_link_id'})
    osm_links['osm_link_id'] = osm_links['osm_link_id'].astype(str)
    print(f'Took {round(((time.time() - tot_time_start)/60), 2)} mins to read links') 
    print(f'There are {len(osm_links)} links.')
    # Nodes
    tot_time_start = time.time()
    osm_nodes = gpd.read_file(rf'base_shapefiles/osm/osm_nodes_{study_area}_raw.geojson').to_crs(epsg=2240)
    osm_nodes = osm_nodes.rename(columns={'id':'osm_node_id'})
    osm_nodes['osm_node_id'] = osm_nodes['osm_node_id'].astype(str)
    print(f'Took {round(((time.time() - tot_time_start)/60), 2)} mins to read nodes')
    print(f'There are {len(osm_nodes)} nodes.')
    return osm_links, osm_nodes

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
    split_links = gpd.GeoDataFrame({'geometry':singlepart}, geometry = 'geometry')
    
    split_links = rejoin_attributes(split_links, gdf)
    
    split_links = split_links.drop(columns='dissolve')
    
    return split_links

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
    
    #remove added columns
    res_intersection_filt = res_intersection_filt.drop(columns=['percent_overlap','original_length','intersected_length'])

    return res_intersection_filt

# Extract Start and End Points Code from a LineString as points
def start_node_point(row, geom):
   return (Point(row[geom].coords.xy[0][0], row[geom].coords.xy[1][0])) #basically look at x and then y coord, use apply to do this for every row of a dataframe

def end_node_point(row, geom):
   return (Point(row[geom].coords.xy[0][-1], row[geom].coords.xy[1][-1]))

# Method for finding nearest point
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


# this function will match the broken up OSM links with all the possible OSM nodes available
def add_nodeids_to_links(links,nodes):
   #this function takes in a set of links and creates an A and B column in the link gdf
   #using the specified id column in the nodes gdf
   #currently used for OSM   
   #turn link index into column to serve as link id
   links['link_id'] = links.index
   
   #create very small buffer on nodes in nodes gdf and set active geo to buffer
   #nodes['geomtery'] = nodes.buffer(0.1)
   #nodes = nodes.set_geometry('geometry')
   
   links_start = links
   links_end = links
   
   #start nodes
   links_start['start_node'] = links_start.apply(start_node_point, geom=links_start.geometry.name, axis=1)
   links_start = links_start.set_geometry('start_node')
   links_start = links_start[['link_id','start_node']]
   #find nearest
   links_start_nearest = ckdnearest(links_start, nodes)
   #filter
   links_start_nearest = links_start_nearest[links_start_nearest['dist'] < 0.001]
   
   #end nodes
   links_end['end_node'] = links_end.apply(end_node_point, geom= links_end.geometry.name, axis=1)
   links_end = links_end.set_geometry('end_node')
   links_end = links_end[['link_id','end_node']]
   links_end_nearest = ckdnearest(links_end, nodes)
   links_end_nearest = links_end_nearest[links_end_nearest['dist'] < 0.001]

   #merge
   links_a = pd.merge(links, links_start_nearest[['link_id','osm_node_id']], on = 'link_id', how = 'left').rename(columns={'osm_node_id':'A'})
   links_b = pd.merge(links_a, links_end_nearest[['link_id','osm_node_id']], on = 'link_id', how = 'left').rename(columns={'osm_node_id':'B'}) 

   #drop
   links_b = links_b.drop(columns=['start_node','end_node'])
   return links_b

#%% get this to work

#input study area name used in previous step
osm_links, osm_nodes = import_osm('studyarea')


#%%
#filter data
osm_links_filt = osm_links[['osm_link_id','geometry']]
osm_nodes_filt = osm_nodes[['osm_node_id','geometry']]

#poly to line
osm_links_poly_to_line = convert_poly_to_line(osm_links_filt)

#split links
osm_links_split = split_links(osm_links_poly_to_line)


#%%

#match nodes
matched_links = add_nodeids_to_links(osm_links_split, osm_nodes_filt)

only_matched = matched_links[ (matched_links['A'].isnull() | matched_links['B'].isnull()) == False]


#%%fix the non-matched ones

non_matched = matched_links[ (matched_links['A'].isnull() | matched_links['B'].isnull())]
non_matched.plot()

#dissolve links that don't need to be added by osmid
non_matched = non_matched.dissolve(by='osm_link_id')

#multilinestring to linestrings using line merge
non_matched['geometry'] = non_matched.geometry.apply(linemerge)
non_matched = non_matched.set_geometry('geometry')
non_matched = non_matched.reset_index().explode().droplevel(level=1)

#readd osm_ids
non_matched = non_matched.drop(columns=['A','B'])
non_matched_fixed = add_nodeids_to_links(non_matched, osm_nodes_filt)

#%% re-add

#reset indexes
only_matched = only_matched.reset_index(drop = True)
non_matched_fixed = non_matched_fixed.reset_index(drop = True)
final_links = only_matched.append(non_matched_fixed)

#make sure final crs is right
final_links = final_links.set_crs(epsg=2240, allow_override=True)

#add in the attributes
final_links = rejoin_attributes(final_links, osm_links)

#write final to file
final_links.to_file(r'base_shapefiles/osm/final_osm_links.geojson', driver = 'GeoJSON')

