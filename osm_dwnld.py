# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 11:12:14 2021

@author: tpassmore6
"""

import requests
import geopandas as gpd
import pandas as pd
import osmnx as ox

def download_osm(gdf,desired_osm_attributes=[]):
    
    #get geometry info from osmnx
    osmnx_nodes, osmnx_links = download_osmnx(gdf)
    
    #get additonal attribute information from overpass
    overpass_links = overpass_download(gdf)
    
    #retrieve only specific osm attributes like highway or oneway if given list
    if len(desired_osm_attributes) > 0:
        overpass_links = overpass_links[desired_osm_attributes]
    
    #merge osmnx and overpass data on link id
    osm_links = pd.merge(osmnx_links, overpass_links, left_on=('osmid'), right_on=('id'), how = 'inner')
    
    return osmnx_nodes, osm_links

def simple_download(studyarea):
    #convert to unprojected if already projected
    if studyarea.crs != 'EPSG:4326':
        studyarea = studyarea.to_crs('EPSG:4326')
    
    #get bounding box
    minx, miny, maxx, maxy = studyarea.total_bounds
    
    #retrieve graph from bbox
    G = ox.graph_from_bbox(maxy, miny, maxx, minx, network_type='drive')
    
    return G


def download_osmnx(studyarea):

    #convert to unprojected if already projected
    if studyarea.crs != 'EPSG:4326':
        studyarea = studyarea.to_crs('EPSG:4326')
    
    #get bounding box
    minx, miny, maxx, maxy = studyarea.total_bounds
   
    #retrieve graph from bbox
    G = ox.graph_from_bbox(maxy, miny, maxx, minx, network_type='all_private', simplify=False)
    
    #retrieve graph from the chosen study polygon
    #G = ox.graph_from_polygon(studyarea.geometry[0], network_type='all_private', simplify= False)
    
    #simplify graph unless different osm ids
    G = ox.simplification.simplify_graph(G, strict = False)
    
    #remove directed links (links for each direction)
    G = ox.utils_graph.get_undirected(G)
    
    #plot it for fun
    ox.plot_graph(G)

    #convert to gdf
    nodes, links = ox.utils_graph.graph_to_gdfs(G)

    #reset index and name u and v to A and B
    links = links.reset_index().rename(columns={'u':'A','v':'B'})
    
    #simplify columns to geo and id
    links = links[['osmid','A','B','key','geometry']]

    #reset nodes index
    nodes = nodes.reset_index()

    return nodes, links

def overpass_download(studyarea):
        
    #convert to unprojected if already projected
    if studyarea.crs != 'EPSG:4326':
        studyarea = studyarea.to_crs('EPSG:4326')
    
    #get bounds and print to see if reasonable
    minx, miny, maxx, maxy = studyarea.total_bounds
    print(f'The bounding box is {minx}, {miny}, {maxx}, {maxy}')
        
    query = f"""
    [out:json]
    [timeout:120]
    ;
    (
      way
        ["highway"]
        ({miny},{minx},{maxy},{maxx});
    );
    out body;
    >;
    out skel qt;
    """
    
    url = "http://overpass-api.de/api/interpreter"
    r = requests.get(url, params={'data': query})
    
    result = r.json()
    
    #simplify for dataframe
    df = pd.json_normalize(result, record_path = ['elements'])
    
    #clean up column names
    df.columns = df.columns.str.replace(r'tags.', '')
    
    #drop the nodes column because it's a list
    #df.drop(columns={'nodes'}, inplace = True)
    
    #find duplicate column names and drop them
    #might need to investigate duplicate columns in future to see if any data worth migrating over
    # colnames = list(df.columns)
    # colnames_lower = [x.lower() for x in colnames]
    # colnames_df = pd.DataFrame(data = {'name':colnames, 'name_lower':colnames_lower})
    # colnames_df = colnames_df[~colnames_df['name_lower'].duplicated()]['name']
    # df = df[colnames_df]
    
    return df