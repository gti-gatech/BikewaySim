# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 11:12:14 2021

@author: tpassmore6
"""

import requests
import geopandas as gpd
import pandas as pd
import osmnx as ox
import pickle
import numpy as np

def download_osm(studyarea_or_bbox,crs,network_type='bike'):
    '''
    Set to network_type=all_private to download everything, temporarily changing this to speed process up
    '''

    if isinstance(studyarea_or_bbox,tuple):
        print('Bounding box provided')
        #check to see if all values between -180 and 180
        bbox = np.array(studyarea_or_bbox)
    elif isinstance(studyarea_or_bbox,gpd.GeoDataFrame):
        print('GeoDataFrame provided')
        gdf = studyarea_or_bbox.copy()
        #convert to correct CRS
        if gdf.crs != 'epsg:4326':
            gdf.to_crs('epsg:4326',inplace=True)
        #get bounding box
        bbox = gdf.total_bounds
    else:
        print('Neither tuple of coords nor geodataframe was provided')

    #get bounding box
    minx, miny, maxx, maxy = bbox
    print(f'The bounding box is {minx}, {miny}, {maxx}, {maxy}')

    #get geometry info from osmnx
    osmnx_nodes, osmnx_links = download_osmnx(minx,miny,maxx,maxy,network_type)
    
    #project
    osmnx_nodes.to_crs(crs,inplace=True)
    osmnx_links.to_crs(crs,inplace=True)

    #get additonal attribute information from overpass
    overpass_links = overpass_download(minx,miny,maxx,maxy)
    
    return osmnx_nodes, osmnx_links, overpass_links

def simple_download(minx,miny,maxx,maxy):
    '''
    Just download the default OSMnx bike layer
    '''
    
    #retrieve graph from bbox
    G = ox.graph_from_bbox(maxy, miny, maxx, minx, network_type='bike')

    #convert to gdf
    nodes, links = ox.utils_graph.graph_to_gdfs(G)

    #reset index
    links = links.reset_index()
    
    #simplify columns to geo and id
    links = links[['osmid','bearing','geometry']]

    #reset nodes index
    nodes = nodes.reset_index()
    
    return nodes, links

def download_osmnx(minx,miny,maxx,maxy,network_type):
    
    #retrieve graph from bbox
    G = ox.graph_from_bbox(maxy, miny, maxx, minx, network_type=network_type, simplify=False)
    #G = ox.graph_from_bbox(maxy, miny, maxx, minx, network_type='all_private', simplify=False)
    
    #retrieve graph from the chosen study polygon
    #G = ox.graph_from_polygon(studyarea.geometry[0], network_type='all_private', simplify= False)
    
    #simplify graph unless different osm ids
    G = ox.simplification.simplify_graph(G, strict = False)
    
    #remove directed links (links for each direction)
    G = ox.utils_graph.get_undirected(G)
    
    #get link bearing
    G = ox.bearing.add_edge_bearings(G)

    #plot it for fun
    ox.plot_graph(G)

    #convert to gdf
    nodes, links = ox.utils_graph.graph_to_gdfs(G)

    #reset index
    links = links.reset_index()
    
    #simplify columns to geo and id
    links = links[['osmid','bearing','geometry']]

    #reset nodes index
    nodes = nodes.reset_index()

    return nodes, links

def overpass_download(minx,miny,maxx,maxy):
        
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

    return df