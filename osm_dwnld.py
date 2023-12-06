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

def download_osm(studyarea_fp,crs,export_fp,desired_osm_attributes:list=None):
    
    #read in study area and convert to WGS 84 if needed
    if isinstance(studyarea_fp,tuple):
        gdf = gpd.read_file(studyarea_fp[0],layer=studyarea_fp[1])
    else:
        gdf = gpd.read_file(studyarea_fp)

    if gdf.crs != 'epsg:4326':
        gdf.to_crs('epsg:4326',inplace=True)
    # ax = gdf.plot(figsize=(10,10),alpha=0.5,edgecolor='k')
    # cx.add_basemap(ax, crs=gdf.crs)

    #get geometry info from osmnx
    osmnx_nodes, osmnx_links = download_osmnx(gdf)
    
    #get additonal attribute information from overpass
    overpass_links = overpass_download(gdf)
    
    #retrieve only specific osm attributes like highway or oneway if given list
    if isinstance(desired_osm_attributes,list):
        overpass_links = overpass_links[desired_osm_attributes]
    
    #merge osmnx and overpass data on link id
    osm_links = pd.merge(osmnx_links, overpass_links, left_on=('osmid'), right_on=('id'), how = 'inner')
    
    #project
    osm_links.to_crs(crs,inplace=True)

    #remove columns with lists in them (handle these later)
    for col in osm_links.columns.tolist():
        if list in [type(x) for x in osm_links.loc[:,col]]:
            osm_links.drop(columns=col,inplace=True)
            print(f"{col} column removed for containing a list")

    #pickle all attributes as is
    with (export_fp/'osm.pkl').open('wb') as fh:
        pickle.dump(osm_links,fh)

    return osmnx_nodes, osm_links

def simple_download(studyarea):
    '''
    Just download the bike layer
    '''
    
    #convert to unprojected if already projected
    if studyarea.crs != 'EPSG:4326':
        studyarea = studyarea.to_crs('EPSG:4326')
    
    #get bounding box
    minx, miny, maxx, maxy = studyarea.total_bounds
    
    #retrieve graph from bbox
    G = ox.graph_from_bbox(maxy, miny, maxx, minx, network_type='bike')
    
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

    return df