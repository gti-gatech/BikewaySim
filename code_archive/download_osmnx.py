#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 08:18:24 2021

@author: tannerpassmore
"""

#%% download osmnx data

import osmnx as ox
import geopandas as gpd
import os
import pandas as pd

def download_osmnx(studyarea):

    #convert to unprojected if already projected
    if studyarea.crs != 'EPSG:4326':
        studyarea = studyarea.to_crs('EPSG:4326')
    

    #retrieve graph from the chosen study polygon
    G = ox.graph_from_polygon(studyarea.geometry[0], network_type='all_private', simplify= False)
    
    #simplify graph unless different osm ids
    G = ox.simplification.simplify_graph(G, strict = False)
    
    #remove directed links
    G = ox.utils_graph.get_undirected(G)
    
    #plot it for fun
    ox.plot_graph(G)
    
    #export
    #ox.io.save_graph_geopackage(G, "base_shapefiles/osm/osmnx.gpkg")

    #convert to gdf
    nodes, links = ox.utils_graph.graph_to_gdfs(G)

    #simplify columns to geo and id
    links = links[['id','geometry']]

    return nodes, links

#%% testing osmnx

# south, west, north, east = [33.778005862211,-84.372907876968,33.7860671619,-84.361588954926]

# G = ox.graph.graph_from_bbox(north, south, east, west, network_type='all_private', simplify = False)
# ox.plot_graph(G)


# G = ox.simplification.simplify_graph(G, strict = False)
# G = ox.utils_graph.get_undirected(G)

# nodes, links = ox.utils_graph.graph_to_gdfs(G)

# G2 = ox.graph.graph_from_bbox(north, south, east, west, network_type='all_private', simplify = False)
# ox.plot_graph(G2)

# G2 = ox.utils_graph.get_undirected(G2)

# nodes2, links2 = ox.utils_graph.graph_to_gdfs(G2)