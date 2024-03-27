# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 11:43:38 2022

@author: tpassmore6
"""

import os
from pathlib import Path
import geopandas as gpd
import pandas as pd
import partridge as ptg
from datetime import date
from scipy.spatial import cKDTree
import numpy as np
import networkx as nx
import pickle

#suppress error messages
#import warnings
# from shapely.errors import ShapelyDeprecationWarning
# warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 
# warnings.filterwarnings("ignore", category=FutureWarning)

#make sure thing is str and doesn't contain decimals
def check_type(item):
    item = str(item)
    if len(item.split('.')) > 1:
        item = item.split('.')[0]
    return item

#TODO change to account for new link structure without double links
def create_bike_graph(links,impedance):
    DGo = nx.DiGraph()  # create directed graph
    for ind, row in links.iterrows():
        DGo.add_weighted_edges_from([(int(row['A']), int(row['B']), float(row[impedance]))],weight=impedance)   
    return DGo

def create_walk_graph(links,impedance):
    G = nx.Graph() # create undirected graph
    for ind, row in links.iterrows():
        G.add_weighted_edges_from([(int(row['A']), int(row['B']), float(row[impedance]))],weight=impedance)   
    return G

# taz and transit snapping
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

def process_results(fp:Path):
    '''
    Take in filepath and processs raptor results to get shortest time between tazs pairs
    '''

    with fp.open(mode='rb') as fh:
        trips = pickle.load(fh)
    #only get successful trips
    trips = trips[trips['status']=='success']

    #remove bus to bus (temporary)
    #trips = trips[trips.apply(lambda row: len([x[-2] for x in row['edge_list'] if x[-2] == 'bus']) <= 1 ,axis=1)]

    #groupby to get minimum
    trips = trips.loc[trips.groupby(['src_taz','dest_taz'])['travel_time'].idxmin().to_list(),:]
    return trips

def load_files(settings):
    '''
    This function loads several files that were created in the previous steps.
    '''
    
    #import snapped tazs and stops from find_candidate_stops
    with (settings['output_fp'] / Path(r'snapped_tazs.pkl')).open('rb') as fh:
        snapped_tazs = pickle.load(fh)
    with (settings['output_fp'] / Path(r'snapped_stops.pkl')).open('rb') as fh:
        snapped_stops = pickle.load(fh)
    
    #import the shape map
    shape_map = gpd.read_file(settings['output_fp'] / 'base_layers.gpkg',layer='transit_shapes')
    shape_map['new_route_id'] = shape_map['new_route_id'].astype(str)

    #import shapes with pt sequence
    shapes = pd.read_csv(settings['gtfs_fp'] / 'gtfs_o/shapes.txt')
    shapes['shape_id'] = shapes['shape_id'].astype(str)

    #read stops file
    stops_file = gpd.read_file(settings['output_fp'] / 'base_layers.gpkg',layer='route_and_stop') 

    return snapped_tazs, snapped_stops, shape_map, shapes, stops_file