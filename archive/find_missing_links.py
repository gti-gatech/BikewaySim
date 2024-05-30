# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 12:29:31 2023

@author: tpassmore6
"""

'''
In the map_match.py code, we record up 10 points after the last recorded point if
the match fails to complete. Here we use those points to see if they overlap with any 

last 10 points not always going to be right where the match breaks (up to 1000 ft past?)

Second method:
    Buffer matched line
    Intersect with the GPS points
    Of non-intersected points, find the first 10 consective points by sequence (make sure sequence is reset)
    Intersect with raw links file
    Retrieve service links first and add their osmids to the file
        see if that fixes routing

'''



import geopandas as gpd
import pandas as pd
from pathlib import Path
import pickle

#export filepath
export_fp = Path.home() / 'Downloads/cleaned_trips'

#%%

#import matched traces
with (export_fp/'matched_traces.pkl').open('rb') as fh:
     matched_traces = pickle.load(fh)

#load dict of traces (replace with database)
with (export_fp/'coords_dict.pkl').open('rb') as fh:
    simp_dict = pickle.load(fh)

#import raw network
raw2 = gpd.read_file(r"C:\Users\tpassmore6\Documents\TransitSimData\networks\filtered.gpkg",layer='osm_links_raw').to_crs('epsg:2240')

#import filtered network
network_fp = r"C:\Users\tpassmore6\Documents\TransitSimData\networks\final_network.gpkg"
edges = gpd.read_file(network_fp,layer="links")


with Path(r"C:/Users/tpassmore6/Documents/TransitSimData/networks/osm.pkl").open('rb') as fh:
    test = pickle.load(fh)[['osm_A_B','osmid']]

#remove raw links that are already there
raw2 = raw2[-raw2['osm_A_B'].isin(set(edges['A_B'].tolist()))]

#buffer raw2
raw2.geometry = raw2.buffer(100)

#%%

matched_trace = matched_traces[17825]
one_trace = simp_dict[17825]

one_trace['sequence'] = range(0,one_trace.shape[0])

unmatched2 = pd.DataFrame()


#%%loop thru and buffer each line
matched_trace['matched_trip'].geometry = matched_trace['matched_trip'].buffer(100)

#then intersect with the points
intersect = gpd.overlay(one_trace,matched_trace['matched_trip'],how='difference')

#find first 10 consecutive points
consecutive = intersect[intersect['sequence'].diff() == 1]
try:
    first_10 = consecutive.iloc[0:10]
except:
    first_10 = consecutive

#add to big points file
unmatched2 = pd.concat([unmatched2,first_10],ignore_index=True)

#%%

#intersect
test = gpd.overlay(unmatched2,raw2)

candidates = raw2[raw2['osm_A_B'].isin(test['osm_A_B'])]

#%%

import contextily as cx

# print to examine
ax = unmatched2.plot(figsize=(5,5),alpha=0.5,edgecolor='k')
ax.set_xlim(unmatched2.total_bounds[0]-1000,unmatched2.total_bounds[2]+1000)
ax.set_ylim(unmatched2.total_bounds[1]-1000,unmatched2.total_bounds[3]+1000)
cx.add_basemap(ax, crs=unmatched2.crs, source=cx.providers.CartoDB.Positron)


#%%import coords that weren't covered by matched line
all_unmatched_points = gpd.read_file(export_fp/"matched_traces/unmatched_points.gpkg",layer='unmatched_points')

#import filtered network
network_fp = export_fp / "final_network.gpkg"
edges = gpd.read_file(network_fp,layer="links")

#import raw network
raw = gpd.read_file(r"C:\Users\tpassmore6\Documents\TransitSimData\networks\osm\transitsim.gpkg",layer='links').to_crs('epsg:2240')[['osmid','geometry']]

#buffer by 100ft and dissolve both raw and filtered network
edges.geometry = edges.buffer(100)
raw.geometry = raw.buffer(100)
raw = raw.dissolve('osmid')
raw.reset_index(inplace=True)
raw.rename(columns={'index':'osmid'})

#%%

#find points that are only in raw
within_raw = gpd.overlay(not_covered,raw)
not_within_edges = gpd.overlay(within_raw,edges,how='difference')

#find points that are not in either network to check
neither = not_covered[-(not_covered['index'].isin(within_raw['index']))]

#get list of osmids to check
osmids = set(not_within_edges['osmid'].tolist())


#%%

#export missing ones
not_within_edges.to_file(export_fp/"matched_traces/not_covered.gpkg",layer='in_raw')

neither.to_file(export_fp/"matched_traces/not_covered.gpkg",layer='neither')

#need to attach trip numbdr