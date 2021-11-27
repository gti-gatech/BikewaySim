# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 12:27:29 2021

@author: tpassmore6
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
import os

#make directory/pathing more intuitive later
user_directory = os.fspath(Path.home()) #get home directory and convert to path string
file_directory = "/Documents/GitHub/BikewaySim_Network_Processing" #directory of bikewaysim network processing code

#change this to where you stored this folder
os.chdir(user_directory+file_directory)

#create attribtue summary folders
if not os.path.exists('attribute_summaries'):
    os.makedirs('attribute_summaries')  

#%% Create CSV/DataFrame

def create_sum_table(gdf):

    gdf = gdf.drop(columns = 'geometry')

    #number of unique values
    num_unique = gdf.nunique().rename('Num Unique')
    
    #filter out values with too many unique values
    gdf_trim = gdf
    #gdf_trim = gdf[list(num_unique[num_unique <= 10].index)]
    
    #how many non-NA values
    non_na_count = gdf_trim.count().rename('Non NA Count')
    
    #how many possible non-NA values
    tot_values = len(gdf_trim)
    #completion rate
    completion = round( non_na_count / tot_values * 100, 0).rename('Completion')
    
    #number of unique values
    num_unique = gdf_trim.nunique().rename('Num Unique')
    
    #enumeration
    enumeration = pd.Series({c: gdf_trim[c].value_counts() for c in gdf_trim}).rename('Enumeration')
    
    #data types
    data_types = pd.Series({c: gdf_trim[c].apply(lambda x: type(x).__name__).value_counts() for c in gdf_trim}).rename('Data Types')
    
    #condensed table
    export = pd.concat([non_na_count, completion, num_unique, enumeration, data_types], axis =1)    
    
# =============================================================================
#     #exploded table, to access each unique value
#     export = pd.concat([non_na_count, completion, num_unique], axis =1).reset_index().rename(columns={'index':'Variable Name'})
#     
#     df = pd.DataFrame(columns={'Variable Name','Values','Num of Values'})
#     for col in gdf_trim:
#         values = list(gdf_trim[col].unique())
#         num_values = list(gdf_trim[col].value_counts(dropna=False))
#         name = [col] * len(values)
#         
#         df_temp = pd.DataFrame({'Variable Name': name, 'Values': values, 'Num of Values': num_values})
#         df = df.append(df_temp)
#         
#     exploded_table = pd.merge(export, df, on = 'Variable Name').reset_index(drop=True)
#     #export.to_csv(rf'attribute_summaries/{network_name}_{study_area}_{link_type}_sumfull.csv')
# =============================================================================
    
    return export

def import_geojson(network_name, study_area, link_type):
        gdf = gpd.read_file(rf'Processed_Shapefiles/{network_name}/{network_name}_{study_area}_{link_type}.geojson')   
        return gdf


# #%% Run all

network_name = ['osm', 'navstreets', 'abm']
study_area = ['study_area']
link_type = ['base', 'road', 'bike', 'service']

# =============================================================================
# for x in network_name:
#      for y in study_area:
#          for z in link_type:
#              if os.path.exists(rf'Processed_Shapefiles/{x}/{x}_{y}_{z}.geojson'):
#                  gdf = import_geojson(x, y, z)
#                  export = create_sum_table(gdf)
#                  export.to_csv(rf'attribute_summaries/{x}_{y}_{z}.csv')
# =============================================================================

#%% overlap analysis

#use this to have a general understanding of how networks stack up
def overlap(base_network, base_name, joining_network, joining_name, tolerance):

    #filter to only geometry
    #base_network = gpd.GeoDataFrame(crs = base_network.crs, geometry = base_network.geometry)


    #buffer base network
    base_network['buffer'] = base_network.buffer(tolerance)

    #set buffer to active geometry
    base_network = base_network.set_geometry('buffer')

    #intersect with joining network
    intersection = gpd.overlay(base_network, joining_network, how='intersection')



    # for x in tolerances:
    #     #tolerance is in feet
    #     #give joining_network buffer and dissolve it
    #     buffer = joining_network.buffer(x).unary_union 
    #     buffer = gpd.GeoDataFrame(crs=joining_network.crs, geometry=[buffer])
        
    #     #intersect within
    #     overlap_results = gpd.overlay(base_network, buffer, how='intersection')
        
    #     # #find sum of line lengths or sum of shared lines
    #     # line_lengths = intersect.length.sum() / 5280
    
    #     # #find length of lines with no match
    #     # difference = (base_network.length.sum() - line_lengths) / 5280
        
    #     # #find ratio of matching
    #     # ratio = base_network.length.sum() / 5280 / line_lengths
        
    #     # #concat data
    #     # new_results = pd.DataFrame({'base network': base_name,
    #     #                            'buffer network': joining_name,
    #     #                            'link type': link_type,
    #     #                            'tolerance': x,
    #     #                            'line lengths': line_lengths,
    #     #                            'difference': difference,
    #     #                            'ratio': ratio
    #     #                            }, index = [f'{base_name}+{joining_name}'])
    #     # #append data to data frame
    #     # overlap_results = overlap_results.append(new_results)
    
    return intersection















# #%% roadway classification
# variables_of_interest_here = ['FUNC_CLASS']

# variables_of_interset_abm = []

# variables_of_interest_osm = ['highway']



# #%% speedlimit analysis

# variables_of_interest_here = ['SPEED_CAT']

# variables_of_interset_abm = []

# variables_of_interest_osm = ['lanes', 'lanes:forward', 'lanes:backward' ]


# #speed cats from HERE
# # =============================================================================
# # 1: > 80 MPH
# # 2: 65-80 MPH
# # 3: 55-64 MPH
# # 4: 41-54 MPH
# # 5: 31-40 MPH
# # 6: 21-30 MPH
# # 7: 6-20 MPH
# # =============================================================================


# #speed bins
# speed_bins = {
#     '25 mph': 7,
#     '35 mph': 5,
#     '30 mph': 7,
#     '55 mph': 3,
#     '15 mph': 7,
#     '5 mph': 8,
#     '2 mph': 8,
#     '10 mph': 7   
#     }

# osm = osm.replace({'maxspeed': speed_bins})

# osm['maxspeed'].value_counts(dropna= False)



# #%% lanewidth analysis
# variables_of_interest_here = ['PHYS_LANES', 'TO_LANES', 'FROM_LANES']

# variables_of_interset_abm = ['LANES', 'LANESTOTAL']

# variables_of_interest_osm = [ 'lanes', 'lanes:forward', 'lanes:backward', ]

# osm['lanes'].value_counts(dropna=False)
# here['PHYS_LANES'].value_counts(dropna=False)



# #%% bike facility

# #none for ABM and HERE

# variables_of_interest_osm = ['highway','highway_1', 'bicycle', 'cycleway', 'cycleway:right', 'cycleway:left', 'cycleway:both']



# #%% run overlap

# #list of tolerances to try                                
# tolerances = list(range(5,50,5))

# #empty dataframe
# columns = ['base network','buffer network', 'link type','tolerance', 'line lengths', 'difference', 'ratio']
# overlap_results = pd.DataFrame(columns= columns)

# #import base
# osm = import_geojson('osm', 'study_area', 'base')
# here = import_geojson('navstreets', 'study_area', 'base')
# abm = import_geojson('abm', 'study_area', 'base')

# #overlap base
# overlap_results = overlap(osm, 'osm', here, 'here', 'base', tolerances, overlap_results)
# overlap_results = overlap(osm, 'osm', abm, 'abm', 'base', tolerances, overlap_results)
# overlap_results = overlap(here, 'here', osm, 'osm', 'base', tolerances, overlap_results)
# overlap_results = overlap(here, 'here', abm, 'abm', 'base', tolerances, overlap_results)    
# overlap_results = overlap(abm, 'abm', osm, 'osm', 'base', tolerances, overlap_results)
# overlap_results = overlap(abm, 'abm', here, 'here', 'base', tolerances, overlap_results)

# #import road
# osm = import_geojson('osm', 'study_area', 'road')
# here = import_geojson('navstreets', 'study_area', 'road')
# abm = import_geojson('abm', 'study_area', 'road')

# #overlap road
# overlap_results = overlap(osm, 'osm', here, 'here', 'road', tolerances, overlap_results)
# overlap_results = overlap(osm, 'osm', abm, 'abm', 'road', tolerances, overlap_results)
# overlap_results = overlap(osm, 'here', here, 'osm', 'road', tolerances, overlap_results)
# overlap_results = overlap(osm, 'here', here, 'abm', 'road', tolerances, overlap_results)   

# #import bike
# osm = import_geojson('osm', 'study_area', 'bike')
# here = import_geojson('navstreets', 'study_area', 'bike')

# #overlap bike
# overlap_results = overlap(osm, 'osm', here, 'here', 'bike', tolerances, overlap_results)
# overlap_results = overlap(here, 'here', osm, 'osm', 'bike', tolerances, overlap_results)
 

# #import service
# osm = import_geojson('osm', 'study_area', 'service')
# here = import_geojson('navstreets', 'study_area', 'service')

# #overlap service
# overlap_results = overlap(osm, 'osm', here, 'here', 'service', tolerances, overlap_results)
# overlap_results = overlap(here, 'here', osm, 'osm', 'service', tolerances, overlap_results)


# #export
# print(overlap_results.head())
# overlap_results.to_csv('overlap_results.csv')

# # =============================================================================
# # outputs, different depending on direction
# # meaning: X miles of gdf1 links are within tolerance of gdf2 links. These links are likely shared
# # we'd expect OSM to be higher than HERE in some cases because of the presense of sidewalks
# # =============================================================================
# # 1.0
# # 370.0
# # 462.0
# # 544.0
# # 1.0
# # 343.0
# # 377.0
# # 384.0
# # =============================================================================
# # =============================================================================


    