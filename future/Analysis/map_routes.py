#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 07:48:01 2020

@author: tannerpassmore
"""


#%%Import modules and import CSV data
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, shape
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

#make directory/pathing more intuitive later
user_directory = os.fspath(Path.home()) #get home directory and convert to path string
file_directory = "\Documents\GitHub\BikewaySim\TransportSim" #directory of bikewaysim network processing code

#change this to where you stored this folder
os.chdir(user_directory+file_directory)

def map_trips(fp_nodes, fp_links, date_name):
    #NOTE need to change this filepath everytime, integrate this into the existing bikewaysim code
    fp_results_csv = fr'trips_bws\results\{date_name}\paths_bike.csv'

    #read in CSV files and set index to N and A
    df_node_csv = gpd.read_file(fp_nodes, ignore_geometry = True)
    df_links = gpd.read_file(fp_links)

    df_results_csv = pd.read_csv(fp_results_csv)

    #not implemented yet
    #read in OD file to get coordinates for OD pair
    #df_samples_in_csv = pd.read_csv(r'trips_bws/samples_in/od_pairs.csv')


    # Join data frames using N in the node csv to the A comlumn in the result csv then plot
    #drop
    df_node_csv = df_node_csv.drop_duplicates()
    
    #joined_results = df_results_csv.join(df_node_csv)
    joined_results = pd.merge(df_results_csv,df_node_csv, left_on='A', right_on='N', how='inner')
    
    #take out trips with less than two points
    test = joined_results.groupby(['trip_id'])['trip_id'].count()
    joined_results = pd.merge(joined_results, test, left_on = 'trip_id', right_index=True, how = 'left', suffixes=(None,'_count'))
    joined_results = joined_results[joined_results['trip_id_count'] > 1 ]
    
    
    joined_results = joined_results.sort_values(by=['trip_id','sequence'], axis=0, ascending = True)


    # Zip the coordinates into a point object and convert to a GeoDataFrame
    geometry = [Point(xy) for xy in zip(joined_results.X, joined_results.Y)]
    gdf = gpd.GeoDataFrame(joined_results, geometry=geometry)
    
    # Aggregate these points with the GroupBy
    gdf = joined_results.groupby(['trip_id'])['geometry'].apply(lambda x: LineString(x.to_list()))
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry').set_crs(epsg=2240)
    
    #export to geojson
    gdf.to_file(rf"Analysis/{date_name}_trips.geojson", driver = 'GeoJSON')

    # Find num of trips on links
    trips_on_links = df_results_csv[['A','B']]
    trips_on_links = trips_on_links[-trips_on_links.isin(['origin','destination'])]
    trips_on_links['trips'] = 1
    trips_on_links['A_B'] = trips_on_links['A'] + "_" + trips_on_links['B']
    trips_on_links = trips_on_links.groupby(['A_B'])['trips'].sum()
    
    df_links_w_trips = pd.merge(df_links,trips_on_links, left_on = 'A_B', right_on = 'A_B' )
    df_links_w_trips.to_file(rf"Analysis/{date_name}_trips_on_links.geojson", driver = 'GeoJSON')
    
    return gdf, df_links_w_trips

def shortest_path_analysis(date_name, trips, trips_on_links):
    #modify later so that trips don't have to be read in if they're already in the memory
    #read in geojson
    if trips is None:
        gdf = gpd.read_file(rf"Analysis/{date_name}_trips.geojson")
        gdf_on_links = gpd.read_file(rf"Analysis/{date_name}_trips_on_links.geojson")
    else:
        gdf = trips
        gdf = trips_on_links

    #trip link length
    gdf['length'] = gdf.length / 5280
    
    #trip miles
    gdf_on_links['trip_miles'] = gdf_on_links['trips'] * gdf_on_links.length
    
    #total trip miles
    print(gdf['trip_miles'].sum())
    
    #describe basic trip characteristics
    print(gdf['length'].describe())
    
    #plot distribution with trips in 0.25 mile bins
    #sns.displot(gdf, x="length", binwidth=0.25)
    
    return gdf, gdf_on_links
    
    #not sure if this runtime is accurate
# =============================================================================
#     #run time
#     run_time = gpd.read_file(fr'trips_bws\results\{date_name}\logs_bike.csv')
#     
#     run_time_tot = round( (pd.to_numeric(run_time['runTime']).sum() / 60), 0)
#     print(run_time_tot)
#     #next see if we can get a list of nodes visited for each trip, and see how closely it matches up to other
# =============================================================================








#%% run map_trips function
bikewaysim_fp_nodes = 'bikewaysim_network/2020 nodes with latlon/2020_nodes_latlon.geojson'
bikewaysim_fp_links = 'bikewaysim_network/2020 links/2020_links.geojson'
bikewaysim_date_name = '062221_072540' #for bikewaysim
bikewaysim_pref_date_name = '062421_092503' #for bikewaysim but with preferences, instead of distances
bikewaysim_trips = None
bikewaysim_pref_trips = None
bikewaysim_trips_on_links = None
bikewaysim_pref_trips_on_links = None

abm_fp_nodes = 'abm/2020 nodes with latlon/2020_nodes_latlon.geojson'
abm_fp_links = 'abm/2020 links/2020_links.geojson'
abm_date_name = '062121_200336' # for abm
abm_trips = None
abm_trips_on_links = None

#%% run map_trips function
bikewaysim_trips, bikewaysim_trips_on_links = map_trips(bikewaysim_fp_nodes, bikewaysim_fp_links, bikewaysim_date_name)
bikewaysim_pref_trips, bikewaysim_pref_trips_on_links = map_trips(bikewaysim_fp_nodes, bikewaysim_fp_links, bikewaysim_pref_date_name)
abm_trips, abm_trips_on_links = map_trips(abm_fp_nodes, abm_fp_links, abm_date_name)

#%% Summary stats

bikewaysim_trips, bikewaysim_trips_on_links = shortest_path_analysis(bikewaysim_date_name, bikewaysim_trips, bikewaysim_trips_on_links)
bikewaysim_pref_trips, bikewaysim_pref_trips_on_links = shortest_path_analysis(bikewaysim_pref_date_name, bikewaysim_pref_trips, bikewaysim_pref_trips_on_links)
abm_trips, abm_trips_on_links = shortest_path_analysis(abm_date_name, abm_trips, abm_trips_on_links)

#%% Find percent of trips on ABM



#set up buffer
abm_links = gpd.read_file(abm_fp_links)
abm_links['buffer'] = abm_links.buffer(10)
abm_links = abm_links.set_geometry('buffer')

#calculate orginal length



#%% Comparing distributions
combined_trips = pd.merge(bikewaysim_trips, abm_trips, how = 'inner', on='trip_id', suffixes=('_bikewaysim','_abm'))

#check for any na values
sns.set(style="darkgrid")

sns.histplot(combined_trips, x="length_abm", label="ABM", binwidth=0.25, color = "red")
sns.histplot(combined_trips, x="length_bikewaysim", label="BikewaySim", binwidth=0.25, color = "skyblue")

plt.legend()
plt.show()

#%% analysis

de


bikewaysim_pref_trips_analysis = gpd.GeoDataFrame(bikewaysim_pref_trips)
bikewaysim_pref_trips_analysis['real_length'] = bikewaysim_pref_trips_analysis.length / 5280

print(bikewaysim_pref_trips_analysis['real_length'].describe())

