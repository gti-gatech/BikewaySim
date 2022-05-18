# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:54:55 2021

@author: tpassmore6
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
file_directory = "\Documents\GitHub\BikewaySim_archive\TransportSim" #directory of bikewaysim network processing code
os.chdir(user_directory+file_directory)

#%% filepaths
bikewaysim_fp_nodes = 'bikewaysim_network/2020 nodes with latlon/2020_nodes_latlon.geojson'
bikewaysim_fp_links = 'bikewaysim_network/2020 links/2020_links.geojson'
bikewaysim_date_name = '062221_072540' #for bikewaysim
bikewaysim_pref_date_name = '062421_092503' #for bikewaysim but with preferences, instead of distances

abm_fp_nodes = 'abm/2020 nodes with latlon/2020_nodes_latlon.geojson'
abm_fp_links = 'abm/2020 links/2020_links.geojson'
abm_date_name = '062121_200336' # for abm

#bikewaysim_trips = gpd.read_file(rf"Analysis/{bikewaysim_date_name}_trips.geojson")
bikewaysim_trips_on_links = gpd.read_file(rf"Analysis/{bikewaysim_date_name}_trips_on_links.geojson")

#bikewaysim_pref_trips = gpd.read_file(rf"Analysis/{bikewaysim_pref_date_name}_trips.geojson")
#bikewaysim_pref_trips_on_links = gpd.read_file(rf"Analysis/{bikewaysim_pref_date_name}_trips_on_links.geojson")

#abm_trips = gpd.read_file(rf"Analysis/{abm_date_name}_trips.geojson")
abm_trips_on_links = gpd.read_file(rf"Analysis/{abm_date_name}_trips_on_links.geojson")

#%%

#if contains 11 at begining then change to 10
bikewaysim_trips_on_links.loc[bikewaysim_trips_on_links['A'].str[0:2] == '11','A_new'] = '10' + bikewaysim_trips_on_links['A'].str[2:]
bikewaysim_trips_on_links.loc[bikewaysim_trips_on_links['B'].str[0:2] == '11','B_new'] = '10' + bikewaysim_trips_on_links['B'].str[2:]
bikewaysim_trips_on_links['A_B_new'] = bikewaysim_trips_on_links['A_new'] + '_' + bikewaysim_trips_on_links['B_new']


#merge trips on links by a_b
merged_trips = pd.merge(bikewaysim_trips_on_links,abm_trips_on_links,left_on='A_B_new',right_on='A_B',suffixes=('_bikewaysim','_abm'))

#filter to just the trips and geo info
merged_trips_clean = merged_trips[['trips_bikewaysim','trips_abm','geometry_bikewaysim']]

#more or less trips
merged_trips_clean['difference'] = merged_trips_clean['trips_bikewaysim'] - merged_trips_clean['trips_abm'] 

#geodataframe
merged_gdf = gpd.GeoDataFrame(merged_trips_clean,geometry='geometry_bikewaysim')

#plot more positive the number, the more trips that go routed on those links
merged_gdf.plot(column='difference',cmap='BrBG',legend=True)

#export and just do in qgis
merged_gdf.to_file('test.geojson',driver='GeoJSON')


#%%trip lengths

#trip lengths
bikewaysim_trips['trip_length'] = bikewaysim_trips.length / 5280
bikewaysim_pref_trips['trip_length'] = bikewaysim_pref_trips.length / 5280
abm_trips['trip_length'] = abm_trips.length / 5280

#link trip miles
bikewaysim_trips_on_links['trip_miles'] = bikewaysim_trips_on_links['trips'] * bikewaysim_trips_on_links.length
bikewaysim_pref_trips_on_links['trip_miles'] = bikewaysim_pref_trips_on_links['trips'] * bikewaysim_pref_trips_on_links.length
abm_trips_on_links['trip_miles'] = abm_trips_on_links['trips'] * abm_trips_on_links.length

















#%% Find percent of trips on ABM

#set up buffer
abm_links = gpd.read_file(abm_fp_links)
abm_links['buffer'] = abm_links.buffer(10)
abm_links = abm_links.add_suffix('_abm_links').set_geometry('buffer_abm_links')
abm_links['dissolve'] = 1
abm_links = abm_links.dissolve(by='dissolve')


#%% bikewaysim
#calculate orginal length
bikewaysim_trips_on_links['link_length'] = bikewaysim_trips_on_links.length

#intersect
test = gpd.overlay(bikewaysim_trips_on_links, abm_links, how='intersection')

#intersect length
test['intersect_per'] = test.length / test['link_length']

#filter
test = test[test['intersect_per'] > 0.99]

#print miles on abm
print(bikewaysim_trips_on_links['trip_miles'].sum())
print(test['trip_miles'].sum())
result = round((test['trip_miles'].sum()) / (bikewaysim_trips_on_links['trip_miles'].sum()),2)
print(result)


#%% bikewaysim_pref

#calculate orginal length
bikewaysim_pref_trips_on_links['link_length'] = bikewaysim_pref_trips_on_links.length

#intersect
test = gpd.overlay(bikewaysim_pref_trips_on_links, abm_links, how='intersection')

#intersect length
test['intersect_per'] = test.length / test['link_length']

#filter
test = test[test['intersect_per'] > 0.99]

#print miles on abm
print(bikewaysim_pref_trips_on_links['trip_miles'].sum())
print(test['trip_miles'].sum())
result = round((test['trip_miles'].sum()) / (bikewaysim_pref_trips_on_links['trip_miles'].sum()),2)
print(result)

#%% most travelled links

ab



#%%summary function

 
    #total trip miles
    print(gdf['trip_miles'].sum())
    
    #describe basic trip characteristics
    print(gdf['length'].describe())
    
    #plot distribution with trips in 0.25 mile bins
    #sns.displot(gdf, x="length", binwidth=0.25)
    
    return gdf, gdf_on_links



#%% Summary stats







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