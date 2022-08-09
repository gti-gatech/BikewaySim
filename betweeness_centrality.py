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
import matplotlib.pyplot as plt
import numpy as np

#make directory/pathing more intuitive later
user_directory = os.fspath(Path.home()) #get home directory and convert to path string
file_directory = r"/Documents/BikewaySimData" #directory of bikewaysim outputs
os.chdir(user_directory+file_directory)


#%% map routes function

def map_trips(links, od_pairs, paths, desired_crs):
    
    #add geo to paths
    paths = pd.merge(paths, links, on=['A','B'], how='left')
    
    #attach od coords
    #convert o and d matches to lines
    od_pairs['ori_geo'] = gpd.points_from_xy(od_pairs['ori_lon'], od_pairs['ori_lat'], crs ='epsg:4326').to_crs(desired_crs)
    od_pairs['dest_geo'] = gpd.points_from_xy(od_pairs['dest_lon'], od_pairs['dest_lat'], crs ='epsg:4326').to_crs(desired_crs)
    od_pairs['ori_match_geo'] = gpd.points_from_xy(od_pairs['ox'], od_pairs['oy'], crs =desired_crs)
    od_pairs['dest_match_geo'] = gpd.points_from_xy(od_pairs['dx'], od_pairs['dy'], crs =desired_crs)

    #lines
    od_pairs['ori_line'] = [LineString(points) for points in zip(od_pairs.ori_geo,od_pairs.ori_match_geo)]
    od_pairs['dest_line'] = [LineString(points) for points in zip(od_pairs.dest_geo,od_pairs.dest_match_geo)]

    #filter
    origs = od_pairs[['trip_id','ori_line']]
    dests = od_pairs[['trip_id','dest_line']]

    #add geo for ori/dest rows
    origs['A'] = 'origin'
    dests['B'] = 'destination'

    #merge to add lines
    paths = pd.merge(paths, origs, on=['A','trip_id'], how = 'left')
    paths = pd.merge(paths, dests, on=['B','trip_id'], how = 'left')
    
    #merge geo columns
    paths = pd.concat([paths[['A','B','sequence','trip_id','time']], 
            paths["geometry"].combine_first(paths["ori_line"]).combine_first(paths["dest_line"])], 
            axis=1)

    #turn to gdf
    paths_geo = gpd.GeoDataFrame(paths, geometry='geometry', crs=desired_crs)

    #create multilinestring for to show each trip
    trip_lines = paths_geo.dissolve(by='trip_id',aggfunc='sum').reset_index()

    return trip_lines

def betweeness_centrality(links, paths, desired_crs):
    #take AB columns
    ab_cols = paths[['A','B']]

    #drop origin/dest col
    ab_cols = ab_cols[-((ab_cols['A']=='origin') | (ab_cols['B']=='destination'))]

    #count occurances
    ab_cols['trips'] = 1
    ab_cols = ab_cols.groupby(['A','B']).sum().reset_index()

    #merge with links
    links_w_trips = pd.merge(links,ab_cols,on=['A','B'],how='left')   
    
    #combine reverse links    
    links_w_trips[['A','B']] = pd.DataFrame(np.sort(links_w_trips[['A','B']], axis=1), columns=["A","B"])
    
    #group it
    links_w_trips = links_w_trips.dissolve(['A','B'], aggfunc='sum').reset_index()
    
    #clean
    links_w_trips = links_w_trips[['A','B','trips']]

    return links_w_trips


#%% run code

#scenarios
scenarios = ['dist','per_dist_2','imp_dist']

#od pairs with closest network node
od_pairs = pd.read_csv(r'bikewaysim_outputs/samples_out/all_tazs_node.csv')

#just od column
impedance = od_pairs['trip_id']

#get link ab column
centrality = gpd.read_file(r'processed_shapefiles/prepared_network/dist/links/links.geojson')[['A','B','geometry']].drop_duplicates()
links_dist = centrality

for scenario in scenarios:
    
    #networkfiles
    nodes = gpd.read_file(rf'processed_shapefiles/prepared_network/{scenario}/nodes/nodes.geojson')
    links = gpd.read_file(rf'processed_shapefiles/prepared_network/{scenario}/links/links.geojson')

    #bikewaysim outputs
    paths = pd.read_csv(rf'bikewaysim_outputs/results/{scenario}/paths_bike.csv')

    #run map trips function to get trip lines
    trip_lines = map_trips(links, od_pairs, paths, 'epsg:2240')
    
    #get trip phys length
    trip_lines['length'] = trip_lines.length
    
    #merge impedance and trip lines
    impedance = pd.merge(impedance,trip_lines[['trip_id','time','length']],on=['trip_id']).rename(columns={'time':f'{scenario}_time','length':f'{scenario}_length'})
    
    #export trip lines
    trip_lines.to_file(r'trb2023/trip_lines.gpkg',layer=scenario,driver='GPKG')

    #get betweenness centrality
    links_w_trips = betweeness_centrality(links, paths, 'epsg:2240')
    
    centrality = pd.merge(centrality, links_w_trips[['A','B','trips']],on=['A','B']).rename(columns={'trips':f'{scenario}_trips'})
    #links_dist = pd.merge(links_dist, links[['A','B','distance']],on=['A','B']).rename(columns={'distance':f'{scenario}'})
    

# impedance = link cost # will be different
# centrality = num of trips on a link # will be different
# links_dist = physical length of link # should be same

#%%find difference in dist and per_dist
impedance['perdiff'] = impedance['per_dist_time'] - impedance['dist_time']

impedance['perdiff'].mean()


#%% find differences btw per_dist and improvement

#get impedance difference between trips
impedance['impdiff'] = impedance['per_dist_time'] - impedance['imp_dist_time']

#get average impedance change
impedance['impdiff'].mean()

#export 


#%% export network centrality
impedance.to_csv(r'trb2023/impedance.csv',index = False)
centrality.to_file(r'trb2023/centrality.gpkg',driver='GPKG')


#%%



# #get od columns
# negative = pd.merge(negative,od_pairs,on='trip_id')[['ori_id','ori_lat','ori_lon','dest_id','dest_lat','dest_lon']]


# origs = negative[['ori_id','ori_lat','ori_lon']].rename(columns={'ori_id':'id','ori_lat':'lat','ori_lon':'lon'})
# dests = negative[['dest_id','dest_lat','dest_lon']].rename(columns={'dest_id':'id','dest_lat':'lat','dest_lon':'lon'})
# comb = origs.append(dests)
# geo = comb.drop_duplicates()
# comb['duplicated'] = comb.id.duplicated()
# comb = comb.groupby('id')['duplicated'].sum().reset_index().merge(geo,on='id')
# comb['geometry'] = gpd.points_from_xy(comb['lon'], comb['lat'], crs="epsg:4326")
# comb = gpd.GeoDataFrame(comb,geometry='geometry')
# comb.plot('duplicated')

# #get geo


# #export

# #%% trip difference

# links_dist['diff_dist'] = links_dist['per_dist'] - links_dist['improvement']
# #count neg
# links_dist.to_file('export_check.geojson')
     
     
#      #(diff_dist<0).sum()

# #export for checking


# #%%


# #merge trips on links by a_b
# merged_trips = pd.merge(bikewaysim_trips_on_links,abm_trips_on_links,left_on='A_B_new',right_on='A_B',suffixes=('_bikewaysim','_abm'))

# #filter to just the trips and geo info
# merged_trips_clean = merged_trips[['trips_bikewaysim','trips_abm','geometry_bikewaysim']]

# #more or less trips
# merged_trips_clean['difference'] = merged_trips_clean['trips_bikewaysim'] - merged_trips_clean['trips_abm'] 

# #geodataframe
# merged_gdf = gpd.GeoDataFrame(merged_trips_clean,geometry='geometry_bikewaysim')

# #plot more positive the number, the more trips that go routed on those links
# merged_gdf.plot(column='difference',cmap='BrBG',legend=True)

# #export and just do in qgis
# merged_gdf.to_file('test.geojson',driver='GeoJSON')


# #%%trip lengths

# #trip lengths
# bikewaysim_trips['trip_length'] = bikewaysim_trips.length / 5280
# bikewaysim_pref_trips['trip_length'] = bikewaysim_pref_trips.length / 5280
# abm_trips['trip_length'] = abm_trips.length / 5280

# #link trip miles
# bikewaysim_trips_on_links['trip_miles'] = bikewaysim_trips_on_links['trips'] * bikewaysim_trips_on_links.length
# bikewaysim_pref_trips_on_links['trip_miles'] = bikewaysim_pref_trips_on_links['trips'] * bikewaysim_pref_trips_on_links.length
# abm_trips_on_links['trip_miles'] = abm_trips_on_links['trips'] * abm_trips_on_links.length



# #%% Find percent of trips on ABM

# #set up buffer
# abm_links = gpd.read_file(abm_fp_links)
# abm_links['buffer'] = abm_links.buffer(10)
# abm_links = abm_links.add_suffix('_abm_links').set_geometry('buffer_abm_links')
# abm_links['dissolve'] = 1
# abm_links = abm_links.dissolve(by='dissolve')


# #%% bikewaysim
# #calculate orginal length
# bikewaysim_trips_on_links['link_length'] = bikewaysim_trips_on_links.length

# #intersect
# test = gpd.overlay(bikewaysim_trips_on_links, abm_links, how='intersection')

# #intersect length
# test['intersect_per'] = test.length / test['link_length']

# #filter
# test = test[test['intersect_per'] > 0.99]

# #print miles on abm
# print(bikewaysim_trips_on_links['trip_miles'].sum())
# print(test['trip_miles'].sum())
# result = round((test['trip_miles'].sum()) / (bikewaysim_trips_on_links['trip_miles'].sum()),2)
# print(result)


# #%% bikewaysim_pref

# #calculate orginal length
# bikewaysim_pref_trips_on_links['link_length'] = bikewaysim_pref_trips_on_links.length

# #intersect
# test = gpd.overlay(bikewaysim_pref_trips_on_links, abm_links, how='intersection')

# #intersect length
# test['intersect_per'] = test.length / test['link_length']

# #filter
# test = test[test['intersect_per'] > 0.99]

# #print miles on abm
# print(bikewaysim_pref_trips_on_links['trip_miles'].sum())
# print(test['trip_miles'].sum())
# result = round((test['trip_miles'].sum()) / (bikewaysim_pref_trips_on_links['trip_miles'].sum()),2)
# print(result)

