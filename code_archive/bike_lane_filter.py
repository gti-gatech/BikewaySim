
#%%import cell
import geopandas as gpd
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' #get rid of copy warning
import numpy as np
import os
import time
import pickle
from shapely.geometry import Point, LineString
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

#set dark grid
sns.set()

#make directory/pathing more intuitive later
user_directory = os.fspath(Path.home()) #get home directory and convert to path string
file_directory = "\Documents\GitHub\BikewaySim_Network_Processing" #directory of bikewaysim network processing code

#change this to where you stored this folder
os.chdir(user_directory+file_directory)

#%% match arc/coa facilities to network

def overlapping_links(df1,df2_buffer,df2_name):
    
    df1 = bikewaysim_links_df_AB
    df2_buffer = bike_lane_df
    
    #create buffer and set geometry to buffer
    df2_buffer['buffer_geo'] = df2_buffer.buffer(30)
    df2_buffer = df2_buffer.set_geometry('buffer_geo')

    #find original length of links, use this to find percentage overlap
    #had to use og instead of original because other overlap code
    df1['og_length'] = df1.length

    #perform overlay with df1 and df2_buffer
    overlapping_links_df = gpd.overlay(df1,df2_buffer, how = 'intersection')

    #percentage overlap
    overlapping_links_df['percent_overlap'] = overlapping_links_df.length / overlapping_links_df['og_length']
    
    #overlap length
    overlapping_links_df['overlap_length'] = overlapping_links_df.length

    #filter to only get joining links with high percentage of overlap with bikewaysim links
    overlapping_links_df_filter = overlapping_links_df[overlapping_links_df['percent_overlap'] >= 0.99]

    #groupby A_B_abm and find max overlap
    idx = overlapping_links_df_filter.groupby(['bikewaysim_A','bikewaysim_B'])['overlap_length'].transform(max) == overlapping_links_df_filter['overlap_length']
    
    #use index to select joining links with the max amount of overlap
    overlapping_links_df_filter = overlapping_links_df_filter[idx]

    #filter columns
    #overlapping_links_df = overlapping_links_df.filter(['A_abmsplit','B_abmsplit','A_navteq','B_navteq'])


    #check shapefile
    #overlapping_links_df.to_file(r'Processed_Shapefiles/testing/overlapping_links.shp')


    #make sure each link has only one bike lane match
    
    return bikewaysim_links_df

#%% add bike lanes

def bike_lanes(bikewaysim_links_df, bike_lane_df, bike_lane_df_name):

    #add suffix to bike lane columns so we know where it came from
    bike_lane_df = bike_lane_df.add_suffix(f'_{bike_lane_df_name}')

    #run overlapping_links code
    links_w_bike_lanes = overlapping_links(bikewaysim_links_df_AB, bike_lane_df, bike_lane_df_name) 

    #join back other attributes
    links_w_bike_lanes = pd.merge(bikewaysim_links_df,links_w_bike_lanes, on=['bikewaysim_A','bikewaysim_B']).set_geometry('geometry')

    return links_w_bike_lanes

#%% import

osm_road = gpd.read_file('Processed_Shapefiles/osm/osm_study_area_road_id.geojson').to_crs(epsg=2240)
arc_bike = gpd.read_file('Processed_Shapefiles/bike_inventories/arc_bike_cleaned.geojson').to_crs(epsg=2240)
coa_bike = gpd.read_file('Processed_Shapefiles/bike_inventories/coa_bike_cleaned.geojson').to_crs(epsg=2240)

bikewaysim_links_df = gpd.read_file('Processed_Shapefiles/bikewaysim/bikewaysim_links_v3.geojson').to_crs(epsg=2240)


#add filter method for osm to get bike lanes
osm_filter = ['cycleway', 'cycleway:right', 'cycleway:both', 'highway_1']

x = lambda row: False if row[osm_filter].isnull().all() else True
osm_bike_lanes = osm_road[osm_road.apply(x, axis = 1) == True]


#%% run

# look at OSM bike lanes first
#bikewaysim_links_w_osm_bike_lanes = bike_lanes(bikewaysim_links_df,osm_bike_lanes, 'osm')
#bikewaysim_links_w_osm_bike_lanes.plot()

# coa second
#bikewaysim_links_w_osm_coa_bike_lanes = bike_lanes(bikewaysim_links_w_osm_bike_lanes, coa_bike, 'coa')
#bikewaysim_links_w_osm_coa_bike_lanes.plot()

# arc last
#bikewaysim_links_w_osm_coa_arc_bike_lanes = bike_lanes(bikewaysim_links_w_osm_coa_bike_lanes, arc_bike, 'arc')
#bikewaysim_links_w_osm_coa_arc_bike_lanes.plot()


#%% testing

bikewaysim_links_df = bikewaysim_links_df
bike_lane_df = arc_bike
bike_lane_df_name = 'arc'

#add suffix to bike lane columns so we know where it came from
bike_lane_df = bike_lane_df.add_suffix(f'_{bike_lane_df_name}').set_geometry(f'geometry_{bike_lane_df_name}')

#filter the bikewaysim_links_df to only the neccessary
bikewaysim_links_df_AB = bikewaysim_links_df[['bikewaysim_A','bikewaysim_B','geometry']]
#rename geo to something memorable
bikewaysim_links_df_AB = bikewaysim_links_df_AB.rename(columns={'geometry':'bikewaysim_geometry'}).set_geometry('bikewaysim_geometry')

df1 = bikewaysim_links_df_AB
df2_buffer = bike_lane_df

#%%

#create buffer and set geometry to buffer
df2_buffer['buffer_geo'] = df2_buffer.buffer(30)
df2_buffer = df2_buffer.set_geometry('buffer_geo')

#find original length of links, use this to find percentage overlap
#had to use og instead of original because other overlap code
df1['og_length'] = df1.length

#perform overlay with df1 and df2_buffer
overlapping_links_df = gpd.overlay(df1,df2_buffer, how = 'intersection')

#percentage overlap
overlapping_links_df['percent_overlap'] = overlapping_links_df.length / overlapping_links_df['og_length']

#overlap length
overlapping_links_df['overlap_length'] = overlapping_links_df.length

#filter to only get joining links with high percentage of overlap with bikewaysim links
overlapping_links_df_filter = overlapping_links_df[overlapping_links_df['percent_overlap'] >= 0.90]

#groupby A_B_abm and find max overlap
idx = overlapping_links_df_filter.groupby(['bikewaysim_A','bikewaysim_B'])['overlap_length'].transform(max) == overlapping_links_df_filter['overlap_length']

#use index to select joining links with the max amount of overlap
overlapping_links_df_filter = overlapping_links_df_filter[idx]

#drop those columns
overlapping_links_df_filter = overlapping_links_df_filter.drop(columns={'og_length','percent_overlap','overlap_length'})

#%%end function


#drop the geo columns
overlapping_links_df_filter = overlapping_links_df_filter.drop(columns={'geometry',f'geometry_{bike_lane_df_name}'})

#make sure each link has only one bike lane match

#join back other attributes
links_w_bike_lanes = pd.merge(bikewaysim_links_df,overlapping_links_df_filter, on=['bikewaysim_A','bikewaysim_B'], how='left').set_geometry('geometry')


#check shapefile
links_w_bike_lanes.to_file(r'Processed_Shapefiles/bikewaysim/bikewaysim_links_v4.geojson', driver='GeoJSON')

#%% analysis

round(links_w_bike_lanes.length.sum() / 5280, 0)
round(links_w_bike_lanes.length.mean() , 0)


#%% bikewaysim links v5: removing disconnected links

gdf = gpd.read_file(r'Processed_Shapefiles/bikewaysim/bikewaysim_links_v4.geojson', driver='GeoJSON')

print(gdf.length.sum() / 5280)
print(len(gdf))

#create a network clip layer
#read in buffer file
bikewaysim_buffer = gpd.read_file('Processed_Shapefiles/discconected_links/connected_network_bikewaysim.geojson')
bikewaysim_buffer = bikewaysim_buffer[['geometry','area']]

gdf = gpd.overlay(gdf, bikewaysim_buffer, how='intersection')

gdf.to_file(r'Processed_Shapefiles/bikewaysim/bikewaysim_links_v5.geojson', driver ='GeoJSON')

len(gdf)

print(gdf.length.sum() / 5280)

print(len(gdf))

gdf_nodes = gpd.read_file(r'Processed_Shapefiles/bikewaysim/bikewaysim_nodes_v4.geojson', driver='GeoJSON').set_crs(epsg=2240, allow_override = True)
print(len(gdf_nodes))

gdf['buffer'] = gdf.buffer(0.001)
gdf = gdf.set_geometry('buffer')

gdf_nodes = gpd.overlay(gdf_nodes, gdf, how='intersection')
print(len(gdf_nodes))

print(len(gdf['bikewaysim_A'].append(gdf['bikewaysim_B']).drop_duplicates()))

#https://stackoverflow.com/questions/31622618/removing-duplicate-geometries-in-shapely

polys = gdf_nodes['geometry']

uniqpolies = []
for poly in polys:

    if not any(p.equals(poly) for p in uniqpolies):
        uniqpolies.append(poly)





#%% this works in QGIS and ArcMAP but not geopandas
gdf['buffer'] = gdf.buffer(10) # 10 ft buffer
gdf = gdf.set_geometry('buffer') # set geom to buffer
gdf['dissolve'] = 1
multipart_links = gdf.dissolve(by='dissolve')
singlepart_links = pd.Series(multipart_links.iloc[0].geometry).tolist()
gdf = gpd.GeoDataFrame({'geometry':singlepart_links}, geometry='geometry').set_crs(epsg=2240)




#%%another method for finding disconnected links
#first pass
#just elminated links that have two nodes that don't connnect to anything
just_nodes = gdf[['abm_A','abm_B','navstreets_A','navstreets_B','osm_A','osm_B','bikewaysim_A','bikewaysim_B']]

abm_nodes = pd.DataFrame(gdf['abm_A'].append(gdf['abm_B']), columns={"abm_id"})
abm_nodes['abm_num_links'] = 1
abm_nodes = abm_nodes.groupby('abm_id').sum()

navstreets_nodes = pd.DataFrame(gdf['navstreets_A'].append(gdf['navstreets_B']), columns={"navstreets_id"})
navstreets_nodes['navstreets_num_links'] = 1
navstreets_nodes = navstreets_nodes.groupby('navstreets_id').sum()

osm_nodes = pd.DataFrame(gdf['osm_A'].append(gdf['osm_B']), columns={"osm_id"})
osm_nodes['osm_num_links'] = 1
osm_nodes = osm_nodes.groupby('osm_id').sum()

bikewaysim_nodes = pd.DataFrame(gdf['bikewaysim_A'].append(gdf['bikewaysim_B']), columns={"bikewaysim_id"})
bikewaysim_nodes['bikewaysim_num_links'] = 1
bikewaysim_nodes = bikewaysim_nodes.groupby('bikewaysim_id').sum()


gdf = pd.merge(gdf, abm_nodes, how = 'left', left_on = 'abm_A', right_index = True).rename(columns={'abm_num_links':'abm_A_num_links'})
gdf = pd.merge(gdf, abm_nodes, how = 'left', left_on = 'abm_B', right_index = True).rename(columns={'abm_num_links':'abm_B_num_links'})
gdf = pd.merge(gdf, navstreets_nodes, how = 'left', left_on = 'navstreets_A', right_index = True).rename(columns={'navstreets_num_links':'navstreets_A_num_links'})
gdf = pd.merge(gdf, navstreets_nodes, how = 'left', left_on = 'navstreets_B', right_index = True).rename(columns={'navstreets_num_links':'navstreets_B_num_links'})
gdf = pd.merge(gdf, osm_nodes, how = 'left', left_on = 'osm_A', right_index = True).rename(columns={'osm_num_links':'osm_A_num_links'})
gdf = pd.merge(gdf, osm_nodes, how = 'left', left_on = 'osm_B', right_index = True).rename(columns={'osm_num_links':'osm_B_num_links'})
gdf = pd.merge(gdf, bikewaysim_nodes, how = 'left', left_on = 'bikewaysim_A', right_index = True).rename(columns={'bikewaysim_num_links':'bikewaysim_A_num_links'})
gdf = pd.merge(gdf, bikewaysim_nodes, how = 'left', left_on = 'bikewaysim_B', right_index = True).rename(columns={'bikewaysim_num_links':'bikewaysim_B_num_links'})

len(gdf[(gdf['abm_A_num_links'] == 1) & (gdf['abm_B_num_links'] == 1)])

len(gdf[(gdf['navstreets_A_num_links'] == 1) & (gdf['navstreets_B_num_links'] == 1)])

len(gdf[(gdf['osm_A_num_links'] == 1) & (gdf['osm_B_num_links'] == 1)])

len(gdf[(gdf['bikewaysim_A_num_links'] == 1) & (gdf['bikewaysim_B_num_links'] == 1)])




#%% Bike lane change over time 

# #plotting help: https://towardsdatascience.com/beautify-coronavirus-bar-charts-ac636d314d98

# df = coa_bike[coa_bike.YEAR_INSTA != 0] #have a date
# df = df.groupby(['YEAR_INSTA','FACILITY_T'])['LENGTH_MIL'].sum()

# df_nd = coa_bike[coa_bike.YEAR_INSTA == 0] #no date
# df_nd = df_nd.groupby('YEAR_INSTA')['LENGTH_MIL'].sum()
# df_nd = df_nd.rename(index={0:"Non Dated"})
# df = df.append(df_nd).cumsum().round(1)

# y_vals = df.index.to_series().rename('Year Installed')
# x_vals = df.rename('Total Lane Miles')

# plt.figure(figsize=(16,9))
# sns.barplot(x=x_vals, y=y_vals, color = 'green')
# plt.title('Bike Infractructure Over Time in the City of Atlanta', size = 15)

# plt.savefig('bike_infra_over_time', dpi = 150)

# plt.show()


# #df = pd.DataFrame({'Year Installed':list(df.index),'Miles':df})

# #plt.bar(df['Year Installed'], df['Miles'])
# #plt.show()
# #df.plot.bar(x='Year Installed', y = 'Miles', title='Bike Infractructure Over Time in the City of Atlanta')

# #%% Gwinnett Miles

# df = arc_bike[arc_bike['county']=='Gwinnett'].to_crs(epsg=2240)
# df['length_mi'] = df.length / 5280
# df.groupby('facil')['length_mi'].sum()

# df['length_mi'].sum().round(1)
# df.plot()
