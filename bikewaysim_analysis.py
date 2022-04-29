# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 09:54:09 2022

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
import folium
from folium.plugins import MarkerCluster
import glob

#make directory/pathing more intuitive later
user_directory = os.fspath(Path.home()) #get home directory and convert to path string
file_directory = r"\Documents\BikewaySimData\bikewaysim_outputs" #directory of bikewaysim network processing code

#change this to where you stored this folder
os.chdir(user_directory+file_directory)

#%% import data

#network files
network_dir = user_directory + r'\Documents\BikewaySimData\processed_shapefiles\prepared_network'
links = gpd.read_file(network_dir+r'\links\links.geojson').to_crs('epsg:4326')
nodes = gpd.read_file(network_dir+r'\nodes\nodes.geojson').to_crs('epsg:4326')

#ods with match to nearest node (figure out why it's duplicating later)
ods = pd.read_csv(r'samples_out\single_taz_to_all_node.csv').drop_duplicates()

#paths_bike
paths_bike = pd.read_csv(r'results\dist\paths_bike.csv')


#%% map trips function

def map_trips(nodes, links, ods, paths):
    
    #take in network, od pairs, and paths and output the route taken for each trip
    paths = paths_bike
    
    #add network geo data to paths
    paths_geo = pd.merge(paths, links, on=['A','B'], how='left')
    
    #add geo for ori/dest rows
    paths_geo.loc[paths_geo['A']=='origin',] = pd.merge(
        paths_geo.loc[paths_geo['A']=='origin',], ods, on=['trip_id']).drop_duplicates()

test = pd.merge(paths_geo.loc[paths_geo['A']=='origin',], ods, on=['trip_id'])


def map_trips(nodes, links, trip, origins, destinations):
    

    
    
    #add geo data to
    joined_results = pd.merge(trip, nodes, left_on='A', right_on='N', how='left')
    
    #drop no geo columns
    joined_results = joined_results[-(joined_results.geometry == None)]
    
    #take out trips with less than two points
    # test = joined_results.groupby(['trip_id'])['trip_id'].count()
    # joined_results = pd.merge(joined_results, test, left_on = 'trip_id', right_index=True, how = 'left', suffixes=(None,'_count'))
    # joined_results = joined_results[joined_results['trip_id_count'] > 1 ]
    
    #make sure it's in the correct order
    joined_results = joined_results.sort_values(by=['trip_id','sequence'], axis=0, ascending = True)

    # Zip the coordinates into a point object and convert to a GeoDataFrame
    #geometry = [Point(xy) for xy in zip(joined_results.X, joined_results.Y)]
    #gdf = gpd.GeoDataFrame(joined_results, geometry='geometry')

    # Aggregate these points with the GroupBy
    lines = joined_results.groupby(['trip_id'])['geometry'].apply(lambda x: LineString(x.to_list()))
    lines = gpd.GeoDataFrame(lines, geometry='geometry').set_crs("epsg:4326")
    
    #add length column
    #lines['length_mi'] = (lines.to_crs("epsg:2240").length / 5280).round(2)
    lines['length_mi'] = (trip.groupby(['tripid'])['dist'].sum()/5280).round(2)
    
    #add time impedance column
    time_impedance = trip.groupby(['trip_id'])['time'].sum()
    lines['impedance_min'] =  time_impedance.round(1)

    return lines



#%% bring in trip results



#analysze
#for single impedance routing
single_impedance = single_taz_to_all[single_taz_to_all['dest_id']=='630']






#%% clean origin file

# origins = pd.read_csv(r'lab_files/od_data/origins.csv')
# origins['ori_lat'] = origins.geometry.y
# origins['ori_lon'] = origins.geometry.x
# origins['origin_link'] = od.apply(lambda row: f'https://www.google.com/maps?z=12&t=m&q=loc:{row.ori_lat}+-{row.ori_lon*-1}', axis=1)
# origins.drop(columns=['geometry']).to_csv(r'lab_files/od_data/origins.csv',index=False)


#%% make grocery store and origin shapefile and destination map
#import
origins = pd.read_csv(r'lab_files/od_data/origins.csv')
grocery_stores = pd.read_csv(r'lab_files/od_data/grocery_stores.csv')

#create point geo from latlon
origins['geometry'] = list(zip(origins.ori_lon,origins.ori_lat))
origins['geometry'] = origins.apply(lambda row: Point(row.geometry),axis=1)

grocery_stores['geometry'] = list(zip(grocery_stores.dest_lon,grocery_stores.dest_lat))
grocery_stores['geometry'] = grocery_stores.apply(lambda row: Point(row.geometry),axis=1)

#make geodataframe
origins = gpd.GeoDataFrame(origins,geometry='geometry',crs="epsg:4326").to_crs("epsg:2240")
grocery_stores = gpd.GeoDataFrame(grocery_stores,geometry='geometry',crs="epsg:4326").to_crs("epsg:2240")

#export
grocery_stores.to_file(r'lab_files/od_data/grocery_stores.shp')
origins.to_file(r'lab_files/od_data/origins.shp')

#
grocery_stores['is_groc']
grocery_stores['']

#make html files
m = comb.explore(column ='is_groc' , categorical = True, cmap=["blue","red"],tooltip=['grocery_name','orig','dest'],tiles="CartoDB positron", marker_kwds=dict(radius=8))
#m2 = grocery_stores.explore(m=m)
m.save(r'lab_files/od_data/ods.html')

#%% import sidewalk links and nodes

sidewalk_nodes = gpd.read_file(r'lab_files/sidewalk_network/sidewalk_nodes.shp').to_crs('epsg:4326').rename(columns={'ID':'N'})[['N','geometry']]
sidewalk_links = gpd.read_file(r'lab_files/sidewalk_network/sidewalks.shp')
crosswalk_links = gpd.read_file(r'lab_files/sidewalk_network/crosswalks.shp')
sidewalks = sidewalk_links.append(crosswalk_links)

improved_sidewalks = gpd.read_file(r'lab_files/sidewalk_network/improved_sidewalks/SWS_links.shp')

#make HTML plot
#need to plot sidewalk layers + origins/destinations
sidewalks.rename(columns:{'sw_status':'Sidewalk Status'},inplace=True)



#%%
#Create map


# #%%

# def linestring_to_list(gdf):
#     all_x=[]
#     all_y=[]
#     all_coords=[]
#     for idx,row in gdf.iterrows():
#         x,y= row.geometry.coords.xy
#         all_x.append(x)
#         all_y.append(y)
#         list_coords = list(zip(y,x))
#         all_coords.append(list_coords)
#     gdf['list_coords'] = all_coords
#     return gdf



# m = folium.Map(location=[sidewalks.dissolve().geometry.centroid.y,sidewalks.dissolve().geometry.centroid.x], 
#                zoom_start=10, tiles="CartoDB positron")

# #Create a PolyLine FeatureGroup
# polylinegroup= folium.FeatureGroup(name='Sidewalks', control=True)
# ##Create a FeatureGroup of Marker
# pointsgroup = folium.FeatureGroup(name='Grocery_Stores', control=True)




# #Create a marker_cluster object as a child of pointsgroup:
# marker_cluster = MarkerCluster().add_to(pointsgroup)
# #Create Marker and add to marker_cluster
# for name,row in grocery_stores.iterrows():
#     folium.Marker([row["dest_lat"], row["dest_lon"]], tooltip=f"{row['grocery_name']}").add_to(marker_cluster) 
# #Use pointsgroup as the child of Map
# m.add_child(pointsgroup)



# sidewalks = linestring_to_list(sidewalks)

# #create a line
# line_cluster = MarkerCluster().add_to(polylinegroup)

# color_map = {
#     'Present':'#dadada',
#     'Absent':'#ff8989',
#     np.nan:'#dadada'
#     }

# for name,row in sidewalks.iterrows():
#     folium.PolyLine(row.list_coords,color=color_map[row.sw_status],popup=f"Sidwalk Status: {row['sw_status']}").add_to(polylinegroup)
    
# #Use pointsgroup as the child of Map
# m.add_child(polylinegroup)    
             
# #Open the LayerControl of the map
# folium.LayerControl().add_to(m)

# #Save Map to local
# m.save("lab_files/test.html")
    


#%%


# m1 = sidewalks.explore(column='sw_status', categorical = True, legend = True, tiles='CartoDB positron')

# #add nodes
# m2 = sidewalk_nodes.explore(m=m1, marker_kwds=dict(radius=2,color='gray'))

# m2.save(rf'lab_files/sidewalk_network/sidewalks.html')

# new_sidewalks = pd.merge(sidewalk_links,improved_sidewalks[['link_id','spd_improv']],on='link_id')

# improve_cond = (new_sidewalks.sw_status == 'Absent') & (new_sidewalks.spd_improv == 2.5)

# new_sidewalks.loc[improve_cond,'sw_status'] = 'New Present'

# m1 = new_sidewalks.explore(column='sw_status', categorical = True, legend = True, tiles='CartoDB positron')

# #add nodes
# m2 = sidewalk_nodes.explore(m=m1, marker_kwds=dict(radius=2,color='gray'))

# m2.save(rf'lab_files/sidewalk_network/new_sidewalks.html')



#%% run code

problem_nums = ["P1_Base","P2_Impedance","P2_Wheelchair","P3_improvement"]

sidewalk_nodes = gpd.read_file(r'lab_files/sidewalk_network/sidewalk_nodes.shp').to_crs('epsg:4326').rename(columns={'ID':'N'})[['N','geometry']]
sidewalk_nodes['N'] = sidewalk_nodes['N'].astype(str).drop_duplicates()



for problem_num in problem_nums: 
    
    trip = pd.read_csv(rf'lab_files/results/{problem_num}/results/paths_walk.csv')
    
    line = map_trips(sidewalk_nodes,trip)
    
    line = line.reset_index() 
    
    line.to_file(f'lab_files/results/{problem_num}/all_trips.shp')
    
    line.drop(columns=['geometry']).to_csv(f'lab_files/results/{problem_num}/all_trips.csv',index=False)

    
    for tripid in origins.orig.astype(str):
        
        #only select certain trips
        selected_lines = line[line['trip_id'].str.split('_',expand=True)[0] == tripid]
    
        if selected_lines.shape[0] > 0:
            #add lines
            m1 = selected_lines.explore(column='trip_id', categorical = True, legend = False, tiles='CartoDB positron')
            
            #get origin
            origin = origins[origins['orig'].astype(str)==tripid]
        
            pois = origin.append(grocery_stores).to_crs('epsg:4326')
            
            #add grocery stores
            m2 = pois.explore(column='type', m=m1, categorical=True, tiles='CartoDB positron', legend = False,
                              marker_kwds=dict(radius=8)
                              )
            
            m2.save(rf'lab_files/results/{problem_num}/html maps/tripid_{tripid}.html')
        
            selected_lines.to_file(f'lab_files/results/{problem_num}/shapefiles/tripid_{tripid}.shp')
        else:
            print(f'No geometry for tripid {tripid}')


def linestring_to_list(gdf):
    all_x=[]
    all_y=[]
    all_coords=[]
    for idx,row in gdf.iterrows():
        x,y= row.geometry.coords.xy
        all_x.append(x)
        all_y.append(y)
        list_coords = list(zip(y,x))
        all_coords.append(list_coords)
    gdf['list_coords'] = all_coords
    return gdf

gdf = gpd.read_file(r'C:/Users/tpassmore6/OneDrive - Georgia Institute of Technology/BikewaySim/Lab9/lab_files/results/P2_Impedance/shapefiles/tripid_0.shp')


def layer_map(points,trips):
    
    m = comb.explore(column ='is_groc' , categorical = True,
                     cmap=["blue","red"],tooltip=['grocery_name','orig','dest'],
                     tiles="CartoDB positron", marker_kwds=dict(radius=8)
                     )

    trips = linestring_to_list(trips)
    
    for trip in trips:
        #Create a PolyLine FeatureGroup
        polylinegroup= folium.FeatureGroup(name=trip['trip_id'], control=True)
        
        #create a line
        line_cluster = MarkerCluster().add_to(polylinegroup)

        for name,row in sidewalks.iterrows():
            folium.PolyLine(row.list_coords,color=color_map[row.sw_status],popup=f"Trip ID: {row['trip_id']}").add_to(polylinegroup)
            
        #Use pointsgroup as the child of Map
        m.add_child(polylinegroup)    
                     
        #Open the LayerControl of the map
        folium.LayerControl().add_to(m)

        #Save Map to local
        m.save("lab_files/test3.html")