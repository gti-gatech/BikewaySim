# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 11:12:14 2021

@author: tpassmore6
"""

import requests
import geopandas as gpd
import pandas as pd
import osmnx as ox

def download_osm(gdf,desired_osm_attributes=[]):
    
    #get geometry info from osmnx
    osmnx_nodes, osmnx_links = download_osmnx(gdf)
    
    #get additonal attribute information from overpass
    overpass_links = overpass_download(gdf)
    
    #retrieve only specific osm attributes like highway or oneway if given list
    if len(desired_osm_attributes) > 0:
        overpass_links = overpass_links[desired_osm_attributes]
    
    #merge osmnx and overpass data on link id
    osm_links = pd.merge(osmnx_links, overpass_links, left_on=('osmid'), right_on=('id'), how = 'inner')
    
    return osmnx_nodes, osm_links

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
    
    #plot it for fun
    ox.plot_graph(G)

    #convert to gdf
    nodes, links = ox.utils_graph.graph_to_gdfs(G)

    #reset index and name u and v to A and B
    links = links.reset_index().rename(columns={'u':'A','v':'B'})
    
    #simplify columns to geo and id
    links = links[['osmid','A','B','key','geometry']]

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
    
    #drop the nodes column because it's a list
    #df.drop(columns={'nodes'}, inplace = True)
    
    #find duplicate column names and drop them
    #might need to investigate duplicate columns in future to see if any data worth migrating over
    # colnames = list(df.columns)
    # colnames_lower = [x.lower() for x in colnames]
    # colnames_df = pd.DataFrame(data = {'name':colnames, 'name_lower':colnames_lower})
    # colnames_df = colnames_df[~colnames_df['name_lower'].duplicated()]['name']
    # df = df[colnames_df]
    
    return df 


# def preprocessed_version(osm_nodes,osm_links,buffer_tolerance):
    
#     osm_links = osm_links[['A','B','key','highway','footway','geometry','bicycle']]
    
#     #remove restricted access roads + sidewalks
#     restr_access = osm_links['highway'].isin(['motorway','motorway_link'])
#     osm_links = osm_links[-restr_access]
        
#     #remove sidewalks unless bikes explicitly allowed
#     remove_sidewalks = (osm_links['footway'].isin(['sidewalk','crossing'])) & (osm_links['bicycle'] != 'yes')
#     osm_links = osm_links[-remove_sidewalks]

#     #turn into graph
    
#     #get largest component of graph
    
#     G = ox.utils_graph.graph_from_gdfs(osm_nodes.set_index(),osm_links.set_index())
#     ox.utils_graph.get_largest_component(G)
#     #back to gdf

#     #remove disconnected links (this takes a while, just use QGIS instead)
#     osm_links['buffer'] = osm_links.buffer(10) #set buffer
#     osm_links_dissolved = osm_links.set_geometry('buffer').dissolve()#dissolve
#     osm_links_networks = osm_links_dissolved.explode().reset_index()#multi to single
#     biggest_network = osm_links_networks.loc[[osm_links_networks.area.idxmax()]]#find biggest connected network
#     biggest_network = gpd.GeoSeries(biggest_network.geometry) # turn into geoseries
#     #osm_links_sjoin = gpd.sjoin(osm_links, biggest_network, how='inner', predicate='touches')#spatial select
    
#     #find links that are covered by largest network polygon 
#     osm_links_sjoin = osm_links.within(biggest_network.geometry[0])

#     #using r trees
#     spatial_index = osm_links.sindex
#     possible_matches_index = list(spatial_index.intersection(biggest_network.bounds))

#     possible_matches = gdf.iloc[possible_matches_index]
#     precise_matches = possible_matches[possible_matches.intersects(polygon)]


#     #remove nodes that aren't in filtered links
#     nodes = osm_links['A'].append(osm_links['B']).unique()
#     osm_nodes = osm_nodes[osm_nodes['osmid'].isin(nodes)]
    
#     biggest_network.to_file('osm/test.geojson')
    
#     #create midpoints
#     osm_node_midpoints = osm_links.centroid
#     osm_node_midpoints_id = osm_links['osmid'].astype(str) + '_mid'
    
#     #add to nodes file
    


#%%

# #%% testing
# import os
# import geopandas as gpd
# import pandas as pd
# import pickle

# directory = r"C:/Users/tpassmore6/Documents/BikewaySimData/base_shapefiles"
# os.chdir(directory)
    
# #give filepath of a shapefile/geojson/geopackage
# #studyareafp = r'C:\Users\tpassmore6\Documents\BikewaySimData\gps_traces_for_matching\sample_trace.geojson'
# studyareafp = r'bikewaysim_study_area/bikewaysim_study_area.shp'
# #studyareafp = r'coa/Atlanta_City_Limits.shp'
# #studyareafp = r'arc/arc_bounds.shp'

# #give study area a name for exporting
# #studyareaname = 'bikewaysim'
# #studyareaname = 'coa'
# #studyareaname = 'arc'

# #read in study area polygon and convert to WGS 84
# gdf = gpd.read_file(studyareafp).to_crs(epsg=4326)
# gdf.plot()

# osm_nodes, osm_links = download_osm(gdf)

# #reproject to desired CRS
# osm_nodes.to_crs('epsg:2240',inplace=True)
# osm_links.to_crs('epsg:2240',inplace=True)


# #%%
# for col in osm_links.columns.tolist():
#     if list in [type(x) for x in osm_links.loc[:,col]]:
#         osm_links.drop(columns=col,inplace=True)
#         print(f"{col} column removed...")



# #%%

# raw_osm = osm_links

# osm_links = raw_osm


# osm_links_filt = gpd.read_file('osm/for_lime/osm_links_lime_032222.geojson')



# #export
# cols = ['A','B','osmid','highway','geometry']
# osm_nodes_filt.to_file('osm/for_lime/osm_nodes_lime.geojson',driver='GeoJSON')
# osm_links_filt[cols].to_file('osm/for_lime/osm_links_lime.geojson',driver='GeoJSON')
# osm_nodes_filt.to_file('osm/for_lime/osm_nodes_lime.shp')
# osm_links_filt[cols].to_file('osm/for_lime/osm_links_lime.shp')

# osm_links_filt = gpd.read_file('osm/for_lime/osm_links_lime_032222.geojson')

# #remove nodes that aren't in filtered links
# nodes = osm_links_filt['A'].append(osm_links_filt['B']).unique()
# osm_nodes_filt = osm_nodes[osm_nodes['osmid'].isin(nodes)]

# osm_nodes_filt.to_file('osm/for_lime/osm_nodes_lime_032222.geojson',driver='GeoJSON')
# osm_nodes_filt.to_file('osm/for_lime/osm_nodes_lime_032222.shp')
# osm_links_filt.to_file('osm/for_lime/osm_links_lime_032222.shp')


        




#%% bad columns



