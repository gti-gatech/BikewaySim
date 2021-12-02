# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 11:12:14 2021

@author: tpassmore6
"""

import requests
import geopandas as gpd
import pandas as pd
import osmnx as ox


def download_osmnx(studyarea):

    #convert to unprojected if already projected
    if studyarea.crs != 'EPSG:4326':
        studyarea = studyarea.to_crs('EPSG:4326')
    

    #retrieve graph from the chosen study polygon
    G = ox.graph_from_polygon(studyarea.geometry[0], network_type='all_private', simplify= False)
    
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
    links = links[['osmid','A','B','geometry']]

    return nodes, links

def overpass_download(studyarea):
        
    #convert to unprojected if already projected
    if studyarea.crs != 'EPSG:4326':
        studyarea = studyarea.to_crs('EPSG:4326')
    
    #get bounds and print to see if reasonable
    miny, minx, maxy, maxx = studyarea.total_bounds
    print(f'The bounding box is {minx}, {miny}, {maxx}, {maxy}')
        
    query = f"""
    [out:json]
    [timeout:120]
    ;
    (
      way
        ["highway"]
        ({minx},{miny},{maxx},{maxy});
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
    df.drop(columns={'nodes'}, inplace = True)
    
    #find duplicate column names and drop them
    #might need to investigate duplicate columns in future to see if any data worth migrating over
    colnames = list(df.columns)
    colnames_lower = [x.lower() for x in colnames]
    colnames_df = pd.DataFrame(data = {'name':colnames, 'name_lower':colnames_lower})
    colnames_df = colnames_df[~colnames_df['name_lower'].duplicated()]['name']
    df = df[colnames_df]
    
    return df 
      
#%% Test running


# import os 

#directory of osm data
# file_directory = r"C:/Users/tpassmore6/Documents/GitHub/BikewaySimDev/base_shapefiles/osm"

# #cahnge directory
# os.chdir(file_directory)

# studyareafp = r'C:/Users/tpassmore6/Documents/GitHub/BikewaySimDev/base_shapefiles/bikewaysim_study_area/bikewaysim_study_area.shp'
# #studyareafp = r'C:/Users/tpassmore6/Documents/GitHub/BikewaySimDev/base_shapefiles/arc/arc_bounds.shp'#
# #studyareafp = r'C:/Users/tpassmore6/Documents/GitHub/BikewaySimDev/base_shapefiles/coa/Atlanta_City_Limits.shp'

# gdf = gpd.read_file(studyareafp).to_crs(epsg=4326)
# gdf.plot()

# osmnx_nodes, osmnx_links = download_osmnx(gdf)
# overpass_links = overpass_download(gdf)

# #test merge
# complete_dataset = pd.merge(osmnx_links, overpass_links, left_on=('osmid'), right_on=('id'), how = 'inner')

# #project to desired CRS
# osmnx_links.to_crs(epsg='2240', inplace = True)
# osmnx_nodes.to_crs(epsg='2240', inplace = True)
# complete_dataset.to_crs(epsg='2240', inplace = True)



# #export simplified data
# osmnx_links.to_file('studyarea_links.gpkg', driver = 'GPKG')
# osmnx_nodes.to_file('studyarea_nodes.gpkg', driver = 'GPKG')

# #export merged data
# complete_dataset.to_file('studyarea_links_attr.gpkg', driver = 'GPKG')
# complete_dataset.to_csv('studyarea_links_attr.csv')

#project and drop in NA rows



#detect lists
#test = complete_dataset.columns[complete_dataset.applymap(type).eq(list).any()]


# query used for nodes
    # # OSM Nodes
    # # Get OSM JSON from Overpass API
    # #Query generated froom Overpass Turbo
    # query = f"""
    # [out:json]
    # [timeout:120]
    # ;
    # way
    #     ["highway"]
    #     ({minx},{miny},{maxx},{maxy});
    # node(w);
    # out;
    # >;
    # out skel qt;
    # """
    
    # url = "http://overpass-api.de/api/interpreter"
    # r = requests.get(url, params={'data': query})
    
    # result = r.json()
    # result = osm2geojson.json2shapes(result)
    
    # #write to GeoJSON
    # features = []
    # for i in range(0,len(result)-1):
    #     features.append(Feature(geometry = result[i]['shape'], properties= result[i]['properties']))
        
    # feature_collection = FeatureCollection(features)
    
    # with open(rf'base_shapefiles/osm/osm_nodes_{name}_raw.geojson', 'w') as f:
    #    dump(feature_collection, f)
    
    # OSM Links
    # Get OSM JSON from Overpass API
## Query generated froom Overpass Turbo#