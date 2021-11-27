# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 22:14:15 2021

@author: - Reid Passmore
"""
import requests
import geopandas as gpd
import pandas as pd

def osm_to_geojson(studyarea):
        
    #convert to unprojected if already projected
    if studyarea.crs != 'EPSG:4326':
        studyarea = studyarea.to_crs('EPSG:4326')
    
    #get bounds and print to see if reasonable
    miny, minx, maxy, maxx = studyarea.total_bounds
    print(f'The bounding box is {miny}, {minx}, {maxy}, {maxx}')
        
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
    
    return df 
      
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
    #Query generated froom Overpass Turbo