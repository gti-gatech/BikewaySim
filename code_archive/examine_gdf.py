# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 12:25:45 2021

@author: tpassmore6
"""

import geopandas as gpd
import bokeh_plots

gdf = gpd.read_file(r'C:/Users/tpassmore6/Documents/ridership_data/CycleAtlanta/Aditi Cycle Atlanta data/Rd_Bike_WGS1984_Integrate5ft_Planarize.gdb')

bokeh_plots.plot_bokeh_map(gdf, 'ROUTE_TYPE', 'Aditi Final Map')


# get values
df = gdf.drop(columns={'geometry'})
val_counts = df.value_counts()

def create_attribute_definitions(gdf, network_name):
        
    #only import the attributes, but leave geometry data out
    gdf = gpd.read_file(networkfp, ignore_geometry = True, bbox=(0,0,0,0))
    
    #data types
    data_types = gdf.dtypes
    
    #number of null values
    #null_values = links.notnull()
    
    sumdata = pd.DataFrame({'data_types': data_types})
    
    #attribute names, number of null values for each attribute, and data type to CSV for data dictionary
    sumdata.to_csv(rf"Processed_Shapefiles/{network_name}/{network_name}.csv")
    
    return gdf, sumdata
    
    
#%% Run

gdf, sumdata = create_attribute_definitions(osmfp, 'osm')