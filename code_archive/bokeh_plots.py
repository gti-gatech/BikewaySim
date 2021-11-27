# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 11:40:58 2021

@author: tpassmore6
"""

#%%import cell
import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.tile_providers import OSM, get_provider
from bokeh.models import GeoJSONDataSource, HoverTool

tile_provider = get_provider(OSM)
output_notebook()

#%% Convert to column data source for html maps
#note, none of the links can multipart geometries, must be all connected


def plot_bokeh_map(gdf, label_tuples, map_title):

    #convert to columndatasource
    #shapefile_source = htmlMap(gdf)

    #must be in this crs to line up with tiles
    gdf = gdf.to_crs('EPSG:3857')

    shapefile_source = GeoJSONDataSource(geojson=gdf.to_json())


    #HTML Road Map Plotting

    #initialize plot figure, arguements can change plot height/width, and title
    p = figure(title=f"{map_title}")
    
    #add in base map
    p.add_tile(tile_provider)
    
    #plot shapefile
    p.multi_line(xs='xs',ys='ys', source = shapefile_source, color = '#7308ff', line_width = 1, legend_label = "Roads")
    
    #save and display
    #save html map and show in browser
    
    p.add_tools(HoverTool(
        show_arrow=False,
        line_policy="nearest",
        #(display name, column name)
        tooltips=[
            *label_tuples#(f"{label}", f"@{label}")
        ]))

    return p
    
#use this if you want to toggle the displayed shapefile by clicking the legend
#p.legend.click_policy = "hide"

#save will save it as a file
#save(p,r"HTML_maps/all_map.html")

# def htmlMap(links):
#     #Convert Data to ColumnDataSource for HTML Maps
#     linksMap = links.copy()
    
#     #Get Line Coords function
#     def getLineCoords(row, geom, coord_type):
#         """Returns a list of coordinates ('x' or 'y') of a LineString geometry"""
#         if coord_type == 'x':
#             return list( row[geom].coords.xy[0] )
#         elif coord_type == 'y':
#             return list( row[geom].coords.xy[1] )
    
#     # Calculate x coordinates of the line
#     linksMap['x'] = linksMap.apply(getLineCoords, geom='geometry', coord_type='x', axis=1)
#     # Calculate y coordinates of the line
#     linksMap['y'] = linksMap.apply(getLineCoords, geom='geometry', coord_type='y', axis=1)
#     #make copy of geometry column
#     linksMap_df = linksMap.drop('geometry', axis=1).copy()
#     #column datasource
#     linksMap_source = ColumnDataSource(linksMap_df)
    
#     return linksMap_source