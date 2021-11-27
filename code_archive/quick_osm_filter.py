# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 22:22:26 2021

@author: tpassmore6
"""

import geopandas as gpd

#import osm data
gdf = gpd.read_file(r'C:/Users/tpassmore6/Documents/GitHub/BikewaySimDev/base_shapefiles/osm/coa_links_attr.gpkg')


#keep these
osm_filter_method = ['primary','primary_link','residential','secondary','secondary_link',
                    'tertiary','tertiary_link','trunk','trunk_link','cycleway','footway',
                    'path','pedestrian','steps', 'service']

osm_links = gdf[gdf["highway"].isin(osm_filter_method)]

#remove sidewalks
osm_links = osm_links[osm_links['footway'] != 'sidewalk']
osm_links = osm_links[osm_links['footway'] != 'crossing']

#export
osm_links.to_file(r'C:/Users/tpassmore6/Documents/GitHub/BikewaySimDev/base_shapefiles/osm/city_of_atlanta_osm_links.gpkg', driver = "GPKG")
