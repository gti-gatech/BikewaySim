# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 12:16:13 2022

@author: tpassmore6
"""
#import modules
import conflation_tools
import pandas as pd
import geopandas as gpd
import os

directory = r"C:/Users/tpassmore6/Documents/BikewaySimData"
os.chdir(directory)

#import marta links
marta_links = gpd.read_file(r'base_shapefiles/marta/for_Reid_May19/links_shape.shp')

#split link id into a and b


#preprocess marta liks
marta_links['A'] = marta_links['link_id'].str.split('_')[0]
marta_links['B'] = marta_links['link_id'].str.split('_')[1]

#import marta nodes
marta_nodes = gpd.read_file(r'base_shapefiles/marta/for_Reid_May19/stops.shp')

#import bikewaysim nodes



#nearest point matching


# Match Points Function



#export match lines

#export marta nodes and links with matching