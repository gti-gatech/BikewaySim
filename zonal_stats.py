# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 16:10:11 2022

@author: tpassmore6
"""

import pandas as pd
import geopandas as gpd
import os
from pathlib import Path

#make directory/pathing more intuitive later
user_directory = os.fspath(Path.home()) #get home directory and convert to path string
file_directory = r"/Documents/BikewaySimData" #directory of bikewaysim outputs
os.chdir(user_directory+file_directory)


#import trip lines
trip_lines = gpd.read_file(r'trb2023/trip_lines.gpkg',layer='dist',driver='GPKG')