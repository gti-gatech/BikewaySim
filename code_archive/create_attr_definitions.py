#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 10:44:32 2021

@author: tannerpassmore
"""

import os
import pandas as pd
import geopandas as gpd
from pathlib import Path
import time

tot_time_start = time.time()

#make directory/pathing more intuitive later
user_directory = os.fspath(Path.home()) #get home directory and convert to path string
file_directory = "/Documents/GitHub/BikewaySim_Network_Processing" #directory of bikewaysim network processing code

#change this to where you stored this folder
os.chdir(user_directory+file_directory)


#%% Filepaths


abmfp = r'Base_Shapefiles/arc/ABM2020-TIP20-2020-150kShapefiles-outputs.gdb'
navstreetsfp = r'Base_Shapefiles/navstreets/Streets.shp'
osmfp = r'Base_Shapefiles/osm/osm_cleaning/final.geojson'


#%% Create attribute definitons function

