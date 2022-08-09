# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 22:02:05 2022

@author: tpassmore6
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import os
from pathlib import Path

#make directory/pathing more intuitive later
user_directory = os.fspath(Path.home()) #get home directory and convert to path string
file_directory = r"/Documents/BikewaySimData" #directory of bikewaysim outputs
os.chdir(user_directory+file_directory)

#%%


#speed
abm_data['SPEEDLIMIT']

conditions = [
    (abm_data['SPEEDLIMIT'] > 80),
    (abm_data['SPEEDLIMIT'] >= 65) & (abm_data['SPEEDLIMIT'] < 80),
    (abm_data['SPEEDLIMIT'] >= 55) & (abm_data['SPEEDLIMIT'] < 64),
    (abm_data['SPEEDLIMIT'] >= 41) & (abm_data['SPEEDLIMIT'] < 54),
    (abm_data['SPEEDLIMIT'] >= 31) & (abm_data['SPEEDLIMIT'] < 40),
    (abm_data['SPEEDLIMIT'] >= 21) & (abm_data['SPEEDLIMIT'] < 30),
    (abm_data['SPEEDLIMIT'] >= 6) & (abm_data['SPEEDLIMIT'] < 20),
    (abm_data['SPEEDLIMIT'] < 6)
]
values = ['> 80 MPH', '65-80 MPH', '55-64 MPH', '41-54 MPH', '31-40 MPH', '21-30 MPH', '6-20 MPH', '< 6 MPH']

abm_data['SPEEDLIMIT'] = np.select(conditions, values)

#oneway
abm_data['oneway'] = abm_data['two_way'] == False