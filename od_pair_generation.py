# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 14:41:04 2021

@author: tpassmore6
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
from pathlib import Path
import itertools

def all_pairings_two(origins,ori_id,dests,dest_id):
    
    
    combs = list(itertools.combinations(origins,dests)





def all_pairs(settings):

    studyarea_name = settings['studyarea_name']
    studyarea_fp = settings['studyarea_fp']
    tazs_fp = settings['tazs_fp']

    #import
    studyarea = gpd.read_file(studyarea_fp)
    tazs = gpd.read_file(studyarea_fp / Path(f'{studyarea_name}/base_layers.gpkg'),mask=studyarea,layer='studyarea')[[settings['tazs_id_col'],'geometry']]]
    tazs.to_crs(studyarea.crs,inplace=True)

    #make sure ids are columns
    tazs['FID_1'] = tazs['FID_1'].astype(str)

    #export
    tazs.to_file(studyarea_fp,layer='tazs_polygons')

    #get centroids
    tazs.geometry = tazs.centroid

    #export again
    tazs.to_file(studyarea_fp,layer='tazs_centroids')

    #get lat and lon columns
    tazs.to_crs('epsg:4326',inplace=True)
    tazs['lat'] = tazs.geometry.y
    tazs['lon'] = tazs.geometry.x

    #get permuations of taz ids
    perms = list(itertools.permutations(tazs[settings['tazs_id_col']],2))

    #make df
    od_pairs = pd.DataFrame(columns=['trip_id','ori_id','dest_id','ori_lat','ori_lon','dest_lat','dest_lon'])

    od_pairs['perms'] = perms
    #seperate into two columns

    od_pairs = pd.merge(od_pairs,tazs,left_on='ori_id',right_on=settings['tazs_id_col'])
    od_pairs.rename(columns:{'lat':'ori_lat','lon':'ori_lon'},inplace=True)

    od_pairs = pd.merge(od_pairs,tazs,left_on='dest_id',right_on=settings['tazs_id_col'])
    od_pairs.rename(columns:{'lat':'dest_lat','lon':'dest_lon'},inplace=True)

    od_pairs['trip_id'] = od_pairs['ori_id'] + '_' + od_pairs['dest_id']

    #export
    od_pairs.to_csv(studyarea_fp / Path(f'{studyarea_name}/od_pairs.csv'), index = False)

#TODO have other method?