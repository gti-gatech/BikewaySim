
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import wkt
from shapely.wkt import dumps
from itertools import compress
from shapely.ops import LineString, Point, MultiPoint

from pathlib import Path
import geopandas as gpd

def add_attributes(base_links, join_links, join_name, buffer_ft):
    
    #give base_links a temp column so each row has unique identifier
    # A_B doesn't always work because there might be duplicates from the split links step
    base_links['temp_ID'] = np.arange(base_links.shape[0]).astype(str)
    
    #make a copy for intersecting
    intersect = base_links.copy()

    #calculate original base_links link length
    intersect['length'] = intersect.length
    
    #create copy of join links to use for bufferring
    buffered_links = join_links.copy()
    
    #buffer the join links by tolerance_ft
    buffered_links.geometry = buffered_links.buffer(buffer_ft)
    
    #drop unique columns
    buffered_links.drop(columns=[f'{join_name}_A',f'{join_name}_B',f'{join_name}_A_B'],inplace=True)

    #dissolve by attributes in join links
    cols = [x for x in buffered_links.columns if x != 'geometry']
    dissolved_links = buffered_links.dissolve(cols)
    
    #intersect join buffer and base links
    overlapping_links = gpd.overlay(base_links, dissolved_links, how='intersection')
    
    #re-calculate the link length to see which join buffers had the greatest overlap of base links
    overlapping_links['percent_overlap'] = (overlapping_links.length / overlapping_links['length'] )
    
    #select matches with greatest link overlap
    max_overlap_idx = overlapping_links.groupby('temp_ID')['percent_overlap'].idxmax().to_list()
    
    #get matching links from overlapping links
    match_links = overlapping_links.loc[max_overlap_idx,:]
    
    #join back to base links
    base_links = pd.merge(base_links,match_links.drop('geometry'),on='temp_ID')

    #drop added columns
    base_links.drop(columns=['temp_ID','length'],inplace=True)
    
    return base_links, buffered_links, dissolved_links, match_links, overlapping_links

def add_osm_attr(liknks,fp):

    #bring in attribute data
    attr = pd.read_pickle(fp / Path('osm.pkl'))

    #attach attribute data to filtered links
    links = pd.merge(links,attr,on='osm_A_B')

    #speed limit

    #change nones to NaN
    links['maxspeed'] = pd.to_numeric(links['maxspeed'],errors='coerce').fillna(links['maxspeed'])

    #covert to ints
    func = lambda x: int(str.split(x,' ')[0]) if type(x) == str else (np.nan if x == None else int(x))
    links['maxspeed'] = links['maxspeed'].apply(func)

    #speed categories
    bins = [0,25,30,70]
    names = ['< 25','25-30','> 30']

    #replace
    links['maxspeed'] = pd.cut(links['maxspeed'],bins=bins,labels=names)
    links['maxspeed'] = links['maxspeed'].astype(str)

    # number of lanes

    #convert none to nan
    links['lanes'] = links['lanes'].apply(lambda x: np.nan if x == None else x)

    #make sure numeric
    links['lanes'] = pd.to_numeric(links['lanes'])

    #speed cats
    bins = [0,3,6,links.lanes.max()]
    names = ['one lane', 'two or three lanes', 'four or more']

    #replace
    links['lanes'] = pd.cut(links['lanes'],bins=bins,labels=names)
    links['lanes'] = links['lanes'].astype(str)

    # road directionality
    links.loc[(links['oneway']=='-1') | (links['oneway'] is None),'oneway'] = 'NA'

    #%% bike facilities

    #get bike specific columns
    bike_columns = [x for x in links.columns.to_list() if (('cycle' in x) | ('bike' in x)) & ('motorcycle' not in x)]
    foot_columns = [x for x in links.columns.to_list() if ('foot' in x)]

    #add lit
    bike_columns = bike_columns + foot_columns + ['lit']

    #anything that contains lane is a bike lane
    links.loc[(links[bike_columns] == 'lane').any(axis=1),'bikefacil'] = 'bike lane'

    #multi use paths or protected bike lanes (i don't think there's a way to tell in OSM)
    mups = ['path','footway','pedestrian','cycleway']
    links.loc[links['highway'].isin(mups),'bikefacil'] = 'mup or pbl'

    #drop excess columns
    links.drop(columns=bike_columns,inplace=True)

    # parking
    parking_columns = [x for x in links.columns.to_list() if 'parking' in x]
    parking_vals = ['parallel','marked','diagonal','on_street']
    links.loc[(links[parking_columns].isin(parking_vals)).any(axis=1),'parking_pres'] = 'yes'
    links.loc[(links[parking_columns]=='no').any(axis=1),'parking_pres'] = 'yes'
    links.drop(columns=parking_columns,inplace=True)


    # sidewalk presence
    sidewalks = [x for x in links.sidewalk.unique().tolist() if x not in [None,'no','none']]
    links['sidewalk_pres'] = 'NA'
    links.loc[links['sidewalk'].isin(sidewalks),'sidewalk_pres'] = 'yes'
    links.loc[links['sidewalk'].isin([None,'no','none']),'sidewalk_pres'] = 'no'
    links.drop(columns=['sidewalk'],inplace=True)

    return links