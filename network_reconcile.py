
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import wkt
from shapely.wkt import dumps
from itertools import compress
from shapely.ops import LineString, Point, MultiPoint

from pathlib import Path
import geopandas as gpd

def add_attributes(base_links, join_links, join_name, buffer_ft, overlap, dissolve=True):
    
    #give base_links a temp column so each row has unique identifier
    # A_B doesn't always work because there might be duplicates from the split links step
    base_links['temp_ID'] = np.arange(base_links.shape[0]).astype(str)
    
    #make a copy for intersecting
    intersect = base_links.copy()

    #make intersect only tempid and geo
    intersect = intersect[['temp_ID','geometry']]

    #calculate original base_links link length
    intersect['length'] = intersect.length
    
    #create copy of join links to use for bufferring
    buffered_links = join_links.copy()
    
    #buffer the join links by tolerance_ft
    buffered_links.geometry = buffered_links.buffer(buffer_ft)
    
    if dissolve:
        #drop unique columns (no longer valid)
        buffered_links.drop(columns=[f'{join_name}_A',f'{join_name}_B',f'{join_name}_A_B'],inplace=True)

        #dissolve by attributes in join links
        cols = buffered_links.columns.to_list()
        cols.remove('geometry')
        print(f'Dissolving by {len(cols)} columns')
        dissolved_links = buffered_links.dissolve(cols).reset_index()
    else:
        dissolved_links = buffered_links
        
    #intersect join buffer and base links
    overlapping_links = gpd.overlay(intersect, dissolved_links, how='intersection')
    
    #re-calculate the link length to see which join buffers had the greatest overlap of base links
    overlapping_links['percent_overlap'] = (overlapping_links.length / overlapping_links['length'] )
    
    #select matches with greatest link overlap
    max_overlap_idx = overlapping_links.groupby('temp_ID')['percent_overlap'].idxmax().to_list()
    
    #get matching links from overlapping links
    match_links = overlapping_links.loc[max_overlap_idx,:]
    
    #drop matches below the overlap threshhold
    match_links = match_links[match_links['percent_overlap'] > overlap]

    #join back to base links
    base_links = pd.merge(base_links,match_links.drop(columns=['geometry']),on='temp_ID',how='left')

    #drop added columns
    base_links.drop(columns=['temp_ID','length'],inplace=True)
    base_links.rename(columns={'percent_overlap':f'{join_name}_pct_overlap'},inplace=True)
    
    return base_links

def add_osm_attr(links,attr_fp):

    #bring in attribute data
    attr = pd.read_pickle(attr_fp)

    #attach attribute data to filtered links
    links = pd.merge(links,attr,on=['osm_A','osm_B','osm_A_B'])

    #speed limit

    #change nones to NaN
    links['maxspeed'] = pd.to_numeric(links['maxspeed'],errors='coerce').fillna(links['maxspeed'])

    #covert to ints
    func = lambda x: int(str.split(x,' ')[0]) if type(x) == str else (np.nan if x == None else int(x))
    links['maxspeed'] = links['maxspeed'].apply(func)

    #speed categories
    bins = [0,24,30,999]
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
    bins = [0,3,6,999]
    names = ['one lane', 'two or three lanes', 'four or more']

    #replace
    links['lanes'] = pd.cut(links['lanes'],bins=bins,labels=names)
    links['lanes'] = links['lanes'].astype(str)

    # road directionality
    links.loc[(links['oneway']=='-1') | (links['oneway'] is None),'oneway'] = 'NA'

    # bike facilities

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

def add_here_attr(links,attr_fp):

    #revelant attributes
    here_attr = ['here_A_B','ST_NAME','SPEED_CAT','LANE_CAT','DIR_TRAVEL','FUNC_CLASS']

    #bring in attribute data
    attr = pd.read_pickle(attr_fp)[here_attr]

    #attach attribute data to filtered links
    links = pd.merge(links,attr,on='here_A_B')

    # speed categories
    here_speed_bins = {
        '1': '> 30', # '> 80 MPH',
        '2': '> 30', # '65-80 MPH',
        '3': '> 30', # '55-64 MPH',
        '4': '> 30', # '41-54 MPH',
        '5': '> 30', # '31-40 MPH',
        '6': '25-30', # '21-30 MPH',
        '7': '< 25', # '6-20 MPH',
        '8': '< 25' # '< 6 MPH'
        }
    links['SPEED_CAT'] = links['SPEED_CAT'].map(here_speed_bins)

    # number of lanes
    here_lane_bins = {
        '1': '1', # 'one lane',
        '2': '2-3', # 'two or three lanes',
        '3': '> 4' # 'four or more'
        }
    links['LANE_CAT'] = links['LANE_CAT'].map(here_lane_bins)

    # road directionality
    here_oneway_bins = {
        'B':'both', # Both Directions
        'F':'oneway', # From Reference Node
        'T':'wrongway', # To Reference Node
        'N': 'NA' # Closed in both directions
        }
    links['DIR_TRAVEL'] = links['DIR_TRAVEL'].map(here_oneway_bins)

    # here functional class (does not correspond to FHWA or HFCS)
    func_class = {
        '1':'highways',
        '2':'major arterials',
        '3':'collectors/minor arterials',
        '4':'collectors/minor atrerials',
        '5':'local'
        }
    links['FUNC_CLASS'] = links['FUNC_CLASS'].map(func_class)

    return links

def add_abm_attr(links,attr_fp):

    #bring in attribute data
    attr = pd.read_pickle(attr_fp)

    #attach attribute data to filtered links
    links = pd.merge(links,attr,on=['here_A','here_B','here_A_B'])
    
    #speed
    links['SPEEDLIMIT']

    conditions = [
        (links['SPEEDLIMIT'] > 80),
        (links['SPEEDLIMIT'] >= 65) & (links['SPEEDLIMIT'] < 80),
        (links['SPEEDLIMIT'] >= 55) & (links['SPEEDLIMIT'] < 64),
        (links['SPEEDLIMIT'] >= 41) & (links['SPEEDLIMIT'] < 54),
        (links['SPEEDLIMIT'] >= 31) & (links['SPEEDLIMIT'] < 40),
        (links['SPEEDLIMIT'] >= 21) & (links['SPEEDLIMIT'] < 30),
        (links['SPEEDLIMIT'] >= 6) & (links['SPEEDLIMIT'] < 20),
        (links['SPEEDLIMIT'] < 6)
    ]
    values = ['> 80 MPH', '65-80 MPH', '55-64 MPH', '41-54 MPH', '31-40 MPH', '21-30 MPH', '6-20 MPH', '< 6 MPH']

    links['SPEEDLIMIT'] = np.select(conditions, values)

    #oneway
    #links['oneway'] = links['two_way'] == False
    links = links[['NAME','SPEEDLIMIT']]

    return links

def add_arc_bike(links):

    links = links[['facil','geometry']]

    links['pbl']=0
    links.loc[links['facil']=='Protected Bike Lane','pbl'] = 1

    links['bl']=0
    links.loc[links['facil']=='Bike Lane','bl'] = 1

    links['mu']=0
    links.loc[(links['facil']=='Multi-Use Path'),'mu'] = 1

    links.drop(columns=['facil'],inplace=True)

    return links
