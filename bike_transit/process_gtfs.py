# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 11:06:30 2022

@author: tpassmore6
"""

import networkx as nx
import pandas as pd
import geopandas as gpd
import time
from pathlib import Path
import subprocess
import os

def create_transfers(settings):
    '''
    Creates "transfers.txt" file neccessary for RAPTOR routing. Uses a 2.5 MPH walking speed and straight-line,
    Euclidean distance to determine tranfer times
    '''
    
    #record the starting time
    time_start = time.time()
    
    #import transit stops txt and turn into geodataframe
    stops = pd.read_csv(settings['gtfs_fp'] / Path('stops.txt'))
    stops = gpd.GeoDataFrame(stops, geometry=gpd.points_from_xy(stops['stop_lon'], stops['stop_lat']), crs='epsg:4326').drop(columns=['stop_lat','stop_lon'])
    stops.to_crs(settings['crs'],inplace=True)
    
    #initialize dataframe for transfer links
    transfer_links = pd.DataFrame(columns=['from_stop_id','to_stop_id','length'])
    
    # set the max transfer threshold in minutes (440ft)
    walk_thresh = settings['transfer_time'] / 60 * settings['walk_spd'] * 5280
    
    #from each stop, calculate the euclidean (straight-line) distance to other stops
    for idx, row in stops.iterrows():
        
        #create empty df
        transfer_link = pd.DataFrame(columns=['from_stop_id','to_stop_id','length'])
        
        #make from stop id column
        transfer_link['from_stop_id'] = stops['stop_id']
        
        #make to stop id column
        transfer_link['to_stop_id'] = row['stop_id']
        
        #calculate distance to all transit stops from centroid
        transfer_link['length'] = stops.distance(row.geometry)
        
        #remove stops outside walking distance
        transfer_link = transfer_link[transfer_link['length'] < walk_thresh]
        
        #append
        transfer_links = transfer_links.append(transfer_link)
        
    #remove self matches
    transfer_links = transfer_links[-(transfer_links['from_stop_id']==transfer_links['to_stop_id'])]   
    
    #convert back to minutes as that's the format needed in transit-routing
    transfer_links['min_transfer_time'] = transfer_links['length'] / 5280 / settings['walk_spd'] * 60
    
    #create reverse links just in case?
    transfer_links_rev = pd.DataFrame({'from_stop_id':transfer_links['to_stop_id'],'to_stop_id':transfer_links['from_stop_id'],'length':transfer_links['length']})
    transfer_links = pd.concat([transfer_links,transfer_links_rev])
    
    #create network graph
    G = nx.DiGraph()  # create un-directed graph
    for ind, row in transfer_links.iterrows():
        G.add_weighted_edges_from([(row['from_stop_id'], row['to_stop_id'], float(row['min_transfer_time']))],weight='min_transfer_time')
 
    #get transitive closure of graph
    G_final = nx.transitive_closure(G,reflexive=None)
    
    #get edge list
    transfers = nx.to_pandas_edgelist(G_final)
    
    #rename columns (min_transfer_time won't be accurate)
    transfers.columns = ['from_stop_id','to_stop_id','min_transfer_time']
    
    #add point geo
    transfers = pd.merge(transfers,stops,left_on='from_stop_id',right_on='stop_id').merge(stops,left_on='to_stop_id',right_on='stop_id')
    
    #turn to gdf
    transfers = gpd.GeoDataFrame(transfers,geometry='geometry_x')
    
    #calculate euclidean distance
    transfers['distance_ft'] = transfers.distance(transfers['geometry_y'])
    
    #convert to time (i thought it was minutes but apprently it's in seconds)
    transfers['min_transfer_time'] = transfers['distance_ft'] / 5280 / settings['walk_spd'] * 60 * 60
    
    #clean up
    transfers = transfers[['from_stop_id','to_stop_id','min_transfer_time']]
    
    #export
    transfers.to_csv(settings['gtfs_fp'] / Path('transfers.txt'),index=False)

    print(f'took {round(((time.time() - time_start)/60), 2)} minutes to create transfers.txt')

def process_gtfs(kwds:dict):

    prev_dir = os.getcwd()
    os.chdir(Path.cwd() / 'transit-routing')

    modes = ''
    for mode in kwds['modes']:
        modes = modes + str(mode) + '\n'
    modes = modes + '-1'

    #feed in inputs
    #input_values = "marta\n20230301\n1\n3\n-1\n0\n0\n0\n0\n"
    input_values = f"{kwds['gtfs_name']}\n{kwds['service_date']}\n{modes}\n0\n0\n0\n0\n"

    # Run the script using subprocess
    process = subprocess.Popen(['python', 'GTFS_wrapper.py'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    output, _ = process.communicate(input=input_values)

    # change back to prev directory
    os.chdir(prev_dir)

    # Print the captured output
    print(output)
    return output

# =============================================================================
# 
# #use this code to run if testing
# #this is just a settings dict that I use for all the other functions i have. i commented out anything that wasn't used for this function
# settings = {
#     #these are for the pre-processing steps
#     'gtfs_fp': Path.home() / Path('Documents/GitHub/transit-routing/GTFS/martalatest'), #filepath for processed GTFS files
#     'crs': 'epsg:2240', # the desired projected CRS to use
#     'transfer_time': 2, # allowed transfer time in minutes (DOES NOT INLCUDE WAIT TIME)
#     'walk_spd': 2.5 # assumed walking speed for transfers in miles per hour
#     }
# 
# create_transfers(settings)
# =============================================================================
