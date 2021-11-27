# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 17:49:52 2021

@author: tpassmore6
"""

import geopandas as gpd
import pandas as pd

#other networks
osmnxfp = r'Base_Shapefiles/osm/osm_links_osmnx.geojson'
osmnxnodes = r'Base_Shapefiles/osm/osm_nodes_osmnx.geojson'

#unsplit osm nodes
osm_unsplitfp = r'Base_Shapefiles/osm/osm_cleaning/osm_links_base.geojson'

studyareafp = r'Base_Shapefiles/bikewaysim_study_area/bikewaysim_study_area.shp'
studyarea_name = 'studyname'

#import with mask


#create nodes




def create_nodes_for_other_networks(fp, network_name, studyareafp, studyarea_name, link_type, A = 'A', B = 'B', layer = 0):
    
    #import links
    studyarea = gpd.read_file(studyareafp)
    df = gpd.read_file(fp, layer, mask = studyarea).to_crs(epsg=2240)
    df = node_ids.rename_nodes(df, network_name, link_type, A, B)
    df_nodes = node_ids.make_nodes(df, network_name, link_type, studyarea_name)
    
    #create other networks directory
    if not os.path.exists(r'Processed_Shapefiles/other_networks'):
        os.makedirs(r'Processed_Shapefiles/other_networks') 
    
    df.to_file(rf'Processed_Shapefiles/other_networks/{network_name}_{studyarea_name}_{link_type}.geojson', driver = 'GeoJSON')
    df_nodes.to_file(rf'Processed_Shapefiles/other_networks/{network_name}_{studyarea_name}_{link_type}.geojson', driver = 'GeoJSON')
    
    return

#abm two links
create_nodes_for_other_networks(abmfp, 'abm', studyareafp, studyarea_name, 'two_links', layer = 1)

#osm original
create_nodes_for_other_networks(osm_unsplitfp, 'osm', studyareafp, studyarea_name, 'orginal')


def summurize_network(summary_table, links, nodes, network_name, studyarea_name, link_type):
    
    #how many links
    num_links = len(links)
    
    #how many nodes
    num_nodes = len(nodes)
    
    #total mileage of links
    length_mi = links.geometry.length / 5280 # create a new distance column and calculate mileage of each link
    sum_miles = round(length_mi.sum(),0)
    
    #average link length
    avg_len = round(links.geometry.length.mean(),1)
    
    #add to summary table
    summary_table.loc[len(summary_table.index)] = [network_name, link_type, num_links, num_nodes, sum_miles, avg_len]
    
    #Print Stats
    print(f'There are {num_links} links, {num_nodes} nodes, {sum_miles} miles of links, and average link length of {avg_len} miles in {network_name}')
    
    return summary_table