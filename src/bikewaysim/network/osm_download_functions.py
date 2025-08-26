# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 11:12:14 2021

@author: tpassmore6
"""

import requests
import geopandas as gpd
import pandas as pd
import osmnx as ox
from tqdm import tqdm
import numpy as np
from bs4 import BeautifulSoup
from shapely.ops import LineString, Point
from bikewaysim.paths import config

def download_geofabrik(states,year,export_fp):
    '''
    Downloads extracts of U.S. states from geofabrik. Multiple states can be downloaded at
    a time if state is a list. Year should be in YY format and greater than 13. If the most
    the current data is wanted, then enter 'current' for year.
    '''
    states = states.split(',')

    if isinstance(year,int):
        year = str(year)

    for state in states:
        print(f"Retrieving {year} extract for {state}")
        url = f"http://download.geofabrik.de/north-america/us/{state}-{year}0101.osm.pbf"
        if year == 'current':
            url = f"http://download.geofabrik.de/north-america/us/{state}-latest.osm.pbf"
            date_url = f"https://download.geofabrik.de/north-america/us/{state}.html"
            response = requests.get(date_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            current_extract_date = url.split('/')[-1].split('-latest.osm.pbf')[0] + \
                '-current-' + \
                    soup.li.text.split('ago and contains all OSM data up to ')[1].split('.')[0] + \
                        ".osm.pbf"
            current_extract_date = current_extract_date.replace(':','-')
            #TODO instead of adding the date to the filepath, create a text file that has the date
            export_fp0 = export_fp / current_extract_date
        else:
            export_fp0 = export_fp / url.split('/')[-1]
        print(url)
        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # Check for HTTP errors
            total_size = int(response.headers.get('content-length', 0))
            with export_fp0.open("wb") as file, tqdm(
                desc=export_fp0.name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):  # Download in chunks
                    size = file.write(chunk)
                    bar.update(size) 

def download_osm(studyarea_or_bbox,crs,network_type='bike'):
    '''
    Set to network_type=all_private to download everything, temporarily changing this to speed process up
    '''

    if isinstance(studyarea_or_bbox,tuple):
        print('Bounding box provided')
        #check to see if all values between -180 and 180
        bbox = np.array(studyarea_or_bbox)
    elif isinstance(studyarea_or_bbox,gpd.GeoDataFrame):
        print('GeoDataFrame provided')
        gdf = studyarea_or_bbox.copy()
        #convert to correct CRS
        if gdf.crs != 'epsg:4326':
            gdf.to_crs('epsg:4326',inplace=True)
        #get bounding box
        bbox = gdf.total_bounds
    else:
        print('Neither tuple of coords nor geodataframe was provided')

    #get bounding box
    minx, miny, maxx, maxy = bbox
    print(f'The bounding box is {minx}, {miny}, {maxx}, {maxy}')

    #get geometry info from osmnx
    osmnx_nodes, osmnx_links = download_osmnx(minx,miny,maxx,maxy,network_type)
    
    #project
    osmnx_nodes.to_crs(crs,inplace=True)
    osmnx_links.to_crs(crs,inplace=True)

    #get additonal attribute information from overpass
    overpass_links = overpass_download(minx,miny,maxx,maxy)
    
    return osmnx_nodes, osmnx_links, overpass_links

def simple_download(minx,miny,maxx,maxy):
    '''
    Just download the default OSMnx bike layer
    '''
    
    #retrieve graph from bbox
    G = ox.graph_from_bbox(maxy, miny, maxx, minx, network_type='bike')

    #convert to gdf
    nodes, links = ox.utils_graph.graph_to_gdfs(G)

    #reset index
    links = links.reset_index()
    
    #simplify columns to geo and id
    links = links[['osmid','bearing','geometry']]

    #reset nodes index
    nodes = nodes.reset_index()
    
    return nodes, links

def download_osmnx(minx,miny,maxx,maxy,network_type):
    
    #retrieve graph from bbox
    G = ox.graph_from_bbox(maxy, miny, maxx, minx, network_type=network_type, simplify=False)
    #G = ox.graph_from_bbox(maxy, miny, maxx, minx, network_type='all_private', simplify=False)
    
    #retrieve graph from the chosen study polygon
    #G = ox.graph_from_polygon(studyarea.geometry[0], network_type='all_private', simplify= False)
    
    #simplify graph unless different osm ids
    G = ox.simplification.simplify_graph(G, strict = False)
    
    #remove directed links (links for each direction)
    G = ox.utils_graph.get_undirected(G)
    
    #get link bearing
    G = ox.bearing.add_edge_bearings(G)

    #plot it for fun
    ox.plot_graph(G)

    #convert to gdf
    nodes, links = ox.utils_graph.graph_to_gdfs(G)

    #reset index
    links = links.reset_index()
    
    #simplify columns to geo and id
    links = links[['osmid','bearing','geometry']]

    #reset nodes index
    nodes = nodes.reset_index()

    return nodes, links

def overpass_download(minx,miny,maxx,maxy):
        
    query = f"""
    [out:json]
    [timeout:120]
    ;
    (
      way
        ["highway"]
        ({miny},{minx},{maxy},{maxx});
    );
    out body;
    >;
    out skel qt;
    """
    
    url = "http://overpass-api.de/api/interpreter"
    r = requests.get(url, params={'data': query})
    
    result = r.json()
    
    #simplify for dataframe
    df = pd.json_normalize(result, record_path = ['elements'])
    
    #clean up column names
    df.columns = df.columns.str.replace(r'tags.', '')

    return df

# Check forward and reverse order
def find_positions(seq, sublist):
    """Find start and end positions of sublist in seq."""
    for i in range(len(seq) - len(sublist) + 1):
        if seq[i:i + len(sublist)] == sublist:
            return i, i + len(sublist) - 1

def get_direction(osmid, node_sequence, way_nodes):
    # Larger list and smaller list
    larger_list = way_nodes.get(osmid)
    smaller_list = node_sequence

    # Double the larger list to account for wraparound
    doubled_larger_list = larger_list[:-1] + larger_list
    len_larger = len(larger_list) - 1

    # Find positions in normal list
    forward_positions = find_positions(larger_list, smaller_list)
    reverse_positions = find_positions(larger_list, smaller_list[::-1])

    if forward_positions:
        start, end = forward_positions
        return 'forward', start, end

    if reverse_positions:
        start, end = reverse_positions
        return 'reverse', start, end
    
    # Find positions in doubled list
    forward_positions = find_positions(doubled_larger_list, smaller_list)
    reverse_positions = find_positions(doubled_larger_list, smaller_list[::-1])

    if forward_positions:
        start, end = forward_positions
        # Adjust for wrapping
        if start >= len_larger:
            start -= len_larger
        if end >= len_larger:
            end -= len_larger
        return 'forward', start, end

    if reverse_positions:
        start, end = reverse_positions
        # Adjust for wrapping
        if start >= len_larger:
            start -= len_larger
        if end >= len_larger:
            end -= len_larger
        return 'reverse', start, end
    
    raise Exception("Something wrong happened")


# TODO there's a more efficient way to do this but it's important that each
# node matches to the right one

from scipy.spatial import cKDTree

def add_start_end_dists(links,raw_links,line_node_ids):
    
    # line_coords = raw_links['geometry'].apply(lambda x: tuple(x.coords))
    way_geo = dict(zip(raw_links['osmid'],raw_links['geometry']))
    
    # node_sequences = []
    start_dists = []
    end_dists = []
    for _, link in links.iterrows():
        #check if link is an area (these won't have way nodes)
        if link['area'] == 'yes':
            start_dists.append(np.nan)
            end_dists.append(np.nan)
            continue
        
        #get the link geo
        link_geo = list(link.geometry.coords)
        #get the list of possible nodes ids (from the raw links)
        possible_node_ids = line_node_ids.get(link['osmid'],False)

        #subset all nodes to just the nodes that could be a part of a link
        raw_geo = list(way_geo[link['osmid']].coords)# = all_nodes.loc[all_nodes['id'].isin(possible_node_ids),'geometry'].tolist()
        btree = cKDTree(raw_geo)
        dist, idx = btree.query(link_geo,k=1)
        #get the sequence of nodes
        node_sequence = [possible_node_ids[i] for i in idx]   
        #get whether it's forward or a wraparound
        #BUG get_direction doesn't always assign a direction
        try:
            result, start, end = get_direction(link['osmid'],node_sequence,line_node_ids)
        except:
            pass
        if start < end:
            #use position of the first point
            if start == 0:
                start_dist = 0
            else:
                start_dist = LineString(raw_geo[0:start+1]).length
            #use position of the second point
            end_dist = LineString(raw_geo[0:end+1]).length
        else:
            #return two ranges [0,point2] and [point2,point1]
            end_dist = LineString(raw_geo[0:end+1]).length
            start_dist = LineString(raw_geo[end:start+1]).length
        
        start_dists.append(round(start_dist,1))
        end_dists.append(round(end_dist,1))
        # node_sequences.append(node_sequence)

    links['start_dist'] = start_dists
    links['end_dist'] = end_dists
    # links['node_sequence'] = node_sequences
    return links

def get_way_node_seq(raw_links):
    raw_links = raw_links.copy()
    
    #get the node sequences (needed for assigning elevation)
    raw_links.set_index('osmid',inplace=True)
    line_node_ids = raw_links['all_tags'].apply(lambda x: x.get('@way_nodes'))
    line_node_ids = line_node_ids.dropna().to_dict()
    raw_links.reset_index(inplace=True)
    
    return line_node_ids

import json
def import_raw_osm_from_geojson(fp,include_tags,remove_tags):

    #import the geojson file
    with fp.open('rb') as fh:
        file_contents = fh.read()
        raw_osm = json.loads(file_contents)['features']

    #list for compiling the geodataframe
    gpkg = []

    for item in raw_osm:
        all_tags = item['properties'].copy()

        tag_values = []

        for tag in include_tags:
            tag_value = all_tags.get(tag,None)     
            tag_values.append(tag_value)
        
        #remove tags from the all tags dict to reduce data burden
        all_tags = {tag:value for tag,value in all_tags.items() if "tiger" not in tag}
        for remove_tag in remove_tags:
            all_tags.pop(remove_tag)
        
        #handle geometry
        geom = item['geometry']['coordinates']
        geom_type = item['geometry']['type']

        if geom_type == 'LineString':
            geom = LineString(geom)
        elif geom_type == 'MultiPolygon':
            geom = LineString(geom[0][0])
        elif geom_type == 'Point':
            geom = Point(geom)
        
        gpkg.append([*tag_values,all_tags,geom_type,geom])

    raw_df = pd.DataFrame(gpkg,columns=[*include_tags,'all_tags','geom_type','geometry'])   
    raw_gdf = gpd.GeoDataFrame(raw_df,crs='epsg:4326')

    raw_gdf.rename(columns={
        '@id':'osmid',
        '@timestamp':'timestamp',
        '@version':'version',
        '@type':'type',
    },inplace=True)

    #seperate ways from nodes
    raw_links = raw_gdf[raw_gdf['type']=='way'].copy()
    raw_nodes = raw_gdf[raw_gdf['type']=='node'].copy()

    #drop duplicate ways (when a way forms a polygon it seems to keep two copies)
    raw_links = raw_links[raw_links['osmid'].duplicated()==False]

    #reverse line geometry if oneway = -1
    #NOTE might need to reverse the nodes too?
    raw_links.loc[raw_links['oneway']=='-1','geometry'] = raw_links.loc[raw_links['oneway']=='-1','geometry'].apply(lambda x: x.reverse()).values
    raw_links.loc[raw_links['oneway']=='-1','oneway'] = 'yes'

    #project
    raw_links.to_crs(config['projected_crs_epsg'],inplace=True)
    raw_nodes.to_crs(config['projected_crs_epsg'],inplace=True)
        
    return raw_links, raw_nodes

