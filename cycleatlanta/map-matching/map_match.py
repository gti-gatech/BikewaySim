import pandas as pd
import geopandas as gpd
import numpy as np
from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching import visualization as mmviz
import pickle
import time
import datetime
from pathlib import Path
from tqdm import tqdm
from shapely.ops import Point, LineString, MultiLineString
import warnings
import itertools
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ShapelyDeprecationWarning)
pd.options.mode.chained_assignment = None

def prepare_network(edges,nodes,tolerance_ft):

    #explode network
    if isinstance(tolerance_ft,bool):
        exploded_edges = edges
        exploded_nodes = nodes
    else: 
        exploded_edges, exploded_nodes = explode_network(edges, nodes, tolerance_ft)

    #add latlon columns
    exploded_nodes['X'] = exploded_nodes.geometry.x
    exploded_nodes['Y'] = exploded_nodes.geometry.y

    #get rid of duplicate edges (shouldn't be a lot of these)
    exploded_edges['length_ft'] = exploded_edges.length
    idxmin = exploded_edges.groupby(['source','target'])['length_ft'].idxmin()
    exploded_edges = exploded_edges.loc[idxmin]

    #make network
    map_con = make_network(exploded_edges,exploded_nodes)

    return exploded_edges, exploded_nodes, map_con

def make_network(edges,nodes,allow_wrongway):
    '''
    Function for making network graph that is specific to leuven
    '''    
    
    #NEW VERSION ASSUMES DIRECTIONAL LINKS ARE ALREADY PROVIDED
    
    # create network graph needed for map matching (using a projected coordinate system so latlon false)
    map_con = InMemMap("network",use_latlon=False,use_rtree=True,index_edges=True)

    for idx, row in nodes.iterrows():
    #     if (allow_wrongway == False) & (row['wrongway'] == True):
    #         continue
        map_con.add_node(row['N'], (row['geometry'].x,row['geometry'].y))

    for idx, row in edges.iterrows():
        # if (allow_wrongway == False) & (row['reverse_link'] == True) & (row['oneway'] == True):
        #     continue
        map_con.add_edge(row['source'], row['target'])
            
    return map_con

def explode_network_midpoint(links,max_nodeid,max_linkid):
    '''
    This is a simplified version of explode network. Instead of using RDP
    and adding many edges/nodes to the network, it simply splits a line into
    segments using the midpoint. The return geometry will use straight line
    distance becuase Leuven draws straight lines anyways.

    '''

    new_links = {}
    new_nodes = {}

    #NEW VERSION ASSUMES THAT THERE ARE ALREADY DIRECTIONAL LINKS BEING PROVIDED
    
    for idx, row in links.iterrows():

        line = row['geometry']
        A = row['source']
        B = row['target']

        mid_id_forwards = max_nodeid + 1
        max_nodeid += 1

        # mid_id_backwards = max_nodeid + 1
        # max_nodeid += 1

        origin = np.array(line.coords)[0]
        dest = np.array(line.coords)[-1]
        mid_point = line.interpolate(line.length/2)

        #add these
        new_nodes[mid_id_forwards] = {
            'N': mid_id_forwards,
            'reverse_link': row['reverse_link'],
            'geometry': mid_point
        }
        # new_nodes[mid_id_backwards] = {
        #     'N': mid_id_backwards,
        #     'wrongway': row['oneway'],
        #     'geometry': mid_point
        # }

        #think more about what oneway means here
        first_line = LineString([origin,mid_point])
        second_line = LineString([mid_point,dest])

        #start to midpoint
        new_links[max_linkid+1] = {
            'source': A,
            'target': mid_id_forwards,
            'linkid': max_linkid+1,
            'source_linkid': row['linkid'],
            'reverse_link': row['reverse_link'],
            'geometry': first_line

        }
        max_linkid += 1

        #midpoint to end
        new_links[max_linkid+1] = {
            'source': mid_id_forwards,
            'target': B,
            'linkid': max_linkid+1,
            'source_linkid': row['linkid'],
            'reverse_link': row['reverse_link'],
            'geometry': second_line

        }
        max_linkid += 1

        # #end to midpoint
        # new_links[max_linkid+1] = {
        #     'source': B,
        #     'target': mid_id_backwards,
        #     'linkid': max_linkid+1,
        #     'source_linkid': row['linkid'],
        #     'wrongway': row['oneway'],
        #     'geometry': second_line
        # }
        # max_linkid += 1

        # #midpoint to start
        # new_links[max_linkid+1] = {
        #     'source': mid_id_backwards,
        #     'target': A,
        #     'linkid': max_linkid+1,
        #     'source_linkid': row['linkid'],
        #     'wrongway': row['oneway'],
        #     'geometry': first_line
        # }
        # max_linkid += 1

    new_nodes = pd.DataFrame.from_dict(new_nodes,orient='index')
    new_links = pd.DataFrame.from_dict(new_links,orient='index')

    return new_links, new_nodes


def explode_network(links,nodes,tolerance_ft):

    '''
    Leuven-matching assumes straight connecting lines between nodes to match to and doesn't allow
    for multi-edges.

    This function takes in a set of edges with reference ids and geometry attributes and breaks them
    down into smaller edges using the RDP algorithm. Higher tolerances will lead to fewer segments.

    An edge that is exploded into two segments due to its curvature will have 4 distinct edges and
    8 distinct nodes. This is to prevent backtracking/u-turning mid link, but does add substantially
    more edges/nodes and will likely increase computation time/memory usage. This is a hack!
    
    New nodes are assigned temporary nodes IDs that start with +1 after the current highest network
    node id. These nodes are then used to split up existing links and add new reference ids
    
    Map matching results can be post processed by removing all the temporary nodes. The remaining list
    of nodes can be used to construct the edge list. 
    
    Inputs:
        edges: dataframe with column A indicating starting node and B indicating ending node and a geometry column
        nodes: dataframe with N column and geometry columm
        tolerance_ft: float with tolerance in units of CRS to perform RDP increases will reduce added edges/nodes and 
    
    '''

    #make copies and subset
    links0 = links[['source','target','linkid','geometry']].copy()
    nodes0 = nodes[['N','geometry']].copy()
    
    #find links with same A/B
    #TODO test this
    links0['A_B'] = np.sort(links0[['source','target']].to_array()).duplicated()

    #rdp algorithm to simplify points
    links0.loc[links0['A_B']==True,'geometry'] = links0['geometry'].apply(lambda row: row.simplify(tolerance_ft))
    links0.set_geometry('geometry',inplace=True)
    
    #sets up the numbering
    numbering = nodes0['N'].max() + 1

    new_links = {}
    new_nodes = {}
    rem = []
    
    for idx, row in links0.iterrows():
        #only if there are more than 2 nodes in link
        if len(np.asarray(row['geometry'].coords)) > 2:
            #add link to remove list
            rem.append(row['linkid'])
            #grab coordinates of line
            line = np.asarray(row['geometry'].coords)
            #go through each shapepoint of the link
            for i in range(0,len(line)-1):
                #form link with two nodes (current and next)
                new_link = LineString([line[i],line[i+1]])
                #first node condition
                #add new link with start node id and new node id
                if i == 0:
                    new_links[(row['source'],numbering+1,row['linkid'])] = new_link
                #last node condition
                #add new link with new node id and last node id
                elif i == len(line) - 2:
                    new_links[(numbering+i,row['target'],row['linkid'])] = new_link
                    new_nodes[numbering+i] = Point(line[i])
                #interstitial node condition
                #add new link with 2 new node ids
                else:
                    new_links[(numbering+i,numbering+i+1,row['linkid'])] = new_link
                    #add first node to list
                    new_nodes[numbering+i] = Point(line[i])
    
            #increment numbering by the number of unique ids assigned
            numbering += len(line) - 2
            
    #turn new link dict to gdf
    new_links = pd.DataFrame.from_dict(new_links,orient='index',columns=['geometry']).reset_index()
    new_links['source'] = new_links['index'].apply(lambda row: row[0])
    new_links['target'] = new_links['index'].apply(lambda row: row[1])
    new_links['linkid'] = new_links['index'].apply(lambda row: row[2])
    new_links = new_links[['source','target','linkid','geometry']]
    new_links = gpd.GeoDataFrame(new_links,geometry='geometry',crs=links.crs)
    
    #turn new nodes from dict to gdf
    new_nodes = pd.DataFrame.from_dict(new_nodes,orient='index',columns=['geometry']).reset_index()
    new_nodes.columns = ['N','geometry']
    new_nodes = gpd.GeoDataFrame(new_nodes,geometry='geometry',crs=links.crs)
    
    #remove links that were split up
    links0 = links0[-links0['linkid'].isin(rem)]
    
    #nodes
    nodes0['added'] = False
    new_nodes['added'] = True
    
    #add new links and nodes
    links0 = pd.concat([links0,new_links],ignore_index=True)
    nodes0 = pd.concat([nodes0,new_nodes],ignore_index=True)

    #add the oneway column back
    links0 = pd.merge(links0,links[['linkid','oneway']],on='linkid')

    return links0, nodes0

#for exporting to gpkg
def convert_datetime_columns_to_str(dataframe):
    for column in dataframe.select_dtypes(include=['datetime64','timedelta64']):
        dataframe[column] = dataframe[column].astype(str)
    return dataframe

def leuven_match(trace:gpd.GeoDataFrame,matching_settings:dict,map_con,network_edges):

    start = time.time()

    #reset index and sequence (do in pre-processing)
    trace.drop(columns=['sequence'],inplace=True)
    trace.reset_index(inplace=True,drop=True)
    trace.reset_index(inplace=True)
    trace.rename(columns={'index':'sequence'},inplace=True)
    
    #get list of points
    gps_trace = list(zip(trace.geometry.x,trace.geometry.y))
    
    #set up matching
    matcher = DistanceMatcher(map_con,**matching_settings)

    #perform matching
    states, last_matched = matcher.match(gps_trace)
    nodes = matcher.path_pred_onlynodes

    #match ratio
    match_ratio = last_matched / (len(gps_trace)-1)
    
    #repeat match if failed and update states (takes longer)
    # if match_ratio < .95:
    #     states, last_matched = matcher.increase_max_lattice_width(50, unique=False)
    #     nodes = matcher.path_pred_onlynodes
    #     match_ratio = last_matched / (len(gps_trace)-1)
    
    #if still few matched points condsider it a failed match
    if (last_matched < 2):
        return 'failed match'

    #turn node list into edge list
    #NOTE we could try to figure out the travel time per link with the nodes data
    edges = post_process_bisected(nodes,network_edges)

    #get interpolated points
    trace['interpolated_point'] = pd.Series([Point(x.edge_m.pi[0],x.edge_m.pi[1]) for x in matcher.lattice_best])

    #drop non-matched
    raw_trace = trace.copy()
    trace = trace[trace['interpolated_point'].notna()]

    #create match lines
    trace['match_lines'] = trace.apply(lambda row: LineString([row['geometry'],row['interpolated_point']]),axis=1)

    #create gdf for interpolated points
    interpolated_points = trace[['sequence','interpolated_point']]
    interpolated_points = gpd.GeoDataFrame(interpolated_points,geometry='interpolated_point',crs=trace.crs)

    #create gdf for match lines
    match_lines = trace[['sequence','match_lines']]
    match_lines = gpd.GeoDataFrame(match_lines,geometry='match_lines',crs=trace.crs)
    match_lines['length'] = match_lines.length

    #add to matched_traces dictionary
    results = {
    'edges':edges, #df of the edge ids + direction of travel
    'last_matched':last_matched, #last gps point reached
    'match_ratio':match_ratio, #percent of points matched
    'max_lattice_width':matcher.max_lattice_width, # record the final lattice width
    #'matched_trip': matched_trip, #gdf of matched lines
    'trace': raw_trace, #gdf of the gps trace
    'match_lines': match_lines, # gdf of distances between interpolated point and gps point
    'interpolated_points': interpolated_points, # gdf of interpolated points on match line
    'match_time_sec': np.round(time.time() - start,1), #time it took to match
    'gps_distance': matcher.path_distance(), # total euclidean distance between all gps points
    'time': datetime.datetime.now(), # record when it was last matched
    'settings': matching_settings # record what settings were used for repeatability
    }

    return results

#TODO modify this to work with the bisected lines
def post_process_bisected(nodes,exploded_edges): 
    '''
    This function post processes results from map matching into 
    a simple edge list indicating the linkid and direction of
    travel.

    This assumes the bisected network was used meaning that certain
    nodes aren't 

    The edges need an A, B, linkid, and source linkid column
    '''

    #turn node list into edge list
    edges = list(zip(nodes,nodes[1:]))

    #turn into dataframe
    new_df = pd.DataFrame(edges, columns=['source', 'target'])

    #two merges, first adds any forward links and the second adds backwards links
    merged_df = pd.merge(new_df,exploded_edges[['source','target','linkid','reverse_link']],on=['source','target'],how='left')
    
    #NOTE only needed if not using a directed network already
    #merged_df['reverse_link'] = merged_df['linkid'].isna()
    #merged_df.loc[merged_df['linkid'].isna(),'linkid'] = pd.merge(new_df, exploded_edges[['source','target','linkid']], left_on=['target', 'source'], right_on=['source', 'target'], how='left')['linkid']

    #create a mapping dict for replacing exploded link id with the source linkid
    mapping = exploded_edges[['linkid','source_linkid']].dropna()
    mapping = dict(zip(mapping['linkid'],mapping['source_linkid']))
    merged_df.loc[~merged_df['linkid'].map(mapping).isna(),'linkid'] = merged_df['linkid'].map(mapping)

    #remove the exploded links (they'll have repeat linkids)
    merged_df.drop(columns=['source','target'],inplace=True)

    # remove adjacent links if they're the same link
    mask = (merged_df == merged_df.shift(-1)).all(axis=1)

    return merged_df[~mask]


def post_process(nodes,exploded_edges): 
    '''
    This function post processes results from map matching into 
    a simple edge list indicating the linkid and direction of
    travel.

    The edges need an A, B, and linkid column
    '''

    #create edges
    edges = list(zip(nodes,nodes[1:]))

    #turn into dataframe
    new_df = pd.DataFrame(edges, columns=['source', 'target'])

    #add linkids
    merged_df = pd.merge(new_df,exploded_edges[['source','target','linkid']],on=['source','target'],how='left')

    # if na then reverse link (true = forward, false = backwards)
    merged_df['forward'] = -merged_df['linkid'].isna()

    # Replace missing values in 'linkid' with the reversed tuple values
    merged_df.loc[merged_df['linkid'].isna(),'linkid'] = pd.merge(new_df, exploded_edges[['source','target','linkid']], left_on=['target', 'source'], right_on=['source', 'target'], how='left')['linkid']

    # Fill any remaining missing values with a default value (you can modify this as needed)
    #merged_df['linkid'] = merged_df['linkid'].fillna(-1).astype(int)

    # remove exploded links using the linkid column
    # exploded links don't have unique linkids
    merged_df.drop(columns=['source','target'],inplace=True)
    
    # remove adjacent links if they're the same link
    mask = (merged_df == merged_df.shift(-1)).all(axis=1)

    return merged_df[~mask]

import folium
import geopandas as gpd
from folium.plugins import MarkerCluster, PolyLineTextPath
from folium.map import FeatureGroup

from branca.colormap import linear


# 
def visualize_trips():
    '''
    This function takes in three or more trip results and displays them
    accordingly
    '''

    return



#todo display previous note about match if it's there from qaqc dict
def visualize_match(tripid,match_dict,edges):

    #tripid = 29837#7257#9806#30000#8429

    # Your GeoDataFrames
    matched_trip = match_dict[tripid]['edges'].merge(edges, on=['linkid','reverse_link'])
    matched_trip = gpd.GeoDataFrame(matched_trip)
    gps_points = match_dict[tripid]['trace']
    match_lines = match_dict[tripid]['match_lines']

    #TODO add a buffer rating here (how many gps point within 500 ft of the matched trip?)
    buffered_line = MultiLineString(matched_trip.geometry.tolist()).buffer(50)
    buffer_rating = gps_points.geometry.intersects(buffered_line).sum()

    #get the start and end point for mapping
    start_pt = gps_points.to_crs(epsg='4326').loc[gps_points['sequence'].idxmin(),'geometry']
    end_pt = gps_points.to_crs(epsg='4326').loc[gps_points['sequence'].idxmax(),'geometry']

    # reproject and get the center of the map
    x_mean = gps_points.to_crs(epsg='4326')['geometry'].x.mean()
    y_mean = gps_points.to_crs(epsg='4326')['geometry'].y.mean()

    # Create a Folium map centered around the mean of the GPS points
    center = [y_mean,x_mean]
    mymap = folium.Map(location=center, zoom_start=14)

    # round speed and acceleration
    gps_points['speed_mph'] = gps_points['speed_mph'].round(0)
    gps_points['acceleration_ft/s**2'] = gps_points['acceleration_ft/s**2'].round(2)

    # Convert GeoDataFrames to GeoJSON
    matched_trip_geojson = matched_trip[['linkid','geometry']].to_crs(epsg='4326').to_json()
    gps_points_geojson = gps_points[['sequence','speed_mph','acceleration_ft/s**2','geometry']].to_crs(epsg='4326').to_json()
    match_lines_geojson = match_lines[['sequence','match_lines']].to_crs(epsg='4326').to_json()

    # Create FeatureGroups for each GeoDataFrame
    matched_trip_fg = FeatureGroup(name='Matched Trip')
    gps_points_fg = FeatureGroup(name='GPS Points')
    match_lines_fg = FeatureGroup(name='Match Lines')

    # Add GeoJSON data to FeatureGroups
    folium.GeoJson(matched_trip_geojson, name='Matched Trip', style_function=lambda x: {'color': 'red'}).add_to(matched_trip_fg)

    # Add circles to the GPS Points FeatureGroup
    # Symbolize by speed and have it be a popup
    colormap = linear.YlOrRd_09.scale(gps_points['speed_mph'].min(),gps_points['speed_mph'].max())
    folium.GeoJson(
        gps_points_geojson,
        name="GPS Points",
        marker=folium.Circle(radius=8, fill_color="orange", fill_opacity=1, color="black", weight=0),
        tooltip=folium.GeoJsonTooltip(fields=['sequence','speed_mph','acceleration_ft/s**2']),
        popup=folium.GeoJsonPopup(fields=['sequence','speed_mph','acceleration_ft/s**2']),
        style_function= lambda feature: {
            'fillColor': colormap(feature['properties']['speed_mph']),
        },
        highlight_function=lambda feature: {"color":"yellow","weight":3}
        ).add_to(gps_points_fg)
    colormap.caption = "Speed (mph)"
    colormap.add_to(mymap)

    # for idx, row in gps_points.iterrows():
    #     folium.Circle(location=[row['lat'], row['lon']], radius=5, color='grey', fill=True, fill_color='grey').add_to(gps_points_fg)

    # Add GeoJSON data to Match Lines FeatureGroup with transparent and grey style
    folium.GeoJson(match_lines_geojson, name='Match Lines', style_function=lambda x: {'color': 'grey', 'opacity': 0.5}).add_to(match_lines_fg)

    # Add FeatureGroups to the map
    matched_trip_fg.add_to(mymap)
    gps_points_fg.add_to(mymap)
    match_lines_fg.add_to(mymap)

    # Add start and end points with play and stop buttons
    start_icon = folium.Icon(color='green',icon='play',prefix='fa')
    end_icon = folium.Icon(color='red',icon='stop',prefix='fa')
    folium.Marker(location=[start_pt.y, start_pt.x],icon=start_icon).add_to(mymap)
    folium.Marker(location=[end_pt.y, end_pt.x],icon=end_icon).add_to(mymap)

    # Add layer control to toggle layers on/off
    folium.LayerControl().add_to(mymap)

    # Add legend with statistics
    #TODO what happened to duration
    legend_html = f'''
        <div style="position: fixed; 
                bottom: 5px; left: 5px; width: 300px; height: 250px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color: white;
                opacity: 0.9;">
        &nbsp; <b>Trip ID: {tripid} </b> <br>
        &nbsp; <b> Match Date: {match_dict[tripid]['time']} </b> <br>
        &nbsp; Start Point &nbsp; <i class="fa fa-play" style="color:green"></i>,
        End Point &nbsp; <i class="fa fa-stop" style="color:red"></i> <br>
        
        &nbsp; Matched Path &nbsp; <div style="width: 20px; height: 5px; background-color: red; display: inline-block;"></div> <br>
        &nbsp; Match Lines Path &nbsp; <div style="width: 20px; height: 5px; background-color: gray; display: inline-block;"></div> <br>
    
        &nbsp; Points Matched: {match_dict[tripid]['last_matched']}/{match_dict[tripid]['trace'].shape[0]} <br>
        &nbsp; Match Ratio: {match_dict[tripid]['match_ratio']:.2f} <br>
        &nbsp; Points within 50ft Buffer: {buffer_rating} / {match_dict[tripid]['trace'].shape[0]} <br>
        &nbsp; GPS Distance: {match_dict[tripid]['gps_distance']:.1f} ft. <br>
        &nbsp; Matched Trace Distance: {matched_trip.length.sum():.0f} ft. <br>
        &nbsp; Mean Matching Distance: {match_dict[tripid]['match_lines'].length.mean():.0f} ft. 

        </div>
        '''
    mymap.get_root().html.add_child(folium.Element(legend_html))

    # Save the map to an HTML file or display it in a Jupyter notebook
    #mymap.save('map.html')
    # mymap.save('/path/to/save/map.html')  # Use an absolute path if needed
    mymap  # Uncomment if you are using Jupyter notebook

    #TODO add in the legend with trip info and then we're golden
    return mymap


# #%% load data

# #file paths
# export_fp = Path.home() / 'Downloads/cleaned_trips'
# network_fp = Path.home() / "Downloads/cleaned_trips/networks/final_network.gpkg"

# #import network
# edges = gpd.read_file(network_fp,layer="links")[['source','target','geometry']]
# nodes = gpd.read_file(network_fp,layer="nodes")[['N','geometry']]

# #explode network
# exploded_edges, exploded_nodes = explode_network(edges, nodes, tolerance_ft = 50)

# #add latlon columns
# exploded_nodes['X'] = exploded_nodes.geometry.x
# exploded_nodes['Y'] = exploded_nodes.geometry.y

# #load all traces
# with (export_fp/'cleaned_traces.pkl').open('rb') as fh:
#     coords_dict, trips_df = pickle.load(fh)

# map_con = make_network(exploded_edges,exploded_nodes,n_col='N',lat_col='Y',lon_col='X',a_col='source',b_col='target',use_latlon=False)

# #note: distance units will be units of projected CRS
# matching_settings = {
#     'max_dist':700,  # maximum distance for considering a link a candidate match for a GPS point
#     'min_prob_norm':0.001, # drops routes that are below a certain normalized probability  
#     'non_emitting_length_factor': 0.75, # reduces the prob of non-emitting states the longer the sequence is
#     'non_emitting_states': True, # allow for states that don't have matching GPS points
#     'obs_noise': 50, # the standard error in GPS measurement 50 ft (could use hAccuracy)
#     'max_lattice_width': 5, # limits the number of possible routes to consider, can increment if no solution is found
#     'increase_max_lattice_width' : True, # increases max lattice width by one until match is complete or max lattice width is 50
#     'export_graph': False # set to true to export a .DOT file to visualize in graphviz or gephi (this feature is messy)
# }

# #load existing matches/if none then create a new dict
# if (export_fp/'matched_traces.pkl').exists():
#     with (export_fp/'matched_traces.pkl').open('rb') as fh:
#         matched_traces, trips_df, failed_match = pickle.load(fh)
# else:
#     matched_traces = dict()
#     failed_match = []

# #%%single match
# #matched_traces = leuven_match(trace,tripid,matched_traces,matching_settings,edges,map_con)

# #%% batch match
# for tripid, trace in tqdm(coords_dict.items()):
#     try:
#         matched_traces, trips_df = leuven_match(trace,tripid,matched_traces,matching_settings,edges,map_con)
#         #update trips_df
#     except:
#         if tripid in matched_traces.keys():
#             failed_match.append(tripid)
#         export_files = (matched_traces,trips_df,failed_match)
#         with (export_fp/'matched_traces.pkl').open('wb') as fh:
#             pickle.dump(export_files,fh)
        
        
# #export
# export_files = (matched_traces,trips_df,failed_match)
# with (export_fp/'matched_traces.pkl').open('wb') as fh:
#     pickle.dump(export_files,fh)
