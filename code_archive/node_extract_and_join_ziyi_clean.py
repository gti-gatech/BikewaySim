# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 14:48:42 2021

@author: Daisy
"""

#%%import cell
import pandas as pd
pd.options.display.max_columns = None  # display all columns
pd.options.display.max_rows = None  # display all columns
import geopandas as gpd
import numpy as np
from shapely import wkt
from shapely.wkt import dumps
from shapely.ops import transform
from shapely.ops import split, snap
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, mapping
import os
import shapely
import pyproj
from itertools import compress

#change this to where you stored this folder
os.chdir(r'C:\Users\tpassmore6\Documents\BikewaySim\Python Network Generation')

#%% import networks
#first we want to import the road networks we filtered in the previous script
#our primary goal here is to figure out how best to combine the ABM network and the navteq network
#we suspect that navteq, basased the on the mileage, number of links, and nodes + all the other information provided makes it a good candidate
#for the base network of bikewaysim

#ABM Links
abm_linksfp = r'shapefiles/clipped_shapefiles/abm/abmLinks_studyArea.shp'
abmLinks = gpd.read_file(abm_linksfp)


#NAVTEQ
navteq_linksfp = r'shapefiles/clipped_shapefiles/navteq/navteqLinks_studyArea.shp'
navteqLinks = gpd.read_file(navteq_linksfp)


#OSM
osm_linksfp = r'shapefiles/clipped_shapefiles/osm/osmLinks_studyArea.shp'
osmLinks = gpd.read_file(osm_linksfp).to_crs(epsg=2240)

#%% Extract Start and End Points Code from a LineString as tuples
def startNode(row, geom):
    return (row[geom].coords.xy[0][0], row[geom].coords.xy[1][0]) #basically look at x and then y coord, use apply to do this for every row of a dataframe

def endNode(row, geom):
    return (row[geom].coords.xy[0][-1], row[geom].coords.xy[1][-1])

#%% rename or create node ID's
def renameNodes(df, network_name, A, B):
    df = df.rename(columns={A:f'A_{network_name}',B:f'B_{network_name}'})
    return df

#%% rename or create node ID's
#need to create new nodes IDs for OSM and TIGER
#first will need to extract nodes
#CAUTION, node IDs will change based on study area, and/or ordering of the data 
def createNodeIDs(df, network_name):
    
    #add start and end coordinates to each line
    #this is so we can match them  
    df['startNode'] = df.apply(startNode, geom='geometry', axis=1)
    df['endNode'] = df.apply(endNode, geom='geometry', axis=1)
    
    #this function will number each node to give it an ID
    nodes_coords = df['startNode'].append(df['endNode'])
    
    #create dataframe
    nodes = pd.DataFrame({f'coords_{network_name}':nodes_coords})
    
    #find number of intersecting links
    nodes[f'num_links_{network_name}'] = 1 #this column will be used to count the number of links
    
    #for each unique ID and coordinate, count number of links
    nodes = nodes.groupby([f'coords_{network_name}'], as_index=False).count()
    
    #give nodes row number name
    nodes[f'ID_{network_name}'] = np.arange(nodes.shape[0])
    
    #extract only the ID and coords column for joining
    nodes = nodes[[f'ID_{network_name}',f'coords_{network_name}']]
    
    
    #perform join back to original dataframe
    #rename id to be A, rename coords to match startNode
    joining = nodes.rename(columns={f'ID_{network_name}':f'A_{network_name}',f'coords_{network_name}':'startNode'})
    df = pd.merge(df, joining, how = 'left', on = 'startNode' )

    #rename id to be B, rename coords to match endNode
    joining = nodes.rename(columns={f'ID_{network_name}':f'B_{network_name}',f'coords_{network_name}':'endNode'})
    df = pd.merge(df, joining, how = 'left', on = 'endNode' )
    df = df.rename(columns={f'ID_{network_name}':f'B_{network_name}'})
        
    
    return df

#%% rename and make id for nodes

abmLinks = renameNodes(abmLinks,'abm','A','B')
navteqLinks = renameNodes(navteqLinks, 'navteq','REF_IN_ID','NREF_IN_ID')

osmLinks = createNodeIDs(osmLinks,'osm')
#%% Since ABM has reversible links, we delete B_A rows if A_B rows already exists 
## this would help with further identification of whether a point lands on a line
def del_dup_row(df, network_name):
    df_dup = pd.DataFrame(np.sort(df[[f"A_{network_name}",f"B_{network_name}"]], axis=1), 
                          columns=[f"A_{network_name}",f"B_{network_name}"]).drop_duplicates()
    df_cleaned = df.merge(df_dup, how = "inner", on = [f"A_{network_name}",f"B_{network_name}"])
    return df_cleaned

# delete duplicate links in abm network
abmLinks = del_dup_row(abmLinks, 'abm')

#%% Create node layer from lines
#do this from the links to make sure that all nodes are included even if they would have been clipped
#by the study area  
def makeNodes(df, network_name): 
    #add start and end coordinates to each line
    #this is so we can match them  
    df['startNode'] = df.apply(startNode, geom='geometry', axis=1)
    df['endNode'] = df.apply(endNode, geom='geometry', axis=1)
        
    nodes_id = df[f'A_{network_name}'].append(df[f'B_{network_name}'])
    nodes_coords = df['startNode'].append(df['endNode'])
    
    nodes = pd.DataFrame({f"ID_{network_name}":nodes_id,f"coords_{network_name}":nodes_coords})
    
    #find number of intersecting links
    nodes[f'num_links_{network_name}'] = 1 #this column will be used to count the number of links
    
    #for each unique ID and coordinate, count number of links
    nodes = nodes.groupby([f'ID_{network_name}',f'coords_{network_name}'], as_index=False).count()
    
    #turn the coordinates into points so we can do spatial mapping
    nodes[f'coords_{network_name}'] = nodes.apply(lambda row: Point([row[f'coords_{network_name}']]), axis=1)
    
    #convert to GeoDataFrame and set geo and CRS
    nodes = gpd.GeoDataFrame(nodes).set_geometry(f'coords_{network_name}').set_crs(epsg=2240)
    
    #export nodes as shapefile
    nodes.to_file(rf'shapefiles/node_matching/{network_name}Nodes.shp')
    
    return nodes

#%% Create node shapefiles

abmNodes = makeNodes(abmLinks, 'abm')
navteqNodes = makeNodes(navteqLinks,'navteq')
osmNodes = makeNodes(osmLinks,'osm')

#%% Alt Method for finding nearest point
#https://gis.stackexchange.com/questions/222315/geopandas-find-nearest-point-in-other-dataframe

from scipy.spatial import cKDTree

def ckdnearest(gdA, gdB):  

    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].reset_index(drop=True)
    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest,
            pd.Series(dist, name='dist')
        ], 
        axis=1)

    return gdf

#%%Function for matching ABM nodes to other network nodes

#later figure out how to do the exporting through this too

def matchNodes(df1_nodes, df1_name, df2_nodes, df2_name, append_name, tolerance_ft):

    #match the nodes using the nearest_neighbor function
    closest_nodes = ckdnearest(df1_nodes, df2_nodes)
   
    #error lines
    #make a line between matched node
    closest_nodes['error_lines'] = closest_nodes.apply(lambda row: LineString([row[f'coords_{df1_name}'], row[f'coords_{df2_name}']]), axis=1)
    closest_nodes = closest_nodes.set_geometry('error_lines').set_crs(epsg=2240)
    closest_nodes['error_distance'] = closest_nodes.length
    
    #filter out matched nodes where the match is greater than specified amount aka tolerence
    matched_nodes = closest_nodes[closest_nodes['error_distance'] <= tolerance_ft]
    
    #QA Check
    #Look at how closely the matched nodes to see if results are reasonable
    print(matched_nodes.length.describe())
    
    #drop line and navteq geo and make ABM geo active
    #changed
    matched_nodes = matched_nodes.filter([f'ID_{df1_name}', f'coords_{df1_name}', f'ID_{df2_name}' ], axis = 1)
    matched_nodes = matched_nodes.set_geometry(f'coords_{df1_name}')
    
    #check uniqueness
    #this makes sure that it was a one-to-one match
    QAcheck = matched_nodes.copy()
    QAcheck['unique'] = 1
    QAcheck = QAcheck.groupby([f'ID_{df2_name}'], as_index=False).count()
    QAcheck = QAcheck[QAcheck['unique']>1]
    print(f'There are {len(QAcheck.index)} {df1_name} nodes matching to the same {df2_name} node, not a one-to-one match')
    

    matched_nodes.to_file(rf'shapefiles/node_matching/{df1_name}_matched_to_{df2_name}_{append_name}.shp')
    return matched_nodes

def remainingNodes(matched_nodes, base_nodes, network_name, append_name):
    #unmatched nodes file 
    unmatched_nodes = base_nodes[-base_nodes[f'ID_{network_name}'].isin(matched_nodes[f'ID_{network_name}'])].reset_index(drop = True)
    print(f'There are {len(unmatched_nodes.index)} {network_name} nodes remaining')
    unmatched_nodes.to_file(rf'shapefiles/node_matching/unmatched_{network_name}_{append_name}.shp')
    return unmatched_nodes

#%% matching

#Filter NAVTEQ
#This should make it so that the only NAVTEQ nodes that ABM can join to represent real intersections
navteqNodes_filt = navteqNodes[navteqNodes['num_links_navteq'] != 2 ].reset_index(drop=True)

matched_abm_navteqfilt = matchNodes(abmNodes, 'abm', navteqNodes_filt, 'navteq', 'navteq_filt', 26)
unmatched_nodes_abm_navteqfilt  = remainingNodes(matched_abm_navteqfilt, abmNodes, 'abm', 'navteq_filt')
unmatched_nodes_navteq_navteqfilt  = remainingNodes(matched_abm_navteqfilt, navteqNodes_filt, 'navteq', 'navteq_filt')

#%% select unmatched navteq points which lie on the abm links and find its interpolated point that lie specifically on abm (for precision split)
def point_on_line(df_point, df_point_name, df_line, df_line_name, tolerance_ft):
    df_point_on_line_p = pd.DataFrame() # dataframe for storing the corresponding interpolated point information 
    df_point_on_line_l = pd.DataFrame() # dataframe for storing the correspoding abm link information
    for index, row in df_point.iterrows():
        # loop through every unmatched point, as long as the point lie on one link of the whole newtork, it would be identified as lie on network
        on_bool_list = df_line["geometry"].distance(row[f"coords_{df_point_name}"]) < tolerance_ft  # bool lit for extracting the specific link the point lie on
        if any(on_bool_list) == True:
            line_idx = list(compress(range(len(on_bool_list)), on_bool_list)) # find the corresponding line
            target_line = df_line.loc[line_idx[0],"geometry"]
            interpolated_point = target_line.interpolate(target_line.project(row[f"coords_{df_point_name}"])) # find the interpolated point on the line
            df_point.at[index, f"lie_on_{df_line_name}"] = "Y"
            df_point_on_line_p.at[index, f"ip_{df_line_name}_point"] = str(Point(interpolated_point)).strip() 
            df_point_on_line_l.at[index, f"ip_{df_line_name}_line"] = str(LineString(target_line)).strip()
            df_point_on_line_p.at[index, f"cor_ID_{df_point_name}"] = row[f"ID_{df_point_name}"]
            df_point_on_line_p.at[index, "A_B"] = df_line.loc[line_idx[0], "A_B"]
            df_point_on_line_l.at[index, f"cor_ID_{df_point_name}"] = row[f"ID_{df_point_name}"]
            df_point_on_line_l.at[index, "A_B"] = df_line.loc[line_idx[0], "A_B"]
        else:
            df_point.at[index, f"lie_on_{df_line_name}"] = "N"
            
    df_point_on = df_point[df_point[f"lie_on_{df_line_name}"] == "Y"].reset_index(drop = True)
    df_point_on_line_p = df_point_on_line_p.reset_index(drop = True)
    df_point_on_line_p[f"ip_{df_line_name}_point"] = df_point_on_line_p[f"ip_{df_line_name}_point"].apply(wkt.loads) # transform from df to gdf
    gdf_point_on_line_p = gpd.GeoDataFrame(df_point_on_line_p, geometry = f"ip_{df_line_name}_point").set_crs(epsg=2240)
    df_point_on_line_l = df_point_on_line_l.reset_index(drop = True)
    df_point_on_line_l[f"ip_{df_line_name}_line"] = df_point_on_line_l[f"ip_{df_line_name}_line"].apply(wkt.loads)
    gdf_point_on_line_l = gpd.GeoDataFrame(df_point_on_line_l, geometry = f"ip_{df_line_name}_line").set_crs(epsg=2240) # transform from df to gdf
       
    print(f"There are {len(df_point_on.index)} found to lie on {df_line_name} network")
    df_point_on.to_file(rf'shapefiles/node_matching/unmatched_{df_point_name}_lie_on_{df_line_name}.shp')
    gdf_point_on_line_p.to_file(rf'shapefiles/node_matching/unmatched_{df_point_name}_lie_on_{df_line_name}_ip_point.shp')
    gdf_point_on_line_l.to_file(rf'shapefiles/node_matching/unmatched_{df_point_name}_lie_on_{df_line_name}_ip_line.shp')
    return df_point_on, gdf_point_on_line_p,gdf_point_on_line_l

# find unmatched navteq nodes lie on abm, find correspoding interpolated abm node, find corresponding abm link        
unmatched_nav_on,corre_abm_point,corre_abm_link = point_on_line(unmatched_nodes_navteq_navteqfilt, "navteq", abmLinks, "abm", tolerance_ft = 25)

#%% add node to existing links
# idea behind:
## step 1:return the multistring as a string first (dataframe), since multistring does not split into individual linestring segment, but just add element to list of linestrings
## step 2: expand list of linestring column into several rows, return a dataframe with more rows 
## step 3: turn the dataframe into a geodataframe

def get_linesegments(point, line):  # function to split line into MultiLineString (ATTENTION: not into individual segments, but to MultiLineString)
     return line.difference(point.buffer(1e-6)) #IMPORTANT: add a buffer here make sure it works

def split_by_node_to_multilinestring(gdf_line, line_geom_name, gdf_point, point_geom_name):
    ab_list = gdf_point["A_B"].unique().tolist()
    gdf_line = gdf_line.drop_duplicates(subset = ["A_B"]) # multiple points could line on the same link, drop duplicates first
    df_split = pd.DataFrame(columns = {"A_B","geometry"}) # dataframe for storing splitted multistring
    df_split["A_B"] = ab_list
    for idx, row in df_split.iterrows():
        ab = row["A_B"]
        df_ab = gdf_point[gdf_point["A_B"] == ab]
        ab_point = MultiPoint([x for x in df_ab[point_geom_name]])
        #print(ab_point)
        ab_line = gdf_line[gdf_line["A_B"] == ab][line_geom_name].values[0]
        #print(df_ab_line)
        split_line = get_linesegments(ab_point, ab_line) # split_line is a geo multilinestring type
        # ATTENTION: format the decimal places to make every row the same, this is important for successfully turning string to geopandas geometry
        split_line = dumps(split_line) # use dump to always get 16 decimal digits irregardles of the total number of digits, dump() change it to MultiLineString to string type
        print(split_line) # spline_line is MultiLineString type, split_line.wkt is string element
        df_split.at[idx, "geometry"] = split_line
    return df_split
    
def transform_multilinestring_to_segment(df_split, network_link, network_name):
    # ATTENTION: make sure it's a multilinestring element, need to make sure the consistence of data type
    # if the added node coincidentally lie on the end, it still turns a LineString element not a MultiLineString element
    df_split = df_split[df_split["geometry"].str.contains("MULTI")] 
    df_split["geometry"] = df_split["geometry"].str.replace("MULTILINESTRING ","").str.slice(1,-1)
    # explode the list of list columns into separate individual rows
    df_split_line = pd.concat([pd.Series(row['A_B'], row['geometry'].split('), (')) for _, row in df_split.iterrows()]).reset_index()
    df_split_line.rename(columns = {"index":"geometry", 0:"A_B"}, inplace = True)
    # modify each linestring point list into a LineString-like string
    df_split_line["geometry"] = df_split_line["geometry"].str.replace(r"(","").str.replace(")","")
    df_split_line["geometry"] = "LineString (" + df_split_line["geometry"] + ")" 
    gdf_split_line = gpd.GeoDataFrame(columns = {"A_B","geometry"}).set_crs(epsg=2240) # construct the geo dataframe
    for i in range(len(df_split_line.index)):
        ab = df_split_line.loc[i, "A_B"]
        geom = df_split_line.loc[i,"geometry"]
        g = shapely.wkt.loads(geom) # the function turns string element into geopandas geometry element
        print(g)
        gdf_split_line.loc[i,"A_B"] = ab
        gdf_split_line.loc[i,"geometry"] = g
    gdf_split_line.to_file(rf'shapefiles/line_splitting/splitted_{network_name}_link.shp')
    network_link = network_link[["A_B","geometry"]]
    rest_network_line = network_link[~network_link["A_B"].isin(df_split["A_B"].unique().tolist())]
    rest_network_line.to_file(rf'shapefiles/line_splitting/unsplitted_{network_name}_link.shp')
    return gdf_split_line, rest_network_line

df_abm_split = split_by_node_to_multilinestring(corre_abm_link, "ip_abm_line", corre_abm_point, "ip_abm_point")
gdf_abm_split, gdf_abm_nosplit = transform_multilinestring_to_segment(df_abm_split, abmLinks, "abm")


#%% This cell is a non-function based procedural programming for you to test, so that you can have a view of variable after each step
ab_list = corre_abm_point["A_B"].unique().tolist()
cor_ab = gpd.GeoDataFrame({"A_B":ab_list})
cor_abm_link = corre_abm_link.drop_duplicates(subset = ["A_B"])
trans = pd.DataFrame(columns = {"A_B","geometry"})
trans["A_B"] = ab_list
    
for idx, row in trans.iterrows():
    ab = row["A_B"]
    df_ab = corre_abm_point[corre_abm_point["A_B"] == ab]
    ab_point = MultiPoint([x for x in df_ab["ip_abm_point"]])
    print(ab_point)
    df_ab_line = cor_abm_link[corre_abm_link["A_B"] == ab]["ip_abm_line"].values[0]
    print(df_ab_line)
    split_line = get_linesegments(ab_point, df_ab_line)
    split_line = dumps(split_line)
    print(split_line)
    #split_line.append(split_line[0])
    trans.at[idx, "geometry"] = split_line
    
# need to do a filtering, there could be point that specific lie at the end, delete lines not multistring
trans = trans[trans["geometry"].str.contains("MULTI")]  
trans["geometry"] = trans["geometry"].str.replace("MULTILINESTRING ","").str.slice(1,-1)
trans_line = pd.concat([pd.Series(row['A_B'], row['geometry'].split('), (')) for _, row in trans.iterrows()]).reset_index()

trans_line.rename(columns = {"index":"geometry", 0:"A_B"}, inplace = True)
trans_line["geometry"] = trans_line["geometry"].str.replace(r"(","").str.replace(")","")
trans_line["geometry"] = "LineString (" + trans_line["geometry"] + ")"
trans_line_gdf = gpd.GeoDataFrame(columns = {"A_B","geometry"}).set_crs(epsg=2240)

for i in range(len(trans_line.index)):
    ab = trans_line.loc[i, "A_B"]
    geom = trans_line.loc[i,"geometry"]
    g = shapely.wkt.loads(geom)
    print(g)
    trans_line_gdf.loc[i,"A_B"] = ab
    trans_line_gdf.loc[i,"geometry"] = g
    
trans_line_gdf.to_file("test.shp")
ori_abm_link = abmLinks[~abmLinks["A_B"].isin(corre_abm_link["A_B"].unique().tolist())][["geometry"]]
ori_abm_link.to_file("test_ori.shp")


#%% This cell is another method for split the line, but it simply insert extra nodes to the existing linestrings, not recommended
# for i in range(len(ab_list)):
#     ab = cor_ab.loc[i,"A_B"]
#     df_ab = corre_abm_point[corre_abm_point["A_B"] == ab]
#     ab_point = MultiPoint([x for x in df_ab["ip_abm_point"]])
#     print(ab_point)
#     df_ab_line = cor_abm_link[corre_abm_link["A_B"] == ab]["ip_abm_line"].values[0]
#     print(df_ab_line)
#     split_line = snap(df_ab_line, ab_point, 0.0001)
#     print(split_line.wkt)
#     #split_line.append(split_line[0])
#     trans.loc[i, "A_B"] = ab
#     trans.loc[i, "geometry"] = split_line.wkt
    
  