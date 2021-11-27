# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 14:12:40 2021

@author: tpassmore6
"""

#%%import cell
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString, Polygon, MultiPoint
import os

#change this to where you stored this folder
#os.chdir('/Users/tannerpassmore/OneDrive - Georgia Institute of Technology/GD FALL/BikewaySim/Python Network Generation')
os.chdir(r'C:\Users\tpassmore6\Documents\BikewaySim\Python Network Generation')

#%% import networks
#first we want to import the road networks we filtered in the previous script
#our primary goal here is to figure out how best to combine the ABM network and the navteq network
#we suspect that navteq, basased the on the mileage, number of links, and nodes + all the other information provided makes it a good candidate
#for the base network of bikewaysim

#ABM Links
abm_linksfp = r'shapefiles/clipped_shapefiles/abm/abmLinks_studyArea_filtered.shp'
abmLinks = gpd.read_file(abm_linksfp).to_crs(epsg=2240)

#NAVTEQ
navteq_linksfp = r'shapefiles/clipped_shapefiles/navteq/navteqLinks_studyArea_filtered.shp'
navteqLinks = gpd.read_file(navteq_linksfp).to_crs(epsg=2240)

#OSM
osm_linksfp = r'shapefiles/clipped_shapefiles/osm/osmLinks_studyArea_filtered.shp'
osmLinks = gpd.read_file(osm_linksfp).to_crs(epsg=2240)


#%% Extract Start and End Points Code from a LineString as touples
def startNode(row, geom):
    return (row[geom].coords.xy[0][0], row[geom].coords.xy[1][0]) #basically look at x and then y coord, use apply to do this for every row of a dataframe

def endNode(row, geom):
    return (row[geom].coords.xy[0][-1], row[geom].coords.xy[1][-1])

#%% rename or create node ID's
def renameNodes(df, network_name, A, B):
    df = df.rename(columns={A:f'A_{network_name}',B:f'B_{network_name}'})
    return df

#%% rename or create node ID's

#this function will rename the node ID columns to follow the syntax used for ABM
def renameNodes(df, network_name, A, B):
    df = df.rename(columns={A:f'A_{network_name}',B:f'B_{network_name}'})
    return df

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
    matched_nodes = matched_nodes.filter([f'ID_{df1_name}', f'coords_{df1_name}', f'ID_{df2_name}' ], axis = 1)
    matched_nodes = matched_nodes.set_geometry(f'coords_{df1_name}')
    
    #check uniqueness
    #this makes sure that it was a one-to-one match
    QAcheck = matched_nodes.copy()
    QAcheck['unique'] = 1
    QAcheck = QAcheck.groupby([f'ID_{df2_name}'], as_index=False).count()
    QAcheck = QAcheck[QAcheck['unique']>1]
    print(f'There are {len(QAcheck.index)} {df1_name} nodes matching to the same {df2_name} node, not a one-to-one match')
    

    matched_nodes.to_file(rf'shapefiles/node_matching/{df2_name}_matched_to_{df1_name}_{append_name}.shp')
    return matched_nodes

def remainingNodes(matched_nodes, base_nodes, network_name, append_name):
    #unmatched nodes file 
    unmatched_nodes = base_nodes[-base_nodes[f'ID_{network_name}'].isin(matched_nodes[f'ID_{network_name}'])]
    print(f'There are {len(unmatched_nodes.index)} {network_name} nodes remaining')
    unmatched_nodes.to_file(rf'shapefiles/node_matching/umatched_{network_name}_{append_name}.shp')
    return unmatched_nodes

#%% matching

#Filter NAVTEQ
#This should make it so that the only NAVTEQ nodes that ABM can join to represent real intersections
navteqNodes_filt = navteqNodes[navteqNodes['num_links_navteq'] != 2 ].reset_index(drop=True)

matched_abm_navteqfilt = matchNodes(abmNodes, 'abm', navteqNodes_filt, 'navteq', 'navteq_filt', 26)
unmatched_nodes_abm_navteqfilt  = remainingNodes(matched_abm_navteqfilt, abmNodes, 'abm', 'navteq_filt')

#how many are ABM nodes that shouldn't match?
twonode = unmatched_nodes_abm_navteqfilt[unmatched_nodes_abm_navteqfilt['num_links_abm'] == 4]


#matched_abm_nofilt = matchNodes(abmNodes, 'abm', navteqNodes, 'navteq', 'no_filtering', 3)
#unmatched_nodes_abm_nofilt = remainingNodes(matched_abm_nofilt, abmNodes, 'abm', 'no_filtering')

#%% Using the number of links to filter nodes that don't represent real intersections

#Filter ABM
#abmNodes_filt = abmNodes[abmNodes['num_links_abm'] != 4 ].reset_index(drop=True) #
#abmNodes_filt = abmNodes_filt[abmNodes_filt['num_links_abm'] != 2 ].reset_index(drop=True) #

#matched_abm_abmfilt = matchNodes(abmNodes_filt, 'abm', navteqNodes, 'navteq', 'abm_filt', 3)
#unmatched_nodes_abm_abmfilt  = remainingNodes(matched_abm_abmfilt, abmNodes, 'abm', 'abm_filt')


#Filter NAVTEQ
#navteqNodes_filt = navteqNodes[navteqNodes['num_links_navteq'] != 2 ].reset_index(drop=True)

#matched_abm_navteqfilt = matchNodes(abmNodes, 'abm', navteqNodes_filt, 'navteq', 'navteq_filt', 3)
#unmatched_nodes_abm_navteqfilt  = remainingNodes(matched_abm_navteqfilt, abmNodes, 'abm', 'navteq_filt')


#Filter ABM and NAVTEQ
#matched_abm_bothfilt = matchNodes(abmNodes_filt, 'abm', navteqNodes_filt, 'navteq', 'both_filt', 3)
#unmatched_nodes_abm_bothfilt = remainingNodes(matched_abm_bothfilt, abmNodes, 'abm', 'both_filt')


