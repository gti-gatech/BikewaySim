# -*- coding: utf-8 -*-
"""
Created on Wed May  5 16:54:50 2021

@author: tpassmore6
"""
#%%
import geopandas as gpd
from shapely.geometry import Point, LineString
import pandas as pd
from collections import Counter

testRoad = gpd.read_file(r'C:/Users/tpassmore6/Documents/GitHub/BikewaySim_Network_Processing/Processed_Shapefiles/osm/osm_study_area_road.geojson')
network_name = 'osm'
link_type = 'road'
write_to_file = True

#%% Extract Start and End Points Code from a LineString as tuples
def startNode(row, geom):
    return (round(row[geom].coords.xy[0][0], 6), round(row[geom].coords.xy[1][0], 6)) #basically look at x and then y coord, use apply to do this for every row of a dataframe

def endNode(row, geom):
    return (round(row[geom].coords.xy[0][-1], 6), round(row[geom].coords.xy[1][-1], 6))


df = testRoad

#add start and end coordinates to each line
#this is so we can match them  
df['startNode'] = df.apply(startNode, geom= df.geometry.name, axis=1)
df['endNode'] = df.apply(endNode, geom= df.geometry.name, axis=1)

 
#this function will number each node to give it an ID
nodes_coords = df['startNode'].append(df['endNode'])
 
#create dataframe
nodes = pd.DataFrame({f'coords_{network_name}':nodes_coords})
 
#find number of intersecting links
nodes[f'num_links_{network_name}'] = 1 #this column will be used to count the number of links
 
#for each unique ID and coordinate, count number of links
nodes = nodes.groupby([f'coords_{network_name}'], as_index=False).count()




#%%
testRoad['dissolve'] = 1
testDiss = testRoad.dissolve(by='dissolve') # dissolve
testParts = pd.Series(testDiss.iloc[0].geometry).tolist() # multipart to singleparts

#%%


#%% Create Nodes
testIntsPt = [list(part.coords) for part in testParts] # make list of endpoints
testIntsPt = [pt for sublist in testIntsPt for pt in sublist] # flatten list of lists
# Filter to only retain points that appear more than twice (a real intersection instead of a turning point) 
testIntsPt = pd.DataFrame.from_dict(Counter(testIntsPt), orient='index').reset_index()
testIntsPt = testIntsPt[testIntsPt[0] >= 2]['index'].tolist()

#create node gdf
node_id = list(range(1,len(testIntsPt)+1))
nodes = gpd.GeoDataFrame({'ID_new':node_id, 'coords':testIntsPt})
nodes['coords'] = nodes['coords'].apply(lambda x: Point(x))
nodes = nodes.set_geometry('coords')

nodes.to_file(f'C:/Users/tpassmore6/Documents/GitHub/BikewaySim_Network_Processing/Processed_Shapefiles/osm_troubleshooting/testing_nodes.geojson', driver = 'GeoJSON')

#%% Create Links

gdf = gpd.GeoDataFrame({'geometry':testParts}, geometry='geometry').set_crs(epsg=2240)

#now need to join attribute information back in
gdf['original_length'] = gdf.length

#create buffer
testRoad['buffer'] = testRoad.buffer(0.01) #make a small buffer around the original layer
testRoad = testRoad.set_geometry('buffer') #make the buffer the primary geometry

#perform intersection
res_intersection = gpd.overlay(gdf, testRoad, how='intersection')
res_intersection['intersected_length'] = res_intersection.length
res_intersection['percent_overlap'] =  res_intersection['intersected_length'] / res_intersection['original_length']

res_intersection_filt = res_intersection[res_intersection['percent_overlap'] >= 1]

#rename old node id columns
#res_intersection_filt = res_intersection_filt.rename(columns={f'A_{network_name}':f'A_{network_name}_og',
 #                                                        f'B_{network_name}':f'B_{network_name}_og',
  #                                                       f'A_B_{network_name}':f'A_B_{network_name}_og'
   #                                                      })

res_intersection_filt = res_intersection_filt.explode().droplevel(level=1)


#add new node ID's to links
#add start and end coordinates to each line
#this is so we can match them  
#res_intersection_filt['startNode'] = res_intersection_filt.apply(startNode, geom= res_intersection_filt.geometry.name, axis=1)
#res_intersection_filt['endNode'] = res_intersection_filt.apply(endNode, geom= res_intersection_filt.geometry.name, axis=1)


#this function will number each node to give it an ID
#nodes_coords = res_intersection_filt['startNode'].append(res_intersection_filt['endNode'])







#make sure it's singlepart geos
#res_intersection_filt = res_intersection.explode()

#need to create new node ids for splitted lines, but keep old ones
#gdf = createNodeIDs(res_intersection_filt, network_name)

#%%
#writing
if write_to_file == True:
    res_intersection_filt.to_file(f'C:/Users/tpassmore6/Documents/GitHub/BikewaySim_Network_Processing/Processed_Shapefiles/osm_troubleshooting/testing3.geojson', driver = 'GeoJSON')
