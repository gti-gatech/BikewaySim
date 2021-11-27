# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 12:47:38 2021

@author: tpassmore6
"""

#%%import cell
import geopandas as gpd
import fiona
import os

#change this to where you stored this folder
os.chdir(r'C:\Users\tpassmore6\Documents\BikewaySim\Python Network Generation')


#%% Find the starting and ending node coordinates for a linestring
#use this to find the starting and ending nodes for each network link
def startNode(row, geom):
    return (row[geom].coords.xy[0][0], row[geom].coords.xy[1][0]) #for each row in the geom column, access the xy coordinates. [0=x coord, 1=y coord][0 means first node]

def endNode(row, geom):
    return (row[geom].coords.xy[0][-1], row[geom].coords.xy[1][-1]) #the [-1] means the last node


#%%Import Study Area
studyAreafp = r'shapefiles/bikewaysim_study_area.shp'
studyArea = gpd.read_file(studyAreafp).to_crs(epsg=2240)
studyArea.to_file(r'shapefiles/clipped_shapefiles/bikewaysim_study_area.shp')

#study area area
studyArea['geometry'].area / 5280**2

#%%City of Atlanta Area
coaLimitsfp = r'shapefiles/CoA/Atlanta_City_Limits.shp'
coaLimits = gpd.read_file(coaLimitsfp).to_crs(epsg=2240)
coaLimits.to_file(r'shapefiles/clipped_shapefiles/coaLimits.shp')

#study area area
coaLimits['geometry'].area / 5280**2

#%%ARC Metro Area
arcAreafp = r'shapefiles/ARC/Regional_Commission_Boundaries.shp'
arcArea = gpd.read_file(arcAreafp).to_crs(epsg=2240)
arcArea = arcArea[arcArea['OBJECTID'] == 1] #select only the ARC region
arcArea.to_file(r'shapefiles/clipped_shapefiles/arcArea.shp')

#study area area
arcArea['geometry'].area / 5280**2

#%%Clayton County & Gwinnett County

countyfp = r'shapefiles/Counties/f4fc19b4-1edb-49ed-aa65-9718bb7a8fc42020313-1-91kw5t.j6v2o.shp'
counties = gpd.read_file(countyfp).to_crs(epsg=2240)

clayton = counties[counties['Name'] == 'Clayton'] #select clayton county
clayton.to_file(r'shapefiles/clipped_shapefiles/clayton.shp')
clayton['geometry'].area / 5280**2

gwinnett = counties[counties['Name'] == 'Gwinnett'] #select gwinnett county
gwinnett.to_file(r'shapefiles/clipped_shapefiles/gwinnett.shp')
gwinnett['geometry'].area / 5280**2

#%%Import Network Data and Clip to Study Area

#put in file path, network name, area to clip by, and the name of the clip area, use last parameter to identify desired GDB layer
def importNetwork(fp, network_name, clip_area, clip_area_name, gdb_layer = 0): 

    links = gpd.read_file(fp, mask = clip_area, layer = gdb_layer).to_crs(epsg=2240) #import the study area
    links['length_mi'] = links.geometry.length / 5280 # create a new distance column and calculate milage of each link
    
    #create a new folder for the network if it doesn't already exist
    if not os.path.exists(f'shapefiles\clipped_shapefiles\{network_name}'):
        os.makedirs(f'shapefiles\clipped_shapefiles\{network_name}')
    
    #export links to shapefile with network name and clip area name
    links.to_file(rf"shapefiles\clipped_shapefiles\{network_name}\{network_name}Links_{clip_area_name}.shp")
    
    #export links to CSV with network name and clip area name
    links.to_csv(rf"shapefiles\clipped_shapefiles\{network_name}\{network_name}Links_{clip_area_name}.csv") #export links to CSV
   
    return links

#%%Import network Links and clip by several masking areas, this step takes awhile
#make sure to identify the right shapefile/geodabase in the filepath
clip_areas = {
    'studyArea': studyArea
    #'coaLimits': coaLimits,
    #'arcArea': arcArea,
    #'clayton': clayton,
    #'gwinnett': gwinnett
    }


osmLinksfp = r'shapefiles/OSM_GA/gis_osm_roads_free_1.shp'
osmDict = {}
#osmLinks, osmLinks_source = importNetwork(osmLinksfp,'osm', masking_area)
for x in clip_areas:
    osmDict[f'osmLinks_{x}'] = importNetwork(osmLinksfp, 'osm', clip_areas[x], x)


#old abm links
#abmLinksfp = r'shapefiles/ABM/2020_links.shp'
#abmLinks, abmLinks_source = importNetwork(abmLinksfp,'abm', masking_area)
#abmLinks = importNetwork(abmLinksfp,'abm', masking_area)

abmLinksNEWfp = r'shapefiles/ABM/ABM2020-TIP20-2020/Output File/ABM2020-TIP20-2020-150kShapefiles-outputs.gdb'
abmDict = {}
for x in clip_areas:
    abmDict[f'abmLinks_{x}'] = importNetwork(abmLinksNEWfp, 'abm', clip_areas[x], x, 'DAILY_LINK').explode()

#used this to examine all the layers without masking
#abmDict = {} # create a dictionary to contain all the abm links
#for x in range(0,len(fiona.listlayers(abmLinksNEWfp))): #loop through each layer of the geodatabase and export to a table
#    abmDict[fiona.listlayers(abmLinksNEWfp)[x]] = gpd.read_file(abmLinksNEWfp, layer = 'DAILY_LINK')

#just abm
abm_links_all = gpd.read_file(abmLinksNEWfp, layer = 'DAILY_LINK')
abm_nodes_all = gpd.read_file(abmLinksNEWfp, layer = 'DAILY_NODE')


#use if you need the other ABM layers
#abmDict = {} # create a dictionary to contain all the abm links
#for x in range(0,len(fiona.listlayers(abmLinksNEWfp))): #loop through each layer of the geodatabase and export to a table
#    abmDict[fiona.listlayers(abmLinksNEWfp)[x]] = importNetwork(abmLinksNEWfp,f'abm_{fiona.listlayers(abmLinksNEWfp)[x]}', masking_area, x)




navteqLinksfp = r'C:\Users\tpassmore6\Documents\ArcGIS\NAVTEQ\Streets.shp'
navteqDict = {}
for x in clip_areas:
    navteqDict[f'navteqLinks_{x}'] = importNetwork(navteqLinksfp, 'navteq', clip_areas[x], x)

#navteqLinks, navteqLinks_source = importNetwork(navteqLinksfp,'navteq', masking_area)



#%%Processing Steps
#remove anything that doesn't need to be in the network
#the network dataset will need to be examined in order to do this step. Although, for OSM, TIGER, and NAVTEQ the code used below should be reusable

#%%OSM Filter

def osmFilter(osmLinks, name ):

    osmLinksFiltered = osmLinks.copy()
    #only have roads
    osmLinksFiltered = osmLinksFiltered[ (osmLinksFiltered['fclass'].str.contains('primary')) |
                            (osmLinksFiltered['fclass'].str.contains('primary_link')) |
                            (osmLinksFiltered['fclass'].str.contains('residential')) |
                            (osmLinksFiltered['fclass'].str.contains('secondary')) |
                            (osmLinksFiltered['fclass'].str.contains('secondary_link')) |
                            (osmLinksFiltered['fclass'].str.contains('tertiary')) |
                            (osmLinksFiltered['fclass'].str.contains('tertiary_link')) |
                            (osmLinksFiltered['fclass'].str.contains('primary_link')) |
                            (osmLinksFiltered['fclass'].str.contains('trunk')) |
                            (osmLinksFiltered['fclass'].str.contains('trunk_link')) ]
    #export to file
    osmLinksFiltered.to_file(rf'shapefiles/clipped_shapefiles/osm/{name}_filtered.shp')

    
    
    #osmBikeLinks
    osmBikeLinks = osmLinks.copy()
    #only have bike specific links
    osmBikeLinks = osmBikeLinks[ (osmLinks['fclass'].str.contains('cycleway')) |
                            (osmLinks['fclass'].str.contains('footway')) |
                            (osmLinks['fclass'].str.contains('path')) |
                            (osmLinks['fclass'].str.contains('pedestrian')) |
                            (osmLinks['fclass'].str.contains('steps')) |
                            (osmLinks['fclass'].str.contains('tertiary')) ]
    #export filtered osm bike links
    osmBikeLinks.to_file(rf'shapefiles/clipped_shapefiles/osm/{name}_bike.shp')
    
    
    return osmLinksFiltered, osmBikeLinks

osmFilteredDict = {}
osmBikeDict = {}

for x in osmDict:
    osmFilteredDict[f'{x}_filtered'], osmBikeDict[f'{x}_bike'] = osmFilter(osmDict[x], x)


#%% ABM Filter

def abmFilter(abmLinks, name):

    abmLinksFiltered = abmLinks.copy()
    #only inlcude arterials and collectors
    abmLinksFiltered = abmLinksFiltered[ (abmLinksFiltered['FACTYPE'] == 10) |
                                (abmLinksFiltered['FACTYPE'] == 11) |
                                (abmLinksFiltered['FACTYPE'] == 14) ]
    
    abmLinksFiltered.to_file(rf'shapefiles/clipped_shapefiles/abm/{name}_filtered.shp')
    
    return abmLinksFiltered

abmFilteredDict = {}
for x in abmDict:
    abmFilteredDict[f'{x}_filtered'] = abmFilter(abmDict[x], x)

#%%Processing Steps for NAVTEQ

def navteqFilter(navteqLinks, name):
    navteqLinksFiltered = navteqLinks.copy()
    
    #remove controlled access roads
    navteqLinksFiltered = navteqLinksFiltered[ navteqLinksFiltered['CONTRACC'].str.contains('N') ]
    
    #remove slowest speed roads
    navteqLinksFiltered = navteqLinksFiltered[ -navteqLinksFiltered['SPEED_CAT'].str.contains('8') ]
    
    #remove roads that don't allow autos
    navteqLinksFiltered = navteqLinksFiltered[ navteqLinksFiltered['AR_AUTO'].str.contains('Y') ]
    
    #remove interstate ramps
    navteqLinksFiltered = navteqLinksFiltered[ navteqLinksFiltered['RAMP'].str.contains('N') ]
    
    navteqLinksFiltered.to_file(rf'C:\Users\tpassmore6\Documents\ArcGIS\bikewaysim\clipped_shapefiles\navteq\{name}_filtered.shp')

    #NAVTEQ Bike Links
    #service roads/non-auto paths
    navteqBikeLinks = navteqLinks.copy()
    #only keep links where autos not allowed
    navteqBikeLinks = navteqBikeLinks[ navteqBikeLinks['AR_AUTO'].str.contains('N') ]
    navteqBikeLinks.to_file(rf'shapefiles/clipped_shapefiles/abm\{name}_bike.shp')
    
    return navteqLinksFiltered, navteqBikeLinks

navteqFilteredDict = {}
navteqBikeDict = {}

for x in navteqDict:
    navteqFilteredDict[f'{x}_filtered'], navteqBikeDict[f'{x}_bike'] = navteqFilter(navteqDict[x], x)

    
#%% General Summary Function

def linkStats(df):
    
    df['startNode'] = df.apply(startNode, geom='geometry', axis=1)
    df['endNode'] = df.apply(endNode, geom='geometry', axis=1)


    startNodeEndNode = df['startNode'].append(df['endNode']).nunique()
    
    sum_data = { 'num of links': len(df.index),       
        'total length mi': df['length_mi'].sum(),
         'average length of link mi': df['length_mi'].mean(),
         'number of nodes': startNodeEndNode
        }
    
    return sum_data

#%% Summaries

osmDict_sumdata = {}
for x in osmDict:
    osmDict_sumdata[f'{x}_sum'] = linkStats(osmDict[x])
    
osmFilteredDict_sumdata = {}
for x in osmFilteredDict:
    osmFilteredDict_sumdata[f'{x}_sum'] = linkStats(osmFilteredDict[x])


abmDict_sumdata = {}
for x in abmDict:
    abmDict_sumdata[f'{x}_sum'] = linkStats(abmDict[x])

abmFilteredDict_sumdata = {}
for x in abmFilteredDict:
    abmFilteredDict_sumdata[f'{x}_sum'] = linkStats(abmFilteredDict[x])
    

navteqDict_sumdata = {}
for x in navteqDict:
    navteqDict_sumdata[f'{x}_sum'] = linkStats(navteqDict[x])
    
navteqFilteredDict_sumdata = {}
for x in navteqFilteredDict:
    navteqFilteredDict_sumdata[f'{x}_sum'] = linkStats(navteqFilteredDict[x])

