# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 17:26:03 2021

@author: tpassmore6
"""

#%% run attribute comparisons
from attribute_summary import * 

abm_road = import_geojson('abm', 'study_area', 'road')
osm_road = import_geojson('osm','study_area','road')
here_road = import_geojson('here', 'study_area', 'road')

#filter to only functional class, lane, and speed limit columns
abm_filt = ['NAME','FACTYPE', 'FCLASS', 'SPEED', 'SPEEDLIMIT', 'LANESTOTAL', 'LANES', 'geometry']
osm_filt = ['name', 'highway', 'maxspeed', 'lanes', 'lanes:backward', 'lanes:forward', 'geometry']
here_filt = ['ST_NAME','FUNC_CLASS', 'SPEED_CAT', 'LANE_CAT', 'PHYS_LANES' , 'TO_LANES', 'FROM_LANES', 'geometry']

abm_road = abm_road[abm_filt]
osm_road = osm_road[osm_filt]
here_road = here_road[here_filt]

#%% estimate combination of networks

abm_base = import_geojson('abm', 'study_area', 'base')
osm_road = import_geojson('osm', 'study_area', 'base')
here_base = import_geojson('here', 'study_area', 'base')

#start with ABM, find HERE links that don't overlap
abm_base['buffer'] = abm_base.buffer(50)
abm_base = abm_base.set_geometry('buffer')

#create distance col, intersect with ABM to find difference
here_diff = gpd.overlay(here_base, abm_base, how='difference')

#add difference to abm_base and then buffer again
abm_base = abm_base.set_geometry('geometry')
here_and_abm = gpd.overlay(abm_base, here_diff, how = 'union')
here_and_abm['buffer'] = here_and_abm.buffer(50)
here_and_abm = here_and_abm.set_geometry('buffer')

#find difference between HERE+ABM and OSM
osm_diff = gpd.overlay(osm_road, here_and_abm, how = 'difference')

#add difference to ABM+HERE
here_and_abm = here_and_abm.set_geometry('geometry')
osm_here_abm = gpd.overlay(osm_diff, here_and_abm, how = 'union')

def start_node(row, geom):
   return (row[geom].coords.xy[0][0], row[geom].coords.xy[1][0]) #basically look at x and then y coord, use apply to do this for every row of a dataframe

def end_node(row, geom):
   return (row[geom].coords.xy[0][-1], row[geom].coords.xy[1][-1])

def return_nodes_and_links(links):
    print(links.length.sum() / 5280)
    links = links.explode()
    print(len(links))
    links['start_node'] = links.apply(start_node, geom= links.geometry.name, axis=1)
    links['end_node'] = links.apply(end_node, geom= links.geometry.name, axis=1)
    nodes = links['start_node'].append(links['end_node']).drop_duplicates()
    print(len(nodes))

#report results
return_nodes_and_links(osm_here_abm)

#%% 

# =============================================================================
# # #speed cats from HERE
# # # =============================================================================
# # # 1: > 80 MPH
# # # 2: 65-80 MPH
# # # 3: 55-64 MPH
# # # 4: 41-54 MPH
# # # 5: 31-40 MPH
# # # 6: 21-30 MPH
# # # 7: 6-20 MPH
# # # 8: < 6 MPH
# # # =============================================================================
# 
# #speed bins
# osm_speed_bins = {
#      '25 mph': 7,
#      '35 mph': 5,
#      '30 mph': 7,
#      '55 mph': 3,
#      '15 mph': 7,
#      '5 mph': 8,
#      '2 mph': 8,
#      '10 mph': 7   
#      }
# 
# osm_road = osm_road.replace({'maxspeed': osm_speed_bins})
# 
# abm_speed_bins = {
#     35: 5,
#     30: 6,
#     25: 6,
#     40: 5,
#     15: 7,
#     0: 8
#     }
# 
# abm_road = abm_road.replace({'SPEEDLIMIT':abm_speed_bins})
# =============================================================================

#%% lanes

# =============================================================================
# here uses per direction
# 1 = one lane
# 2 = two or three lanes
# 3 = four or more lanes
# =============================================================================
# =============================================================================
# 
# osm_lane_bins = {
#     1: 1,
#     2: 1,
#     3: 2,
#     4: 2,
#     5: 2,
#     6: 3
#     }
# 
# 
# =============================================================================
#%% abm

abm_road['distance'] = abm_road.length / 5280

#completion rate
abm_completion_lanes = abm_road[abm_road['LANESTOTAL'].isna() == False]['distance'].sum() / abm_road['distance'].sum()
abm_completion_speed = abm_road[abm_road['SPEEDLIMIT'].isna() == False]['distance'].sum() / abm_road['distance'].sum()

abm_groupby_lanes = abm_road.groupby(by=['LANESTOTAL']).sum()['distance']
abm_groupby_speed = abm_road.groupby(by=['SPEEDLIMIT']).sum()['distance']

print(abm_groupby_lanes)
print(abm_road.length.sum() / 5280)
#abm_sum = create_sum_table(abm_road)
#print(abm_sum.loc['LANES'])
#print(abm_sum.loc['SPEEDLIMIT'])



#%%intersect with 


def buffer_and_intersect(base_network, base_name, joining_network, joining_name, tolerance):

    #dissolve by

    #buffer base network
    base_network['buffer'] = base_network.buffer(tolerance)
    
    #set buffer to active geometry
    base_network = base_network.set_geometry('buffer')
    
    #intersect with joining network
    intersection = gpd.overlay(joining_network, base_network, how='intersection')
    
    #write to file
    #intersection.to_file('test1.geojson', driver = 'GeoJSON')
    
    return intersection

osm_intersect = buffer_and_intersect(abm_road, 'abm', osm_road, 'osm', 10)

#%%

#create abm buffer
here_road['buffer'] = here_road.buffer(10)

#set
here_road = here_road.set_geometry('buffer')

#intersect with here
abm_intersect = gpd.overlay(abm_road, here_road, how='intersection')
abm_intersect['distance'] = abm_intersect.length / 5280

#total intersection distance
tot_dist1 = abm_intersect.length.sum() / 5280

#report values
here_groupby_lanes = abm_intersect.groupby(by=['LANE_CAT']).sum()['distance']
here_groupby_speed = abm_intersect.groupby(by=['SPEED_CAT']).sum()['distance']

abm_groupby_lanes = abm_intersect.groupby(by=['LANE_CAT','LANESTOTAL']).sum()['distance']
abm_groupby_speed = abm_intersect.groupby(by=['SPEED_CAT','SPEEDLIMIT']).sum()['distance']

#lol
abm_groupby_lanes / here_groupby_lanes *100


#%%

#create abm buffer
here_road['buffer'] = here_road.buffer(10)

#set
here_road = here_road.set_geometry('buffer')


#intersect with osm
osm_intersect = gpd.overlay(osm_road, here_road, how='intersection')
osm_intersect['distance'] = osm_intersect.length / 5280

#total intersection distance
tot_dist = osm_intersect.length.sum() / 5280

#report values
here_groupby_lanes = osm_intersect.groupby(by=['LANE_CAT'], dropna=False).sum()['distance']
here_groupby_speed = osm_intersect.groupby(by=['SPEED_CAT'], dropna=False).sum()['distance']

#reprot values
osm_groupby_lanes = osm_intersect.groupby(by=['LANE_CAT','lanes'], dropna=False).sum()['distance']
osm_groupby_speed = osm_intersect.groupby(by=['SPEED_CAT','maxspeed'], dropna=False).sum()['distance']

print(osm_groupby_lanes / here_groupby_lanes *100)
print(osm_groupby_lanes / here_groupby_lanes * 100)

#%%



#intersect here with osm
osm_intersect = gpd.overlay(osm_road, here_intersect)



#find link distance
osm_intersect['distance'] = osm_intersect.length / 5280

#completion rate
osm_completion_lanes = osm_intersect[osm_intersect['lanes'].isna() == False]['distance'].sum() / osm_intersect['distance'].sum()
osm_completion_speed = osm_intersect[osm_intersect['maxspeed'].isna() == False]['distance'].sum() / osm_intersect['distance'].sum()

#group by number of lanes
osm_groupby_lanes = osm_intersect.groupby(by=['lanes']).sum()['distance']
osm_groupby_speed = osm_intersect.groupby(by=['maxspeed']).sum()['distance']

osm_sum = create_sum_table(osm_intersect)
#print(osm_sum.loc['lanes'])
#print(osm_sum.loc['maxspeed'])

print(osm_groupby_speed)
print(osm_intersect.length.sum() / 5280)

print(osm_groupby_lanes)

#%%

here_intersect = buffer_and_intersect(abm_road, 'abm', here_road, 'here', 10)

#find link distance
here_intersect['distance'] = here_intersect.length / 5280

#completion rate
here_completion_lanes = here_intersect[here_intersect['LANE_CAT'].isna() == False]['distance'].sum() / here_intersect['distance'].sum()
here_completion_speed = here_intersect[here_intersect['SPEED_CAT'].isna() == False]['distance'].sum() / here_intersect['distance'].sum()

#group by number of lanes
#here_groupby_lanes = here_intersect.groupby(by=['PHYS_LANES']).sum()['distance']
here_groupby_lanes = here_intersect.groupby(by=['LANE_CAT']).sum()['distance']
here_groupby_speed = here_intersect.groupby(by=['SPEED_CAT']).sum()['distance']

#here_sum = create_sum_table(here_intersect)
#print(here_sum.loc['LANE_CAT'])
#print(here_sum.loc['SPEED_CAT'])

print(here_groupby_lanes)

#%% HERE and OSM

#find link distance
here_road['distance'] = here_road.length / 5280

#completion rate
here_completion_lanes = here_road[here_road['LANE_CAT'].isna() == False]['distance'].sum() / here_road['distance'].sum()
here_completion_speed = here_road[here_road['SPEED_CAT'].isna() == False]['distance'].sum() / here_road['distance'].sum()

#group by number of lanes
#here_groupby_lanes = here.groupby(by=['PHYS_LANES']).sum()['distance']
here_groupby_lanes = here_road.groupby(by=['LANE_CAT']).sum()['distance']
here_groupby_speed = here_road.groupby(by=['SPEED_CAT']).sum()['distance']

#here_sum = create_sum_table(here_road)
#print(here_sum.loc['LANE_CAT'])
#print(here_sum.loc['SPEED_CAT'])

print(here_groupby_speed)
print(here_road.length.sum() / 5280)

print(here_groupby_lanes)
#%%

osm_intersect_here = buffer_and_intersect(here_road, 'here', osm_road, 'osm', 10)

#find link distance
osm_intersect_here['distance'] = osm_intersect_here.length / 5280

#completion rate
osm_completion_lanes = osm_intersect_here[osm_intersect_here['lanes'].isna() == False]['distance'].sum() / osm_intersect_here['distance'].sum()
osm_completion_speed = osm_intersect_here[osm_intersect_here['maxspeed'].isna() == False]['distance'].sum() / osm_intersect_here['distance'].sum()

#group by number of lanes
osm_groupby_lanes = osm_intersect_here.groupby(by=['lanes']).sum()['distance']
osm_groupby_speed = osm_intersect_here.groupby(by=['maxspeed']).sum()['distance']

osm_sum = create_sum_table(osm_intersect_here)
#print(osm_sum.loc['lanes'])
#print(osm_sum.loc['maxspeed'])


print(osm_groupby_speed)
print(osm_intersect_here.length.sum() / 5280)

print(osm_groupby_lanes)