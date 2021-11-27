# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 22:49:14 2021

@author: tpassmore6
"""

#%% Study Area File Paths

bikewaysim_study_areafp = r'Base_Shapefiles/bikewaysim_study_area/bikewaysim_study_area.shp'
city_of_atlantafp = r'Base_Shapefiles/coa/Atlanta_City_Limits.shp'
atlanta_regional_commissionfp = r'Base_Shapefiles/arc/arc_bounds.shp'

#add new study areas if desired



#%% Import Study Areas

bikewaysim_study_area = importStudyArea(bikewaysim_study_areafp, 'study_area', write_to_file)
#city_of_atlanta = importStudyArea(city_of_atlantafp, 'coa', write_to_file)
#atlanta_regional_commission = importStudyArea(atlanta_regional_commissionfp, 'arc', write_to_file)

#add new study areas if desired 

#%% Network Data Filepaths

abmfp = r'Base_Shapefiles/arc/ABM2020-TIP20-2020-150kShapefiles-outputs.gdb'
navteqfp = r'Base_Shapefiles/navteq/Streets.shp'
osmfp = r'Base_Shapefiles/osm/osm_data_GA.geojson'
#rc_routesfp = r'Base_Shapefiles/gdot/rc_routes.geojson'

#add new networks if desired


#%% abm
    
abm = import_network(abmfp, 'abm', bikewaysim_study_area,'study_area', write_to_file, 'DAILY_LINK')    
abm = renameNodes(abm,'abm','A','B')
abm = create_a_b(abm,'abm')
abm_road = filter_to_roads(abm, 'abm', write_to_file)
abm_road_nodes = makeNodes(abm_road, 'abm', 'road', write_to_file)
ignore_fields(abm, 'abm')
abm_sumdata = summurize_network(abm,'abm')

#%% delete all abm variables
del [abm, abm_road, abm_road_nodes]

#%% navteq
navteq = import_network(navteqfp, 'navteq', bikewaysim_study_area,'study_area', write_to_file)  
navteq = renameNodes(navteq, 'navteq','REF_IN_ID','NREF_IN_ID') 
navteq = create_a_b(navteq, 'navteq')
navteq_road = filter_to_roads(navteq, 'navteq', write_to_file)
navteq_bike = filter_to_bike(navteq, 'navteq', write_to_file)
navteq_road_nodes = makeNodes(navteq_road,'navteq', 'road', write_to_file)
navteq_bike_nodes = makeNodes(navteq_bike,'navteq', 'bike', write_to_file)
ignore_fields(navteq, 'navteq')
navteq_sumdata = summurize_network(navteq, 'navteq')

#%% delete all navteq variables
del [navteq, navteq_road, navteq_bike, navteq_road_nodes]


#%% osm
osm = import_network(osmfp, 'osm', bikewaysim_study_area,'study_area', write_to_file)   
osm = createNodeIDs(osm,'osm')
osm = create_a_b(osm,'osm')
osm_road = filter_to_roads(osm, 'osm', write_to_file)
osm_bike = filter_to_bike(osm, 'osm', write_to_file)
osm_road_nodes = makeNodes(osm_road,'osm', 'road', write_to_file)
osm_bike_nodes = makeNodes(osm_bike,'osm', 'bike', write_to_file)
ignore_fields(osm,'osm')
osm_sumdata = summurize_network(osm, 'osm')

#%% for osm, make one for others later

fields = osm.columns[osm.isna().all()].tolist()

with open(f'Ignore_Fields\osm_ignore_empty.pkl', 'wb') as f:
        pickle.dump(fields, f)


#%% delete all osm variables
del [osm, osm_road, osm_bike, osm_road_nodes]

#%% rc_routes
#rc_routes = import_network(rc_routesfp, 'rc_routes', bikewaysim_study_area, 'study_area', write_to_file)
#rc_routes_road = filter_to_roads(rc_routes,'rc_routes', write_to_file)
#rc_routes_bike = filter_to_bike(rc_routes,'rc_routes', write_to_file)
#rc_routes_sumdata = summurize_network(rc_routes,'rc_routes')


#i think this is all spam

#match back to nodes_coords and then figure out how many unique nodes
nodes_coords_fin = pd.merge(nodes, matched_nodes, how='left', left_on = 'tup_coords_new_abm', right_on = 'tup_coords_new_abm',
                            suffixes = (None, 'new'))

#create new geometry column and add old geometry if new geometry is None
replace_if_empty1 = lambda row: row['coords_new_abm'] if row['ip_abm_point'] == None else row['coords_new_abm']

#create new id column and add old id if new geometry is NaN
replace_if_empty2 = lambda row: row['ID_abm'] if row['ID_new_abm'] == np.nan else str(row['ID_abm'])

#create new column and make that the primary geometry
nodes_coords_fin['geometry_fin'] = nodes_coords_fin.apply(replace_if_empty1, axis = 1)
nodes_coords_fin['ID_fin'] = nodes_coords_fin.apply(replace_if_empty2, axis = 1)
nodes_coords_fin = nodes_coords_fin.set_geometry('geometry_fin')

#drop old columns
#nodes_coords_fin = nodes_coords_fin.drop(columns = ['ID_abm','ID_abmnew', 'coords_new_abmnew','ip_abm_point', 'ID_new_abm', 'dist'])
#filter
nodes_coords_fin = nodes_coords_fin.filter(['geometry_fin','ID_fin','cor_ID_navteq','ID_new_abm'])


#redo A B columns in the df_links
df_links_new = pd.merge(df_links, matched_nodes, how='left', left_on='startNode', right_on = 'coords_new_abm1').rename(
    columns={'ID_new_abm':'A_new'}).drop(columns={'coords_new_abm1'})
df_links_new = pd.merge(df_links_new, matched_nodes, how='left', left_on='endNode', right_on = 'coords_new_abm1').rename(
    columns={'ID_new_abm':'B_new'}).drop(columns={'coords_new_abm1'})

df_links_new = df_links_new.drop(columns={'startNode','endNode',})












#rerun the matching process with navteq
matched_abm_navteq_new = matchNodes(new_abm_nodes, 'new_abm', navteq_road_nodes_filt, 'navteq', 26)

unmatched_nodes_abm  = remainingNodes(matched_abm_navteq_new, new_abm_nodes, 'new_abm')
unmatched_nodes_osm_osmfilt  = remainingNodes(matched_abm_navteq_new, navteq_road_nodes_filt, 'navteq')

#what's next:
    #match existing navteq node ids to abm
    #add in rest of navteq nodes
    #use that code at the bottom
    #make sure the exports work properly

#extract the nodes from the new ABM links and repeat the find nearest step to officially perform the join

#add in rest of links below

#next we 

def match_links(df1_nodes, df1_name, df2_links, df2_name, 'new_abm'):
    
    #drop unnecessary columns
    df1_nodes = df1_nodes.filter([f'ID_{df1_name}', f'ID_{df2_name}'], axis = 1)
    
    #add matched nodes to the df2_nodes links
    #start with matching the A nodes
    joining = df1_nodes.rename(columns={f'ID_{df2_name}':f'A_{df2_name}'}) #rename the df2 coords in df1 as the same as df2_links
    matched_links = pd.merge(df2_links, joining, how = 'left', on=[f'A_{df2_name}'])
    matched_links = matched_links.rename(columns={f'ID_{df1_name}':f'A_{df1_name}'})
    
    match the B nodes
    joining = df1_nodes.rename(columns={f'ID_{df2_name}':f'B_{df2_name}'}) #rename the df2 coords in df1 as the same as df2_links
    matched_links = pd.merge(matched_links, joining, how = 'left', on=[f'B_{df2_name}'])
    matched_links = matched_links.rename(columns={f'ID_{df1_name}':f'B_{df1_name}'})
    
    #Export matched_links
    matched_links = matched_links.drop(columns={'startNode','endNode'})
    
    #export
    
    return matched_links
    
 abm_v3 = match_ids(matched_abm_navteq_new, 'navteq', new_abm)
 
 
 

#%%
#test = osm_road.copy()
#split_osm = osm_bike.copy()
#split_osm['dissolve'] = 1

#split_osm_diss = split_osm.dissolve(by='dissolve') # dissolve, it didn't work if no inputs given
#split_osm_parts = pd.Series(split_osm_diss.iloc[0].geometry).tolist() # multipart to singleparts

#osm_split_links = gpd.GeoDataFrame({'geometry':split_osm_parts}, geometry='geometry').set_crs(epsg=2240)
#osm_split_links['original_length'] = osm_split_links.length

#og_osm = osm_bike.copy()
#og_osm['buffer'] = og_osm.buffer(1) #make a small buffer on OSM roads layer
#og_osm = og_osm.set_geometry('buffer')

#res_intersection = gpd.overlay(osm_split_links, og_osm, how='intersection')

#res_intersection['intersected_length'] = res_intersection.length
#res_intersection['percent_overlap'] =  res_intersection['intersected_length'] / res_intersection['original_length']

#res_intersection_filt = res_intersection[res_intersection['percent_overlap'] >= 0.99]

#test_links = res_intersection_filt

#test_links.to_file(r'C:\Users\tpassmore6\Documents\GitHub\BikewaySim_Network_Processing\Processed_Shapefiles\osm_troubleshooting\test_links_bike.geojson', driver = 'GeoJSON')

#next, get coordinates and then put this in the network_import section

#testIntsPt = [list(part.coords) for part in testParts] # make list of endpoints
#testIntsPt = [pt for sublist in testIntsPt for pt in sublist] # flatten list of lists
# Filter to only retain points that appear more than twice (a real intersection instead of a turning point) 
#testIntsPt = pd.DataFrame.from_dict(Counter(testIntsPt), orient='index').reset_index()
#testIntsPt = testIntsPt[testIntsPt[0] >= 3]['index'].tolist()

#test_export = gpd.GeoDataFrame({'point_coords':testIntsPt})
#convert_to_points = lambda x: Point(x['point_coords'])
#test_export['geometry'] = test_export.apply(convert_to_points, axis = 1)

#test_export = test_export.drop(columns={'point_coords'}).set_geometry('geometry')
#test_export.to_file(r'C:\Users\tpassmore6\Documents\GitHub\BikewaySim_Network_Processing\Processed_Shapefiles\osm_troubleshooting\test_nodes_bike.geojson', driver = 'GeoJSON')

