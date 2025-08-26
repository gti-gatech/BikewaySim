
# Export calibration network to QGIS

# # drop cols we don't need
# fwd_links = links[links['reverse_link']==False].set_index('linkid')
# rev_links = links[links['reverse_link']==True].set_index('linkid')
# merged = pd.merge(fwd_links,rev_links,left_index=True,right_index=True,how='outer')

# cols = set([x.removesuffix('_x').removesuffix('_y') for x in merged.columns if ('_x' in x) | ('_y' in x)])

# # Function to condense two columns
# def condense_columns(col1, col2):
#     if pd.isna(col1):  # if col1 is NaN
#         return col2
#     elif pd.isna(col2):  # if col2 is NaN
#         return col1
#     elif col1 == col2:  # if values are equal
#         return col1
#     else:  # if values are different and neither is NaN
#         return str([col1, col2])
    
# new_cols = {}

# for col in tqdm(cols):
#     new_cols[col] = [condense_columns(col1,col2) for col1, col2 in merged[[col+'_x',col+'_y']].values]

# undirected_links = pd.DataFrame.from_dict(new_cols,orient='columns')
# undirected_links.index = merged.index
# undirected_links.reset_index(inplace=True)
# undirected_links = gpd.GeoDataFrame(undirected_links,crs=config['projected_crs_epsg'])
# order_cols = ['linkid', 'osmid', 'link_type', 'oneway', 'highway', 'name', 'all_tags',
#               'lanes', 'speed', 'AADT', 'ascent_grade_cat', 'facility_fwd', 'year',
#               '2lpd','3+lpd',
#               '(30,40] mph', '(40,inf) mph', '(30,inf) mph',
#               '[4k,10k) aadt', '[10k,inf) aadt', 
#               '[4,6) grade', '[6,inf) grade',
#               'bike lane', 'cycletrack', 'multi use path',
#               'bike lane report', 'multi use path report','lanes report','above_4 report',
#               'gdot_base','new_base',
#               'travel_time_min', 'geometry']
# undirected_links[order_cols].to_file(config['calibration_fp']/'calibration_network.gpkg',layer='final')