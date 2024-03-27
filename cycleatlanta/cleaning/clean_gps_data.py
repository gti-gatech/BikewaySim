# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 20:51:47 2023

@author: tpassmore6
"""

#imports
import pandas as pd
import geopandas as gpd
import datetime
pd.options.mode.chained_assignment = None  # default='warn'
import pickle
from pathlib import Path
from tqdm import tqdm

def calculate_coordinate_metrics(coords):
    
    #find acceleration from change in GPS speed
    coords.loc[:,'acceleration_ft/s**2'] = coords['speed_mph'].diff()
    
    #find time and euclidean distance between successive points
    coords.loc[:,'delta_time'] = coords['datetime'].diff() #/ datetime.timedelta(seconds=1)
    coords.loc[:,'delta_distance_ft'] = coords.distance(coords.shift(1))

    #get cumulative versions
    coords.loc[:,'traversed_distance_ft'] = coords['delta_distance_ft'].cumsum()
    coords.loc[:,'time_elapsed'] = (coords['datetime'] - coords['datetime'].min()) #/ datetime.timedelta(seconds=1)
    
    #calculate speed in mph
    feet_per_mile = 5280
    seconds_per_hour = 3600
    coords.loc[:,'calculated_speed_mph'] = (coords['delta_distance_ft'] / feet_per_mile) / ((coords['delta_time'] / datetime.timedelta(seconds=1)) * seconds_per_hour)

    #attach sequence and total points for easier GIS examination
    coords['sequence'] = range(0,coords.shape[0])

    return coords

def calculate_trip_metrics(key,coords,trips_df):
    # total points and duration
    trips_df.at[trips_df['tripid'] == key, 'total_points'] = coords.shape[0]
    trips_df.at[trips_df['tripid'] == key, 'duration'] = coords['datetime'].max() - coords['datetime'].min()

    # time
    #trips_df.at[trips_df['tripid'] == key, 'min_delta_time'] = coords['delta_time'].min()
    trips_df.at[trips_df['tripid'] == key, 'max_delta_time'] = coords['delta_time'].max()
    trips_df.at[trips_df['tripid'] == key, 'mean_delta_time'] = coords['delta_time'].mean()
    
    # distance
    #trips_df.at[trips_df['tripid'] == key, 'min_distance_ft'] = coords['delta_distance_ft'].min()
    trips_df.at[trips_df['tripid'] == key, 'max_distance_ft'] = coords['delta_distance_ft'].max()
    trips_df.at[trips_df['tripid'] == key, 'avg_distance_ft'] = coords['delta_distance_ft'].mean()
    trips_df.at[trips_df['tripid'] == key, 'total_distance_ft'] = coords['traversed_distance_ft'].max()
    
    #distance between first and last point
    first_point = coords.at[coords['datetime'].idxmin(),'geometry']
    last_point = coords.at[coords['datetime'].idxmax(),'geometry']
    trips_df.at[trips_df['tripid'] == key, 'first_to_last_ft'] = first_point.distance(last_point)

    # speed
    trips_df.at[trips_df['tripid'] == key, 'max_speed_mph'] = coords['speed_mph'].max()  
    trips_df.at[trips_df['tripid'] == key, 'min_speed_mph'] = coords['speed_mph'].min()
    trips_df.at[trips_df['tripid'] == key, 'avg_speed_mph'] = coords['speed_mph'].mean()
    
    return trips_df


#%% import data

export_fp = Path.home() / 'Documents/BikewaySimData/Projects/gdot/gps_traces'

with (export_fp/'raw_coords.pkl').open('rb') as fh:
    coords_dict, trips_df = pickle.load(fh)

#set the minimum number of points required for a trip to be considered
#300 is based on the number of points needed for a second-by-second trace of five minutes
point_threshold = 5 * 60

#%% speed deviation, use speed between points to filter out bad values

remove_list = []
itp = gpd.read_file(Path.home()/'Documents/BikewaySimData/Data/Study Areas/itp.gpkg')

speed_deviation_max_mph = 47
pause_threshold_min = 10

for tripid, coords in tqdm(coords_dict.items()):
    #tripid = 304
    #coords = coords_dict[tripid]

    '''
    This loop searches for unrealistic jumps between gps points
    and removes these points until there are no more jumps.

    It first looks at the first point in the data to see if there
    is a jump after it. This helps remove points that were recorded before
    the GPS had been triangulated.
    
    Then it looks at subsequent points and removes them if the threshold is
    exceeded.
    
    All the while, trips are removed if this process eliminates to many points
    '''

    #calculate metrics for the coordinates dataframe
    coords = calculate_coordinate_metrics(coords)
    # Calculate summary statistics for trips_df #
    trips_df = calculate_trip_metrics(tripid,coords,trips_df)

    #if the trip drops below the point_threshold, add it to the remove list (repeats throughout)
    if coords.shape[0] < point_threshold:
        remove_list.append(tripid)
        continue
    
    # if second value has speed above speed_deviation_max_mph remove the first point until no jump
    while coords.iloc[1]['speed_mph'] >= speed_deviation_max_mph:
        coords = coords.iloc[1:,:]
        if coords.shape[0] < 3:
            remove_list.append(tripid)
            break
        #recalculate metrics
        coords = calculate_coordinate_metrics(coords)
    
    if coords.shape[0] < point_threshold:
        remove_list.append(tripid)
        continue

    # remove point if speed_deviation_max_mph between it and previous point
    while (coords['speed_mph']>=speed_deviation_max_mph).sum() > 0:
        # only keep if below speed_deviation_max_mph mph or it's the first row and has na as the value
        coords = coords[(coords['delta_distance_ft']<speed_deviation_max_mph) | (coords['delta_distance_ft'].isna())]
        # make sure there are enough points
        if coords.shape[0] < point_threshold:
            remove_list.append(tripid)
            break
        #recalculate metrics
        coords = calculate_coordinate_metrics(coords)

    if coords.shape[0] < point_threshold:
        remove_list.append(tripid)
        continue
    
    '''
    Only keep trips that are entirely inside the perimeter (within Interstate 285) as
    some trips were recorded out of country or in different states.
    
    We do this now incase there were bad points that were outside study area that got removed in this step
    '''
    within = coords.clip(itp)
    if within.shape[0] < coords.shape[0]:
        remove_list.append(tripid)
        continue

    '''
    Identifies GPS trip chains and trim trips

    Some trips will have gaps between point recordings. When the gap is large, it
    indicates that the person may have forgotten to stop their recording,
    stopped at a destination along the way (trip chaining), or that there was some error.

    This algorithm checks for cases in which the person probably forgot to stop
    the recording and trims the excess points.

    If a chain was detected, then only the first leg of the trip is retained.
    There are only a small number of these.

    Step 0: Set the pause threshold
    Step 1: For each trip, count the number of points that exceed this threshold
    Step 2: For the first pause detected, trim trip if all subsequent points are
    within pause threshold biking distance (using avg speed of 8 mph)
    Step 3: If not all points are contained, then consider these points to be part of a new trip
    chain. Repeat step 1 until all legs of trip chains are identfied.
    Step 4: Remove trip chains from databases and export them for later examination.

    '''

    # distance travelled assuming 8 mph
    # mins * hr/mins * mile/hr * ft/mile = ft
    # divided by two to be conservative with street network
    travel_dist =  pause_threshold_min * (1/60) * 8 * 5280 / 2
    
    # reset index to ensure subsequent numbers
    coords.reset_index(drop=True,inplace=True)    

    #while (coords['delta_time'] > datetime.timedelta(minutes=pause_threshold_min)).any():
    if (coords['delta_time'] > datetime.timedelta(minutes=pause_threshold_min)).any():
        
        #find first pause
        first_pause = (coords['delta_time'] > datetime.timedelta(minutes=pause_threshold_min)).idxmax()
        
        #trim trip
        coords = coords.loc[0:first_pause]        
        
    coords = calculate_coordinate_metrics(coords)
        
    # '''
    # Drop points that have a speed less than 2 mph to clean up the traces (except for first)
    # Then reduce number of points 
    # '''
    
    coords = coords[coords['speed_mph'].abs()>2]
    
    if coords.shape[0] < point_threshold:
        remove_list.append(tripid)
        continue
    
    coords = calculate_coordinate_metrics(coords)
    
    spacing_ft = 50
    current_spacing = spacing_ft
    
    #start with first point
    keep = [coords.index.tolist()[0]]
    
    for index, value in coords['traversed_distance_ft'].items():
        if value > current_spacing:
            keep.append(index)
            current_spacing = value + spacing_ft
    #remove points
    coords = coords.loc[keep]
    
    # #update dict and metrics
    # coords = calculate_coordinate_metrics(coords)
    trips_df = calculate_trip_metrics(tripid,coords,trips_df)
    coords_dict[tripid] = coords

#remove these trips
for key in remove_list:
    try:
        coords_dict.pop(key)
    except:
        continue
trips_df = trips_df[~trips_df['tripid'].isin(remove_list)]
print(len(remove_list),'trips removed')

#export
export_files = (coords_dict,trips_df)

with (export_fp/'cleaned_traces.pkl').open('wb') as fh:
    pickle.dump(export_files,fh)

#%%

    # #use for finding how many trips have long pauses
    # for y in range(1,21):
    #     print((coords[coords.groupby('tripid')['datetime'].diff() > datetime.timedelta(minutes=y)]['tripid'].nunique()))
    
    # #must have at least 5 mins of travel
    # trip_chain_dict = {}
    # remove_trip_chains = []


        # #find distance to all subsequent points within allowed travel distance after pause (maybe do half to account for road network?)
        # noise = coords.loc[first_pause:,'geometry'].distance(coords.loc[first_pause,'geometry']) < travel_dist
        
        # if noise.all():
        #     #if all points are contained, trim the trip end break out of loop
        #     legs[leg_index] = coords[~coords.index.isin(noise[noise].index)]
        #     break
        # else:
        #     #add points up until pause as leg in legs
        #     legs[leg_index] = coords.loc[:first_pause-1]
        #     leg_index += 1
        #     #break loop if pause is the last point
        #     if coords.index.tolist()[-1] == first_pause:
        #         break
        #     #trim coords df to include all points after pause and loop unless end condition met
        #     coords = coords.loc[first_pause+1:]
        #     #end condition
        #     if ~(coords['delta_time'] > datetime.timedelta(minutes=pause_threshold_min)).any():
        #         legs[leg_index] = coords
    
    # if len(legs) > 1:
    #     remove_trip_chains.append(tripid)
    #     for leg, items in legs.items():
    #         trip_chain_dict[(tripid,leg)] = legs[leg]
    # else:
    #     coords_dict[tripid] = legs[0]
            
# #%% assemble trip dataframe for trip chains

# i = 0
# df = {}

# #assemble new dataframe
# for tripid, item in tqdm(trip_chain_dict.items()):
#     item = calculate_coordinate_metrics(item)
    
#     leg = tripid[1]
#     tripid = tripid[0]
    
#     # duration
#     duration = item['datetime'].max() - item['datetime'].min()
    
#     # time
#     min_delta_time = item['delta_time'].min()
#     max_delta_time = item['delta_time'].max()
#     mean_delta_time = item['delta_time'].mean()
    
#     # distance
#     min_distance_ft = item['delta_distance_ft'].min()
#     max_distance_ft = item['delta_distance_ft'].max()
#     avg_distance_ft = item['delta_distance_ft'].mean()
#     total_distance_ft = item['delta_distance_ft'].sum()
    
#     # speed
#     max_speed_mph = item['speed_mph'].max()  
#     min_speed_mph = item['speed_mph'].min()
#     avg_speed_mph = item['speed_mph'].mean()

#     tot_points = item.shape[0]
    
#     df[i] = [tripid,leg,duration,min_delta_time,max_delta_time,
#              mean_delta_time,min_distance_ft,max_distance_ft,
#              avg_distance_ft,total_distance_ft,max_speed_mph,min_speed_mph,
#              avg_speed_mph,tot_points]
#     i += 1
    
# trip_chains_df = pd.DataFrame.from_dict(df,
#                                       orient='index',
#                                       columns=['tripid','leg','duration','min_delta_time',
#                                                'max_delta_time','mean_delta_time',
#                                                'min_distance_ft','max_distance_ft','avg_distance_ft',
#                                                'total_distance_ft','max_speed_mph','min_speed_mph',
#                                                'avg_speed_mph','tot_points'])

# #export
# export_files = (trip_chain_dict,trip_chains_df)

# with (export_fp/'trip_chains.pkl').open('wb') as fh:
#     pickle.dump(export_files,fh)

# #remove from main database
# for key in remove_trip_chains:
#     try:
#         coords_dict.pop(key)
#     except:
#         continue
# trips_df = trips_df[~trips_df['tripid'].isin(remove_trip_chains)]
# print(len(remove_trip_chains),'trip chains removed')

# #%% new trip segements must be at least five minutes and have at least 3 points

# point_threshold = 300

# #remove with less than 300
# below_5 = trips_df['initial_total_points'] < point_threshold
# tripids_below_5 = trips_df.loc[below_5,'tripid'] 
# trips_df = trips_df[~below_5]

# for tripid in tripids_below_5:
#     coords_dict.pop(tripid)

#%% spacing

#take cumulative distance from original
#find when value exceeds 50ft and record it

# print('Reducing number of points')
# for key, item in tqdm(coords_dict.items()):
#     '''
#     Drop points that have a speed less than 2 mph to clean up the traces (except for first)
#     Then reduce number of points 
#     '''
    
#     item = calculate_coordinate_metrics(item)
    
#     item = item[item['speed_mph'].abs()>2]
    
#     item = calculate_coordinate_metrics(item)
    
#     spacing_ft = 50
#     current_spacing = spacing_ft
    
#     #start with first point
#     keep = [item.index.tolist()[0]]
    
#     for index, value in item['distance_ft_cumulative'].items():
#         if value > current_spacing:
#             keep.append(index)
#             current_spacing = value + spacing_ft
    
#     #count number of points dropped
#     #trips_df.at[trips_df['tripid']==tripid,'final_tot_points'] = item.shape[0]
#     #redo
#     trips_df = calculate_trip_metrics(tripid,item, trips_df)
    
#     #update dict
#     coords_dict[key] = item.loc[keep]

#%% export step



# #%%
# export = coords_dict[6031].copy()

# #for exporting to gpkg
# def convert_datetime_columns_to_str(dataframe):
#     for column in dataframe.select_dtypes(include=['datetime64','timedelta64']):
#         dataframe[column] = dataframe[column].astype(str)
#     return dataframe

# export = convert_datetime_columns_to_str(export)

# export.to_file(Path.home()/'Downloads/scratch.gpkg',layer='many_pauses')


# #%%
# export = new_coords_dict[(3712,25)].copy()

# #for exporting to gpkg
# def convert_datetime_columns_to_str(dataframe):
#     for column in dataframe.select_dtypes(include=['datetime64','timedelta64']):
#         dataframe[column] = dataframe[column].astype(str)
#     return dataframe

# export = convert_datetime_columns_to_str(export)

# export.to_file(Path.home()/'Downloads/scratch.gpkg',layer='one_pauses')


# #%% add trip and user data to trips_df

# #trip info
# export_fp = Path.home() / 'Downloads/cleaned_trips'
# trip = pd.read_csv(export_fp/"trip.csv", header = None)
# col_names = ['tripid','userid','trip_type','description','starttime','endtime','initial_num_points']
# trip.columns = col_names
# #trip.drop(columns=['num_points'],inplace=True)

# trips_df = pd.merge(trips_df,trip,on='tripid')

# # import user csv
# user = pd.read_csv(export_fp/"user.csv", header=None)
# user_col = ['userid','created_date','device','email','age','gender','income','ethnicity','homeZIP','schoolZip','workZip','cyclingfreq','rider_history','rider_type','app_version']
# user.columns = user_col
# user.drop(columns=['device','app_version','app_version','email'],inplace=True)

# # merge trip and users
# #join the user information with trip information
# trips_df = pd.merge(trips_df,user,on='userid')


# ### Mask start and end location by 500ft ###
# #calculate distance difference and time difference from previous point
# item['distance_from_prev'] = item.distance(item.shift(1))
# #make first a 0
# item.iat[0,-1] = 0
# #find cumulative distance
# item['cumulative_dist'] = item['distance_from_prev'].cumsum()
# #find first 500ft
# first_500 = item['cumulative_dist'] < 500
# #find last 500ft
# last_500 = (item.iat[-1,-1] - item['cumulative_dist']) < 500
# #remove first and last 500 ft
# item = item[(first_500 | last_500) == False]


# ### hAccuracy ###
# '''
# In this step we remove points if the haccracy value is more than 2.5 standard deviations above the mean value.
# '''
# hAccuracy_filt = coords.groupby('tripid')['hAccuracy'].transform(lambda x: (x - x.mean()) > (x.std() * 2.5))
# coords = coords[-hAccuracy_filt]




#%% run rdp algo (removes too many points)

# =============================================================================
# simp_dict = {}
# 
# for key, item in coords_dict.items():
# 
#     tolerance_ft = 5
#         
#     #turn all trips into lines
#     line = LineString(item.sort_values('datetime')['geometry'].tolist())
#     
#     #simplify using douglass-peucker algo 
#     line = line.simplify(tolerance_ft, preserve_topology = False)
#     
#     #create dataframe
#     df = pd.DataFrame({'x':line.coords.xy[0],'y':line.coords.xy[1]})
#     
#     #create tuple for merging on
#     df['geometry_tup'] = [(xy) for xy in zip(df.x,df.y)]
#     item['geometry_tup'] = [(xy) for xy in zip(item.geometry.x,item.geometry.y)]
#     
#     #merge
#     simplified_traces = pd.merge(item,df,on='geometry_tup')
#     
#     #drop the tuple
#     simplified_traces.drop(columns=['geometry_tup'],inplace=True)
#         
#     #export
#     simp_dict[key] = simplified_traces
# =============================================================================




#%%


# #%% deprecated below


# #not sure what this was for
# # for x in random_trips:
    
# #     #subset dataframes
# #     sub_coords_max = coords_max[coords_max['tripid']==x].geometry.item()
# #     sub_coords_min = coords_min[coords_min['tripid']==x].geometry.item()
   
# #     #include these coordinates
# #     include_these = sample_coords.apply(lambda row: True if (row['tripid'] == x) & ((row['geometry'].distance(sub_coords_min) <= 500) | (row['geometry'].distance(sub_coords_max) <= 500)) else False, axis = 1)
    
# #     #set values to true
# #     sample_coords.loc[include_these,'include'] = False

# #doesn't like work related for some reason?
# #user_info['trip_type'] = user_info['trip_type'].str.replace('Work-related','workrelated')
# #user_info['trip_type'] = user_info['trip_type'].str.replace('Work-Related','workrelated')
# #user_info['tripid'].isin(points['tripid'])
# #unique_users = points['tripid'].drop_duplicates()
# #user_info = user_info[-(user_info['trip_type'] == 'workrelated')]





# #%% turn gps traces into linestrings


# def turn_into_linestring(points,name):
#     #turn all trips into lines
#     lines = points.sort_values(by=['tripid','datetime']).groupby('tripid')['geometry'].apply(lambda x: LineString(x.tolist()))

#     #get start time
#     start_time = points.groupby('tripid')['datetime'].min()

#     #get end time
#     end_time = points.groupby('tripid')['datetime'].max()
    
#     #turn into gdf
#     linestrings = gpd.GeoDataFrame({'start_time':start_time,'end_time':end_time,'geometry':lines}, geometry='geometry',crs=project_crs)
    
#     #write to file
#     linestrings.to_file(rf'C:\Users\tpassmore6\Documents\GitHub\ridership_data_analysis\{name}.geojson', driver = 'GeoJSON')

#     return linestrings

# all_lines = turn_into_linestring(sample_coords, 'all')
# sample_lines = turn_into_linestring(sample_coords_cleaned, 'cleaned')


# #%% some cleaning

# #begining
# all_coords['datetime'].min()
# #end
# all_coords['datetime'].max()
# # Data goes from 01-01-2010 to 06-09-2016

# #put month in new column
# all_coords['monthyear'] = pd.DatetimeIndex(all_coords['datetime']).to_period('M')

# #%% sum stats

# #get number of trips
# all_coords['tripid'].nunique()
# #28,224

# #get number of users
# all_coords['userid'].nunique()
# #1,626

# #average number of points per trip
# points_per_trip = all_coords.groupby('tripid').size()
# points_per_trip.plot(kind="hist")
# points_per_trip.size()
# #about 2,076

# #distribution of dates
# all_coords['monthyear'].plot(kind="hist")

# #summary stats
# all_coords.describe()

# #%% read gdf
# all_trips_gdf = gpd.read_file(r'C:\Users\tpassmore6\Documents\GitHub\ridership_data_analysis\test.geojson')

# #project
# all_trips_gdf.to_crs(epsg=2240,inplace=True)

# #trip distance
# all_trips_gdf['length_mi'] = all_trips_gdf.length / 5280

# #describe trip distance
# all_trips_gdf['length_mi'].describe()

# #drop anything less than 500 ft
# all_trips_gdf = all_trips_gdf[all_trips_gdf['length_mi'] >= 0.25 ]
# all_trips_gdf = all_trips_gdf[all_trips_gdf['length_mi'] < 20 ]

# #describe trip distance
# all_trips_gdf['length_mi'].describe()

# #merge
# all_trips_gdf = pd.merge(all_trips_gdf,trip,on='tripid')

# #num users
# all_trips_gdf['userid'].nunique()
# #1398

# #num trips
# all_trips_gdf['tripid'].nunique()
# #27,620


# #%% import cycleatlanta data from aditi

# clean_notes = pd.read_csv(r'C:/Users/tpassmore6/Documents/ridership_data/CycleAtlanta/Aditi Cycle Atlanta data/CATL_data/catl-data-2014-06-08/clean_notes.csv')
# clean_trips = pd.read_csv(r'C:/Users/tpassmore6/Documents/ridership_data/CycleAtlanta/Aditi Cycle Atlanta data/CATL_data/catl-data-2014-06-08/clean_trips.csv')
# clean_user = pd.read_csv(r'C:/Users/tpassmore6/Documents/ridership_data/CycleAtlanta/Aditi Cycle Atlanta data/CATL_data/catl-data-2014-06-08/clean_user.csv')

# #%% raw data chris

# #from 10/10/2012-5/2/2013
# df = pd.read_csv('C:/Users/tpassmore6/Documents/ridership_data/fromchris/CycleAtlanta/CycleAtlanta/9-10-16 Trip Lines and Data/raw data/coord.csv', header=None)

# df.sort_values(df[1], inplace= True)
# df.head()
# df.tail()

# df[1].max()
# df[1].min()


# fiona.listlayers(r'C:/Users/tpassmore6/Downloads/OTHER.gpx')

# layer = fiona.open(r'C:/Users/tpassmore6/Downloads/OTHER.gpx', layer = 'tracks')

# geom = layer[0]

# coords = geom['geometry']['coordinates'][0]

# points = [] 
# for x in coords:
#     points.append(Point(x))
    
# #df = gpd.GeoDataFrame({'geometry':points})
# #gdf.to_file(r'C:/Users/tpassmore6/Downloads/OTHER.geojson', driver = 'GeoJSON')

# #%% trip lines chris

# trip_lines = gpd.read_file(r'C:/Users/tpassmore6/Documents/ridership_data/fromchris/CycleAtlanta/CycleAtlanta/9-10-16 Trip Lines and Data/trip lines/routes_sub_lines.shp')

# #%% route points chris

# route_points = gpd.read_file(r'C:/Users/tpassmore6/Documents/ridership_data/fromchris/CycleAtlanta/CycleAtlanta/9-10-16 Trip Lines and Data/route points/routes_sub_0.shp')

# #old kalman filter code
# import numpy as np
# from pykalman import KalmanFilter


# #get rid of speed because i don't think it matters here

# #convert to metric projection
# sample.to_crs('epsg:26967',inplace=True)

# sample['x'] = sample.geometry.x
# sample['y'] = sample.geometry.y

# #convert our measurements to numpy
# measurements = sample[['x','y','speed','hAccuracy','dt']].to_numpy()

# #we have 5 states that we're tracking (use the intial values)
# #we don't need to keep track of haccuracy or speed or time, we just want latlon
# #initial_state_mean = [measurements[0,0],measurements[0,1]]
# initial_state_mean = [measurements[0,0],measurements[0,1],0,sample['hAccuracy'].median(),0]

# #we don't have a kinematic equation for explaining the expected next state (because no vector/bearing)
# #so instead we just assume a linear transition based on change in the variable

# #x_new = x + x_v*dt #since we don't know angle/bearing we have to assume position is constant
# #y_new = y + y_v*dt
# #speed_new = speed #assuming constant speed
# #haccuracy = haccuracy #assuming constant haccuracy


# transition_matrix = [[1,0,1,1,0],
#                      []]

# #these are the values we're getting from the phone
# observation_matrix = np.eye(5)

# #
# kf1 = KalmanFilter(transition_matrices = transition_matrix,
#                   observation_matrices = observation_matrix,
#                   initial_state_mean = initial_state_mean,
#                   )

# #
# kf1 = kf1.em(measurements, n_iter=5)
# (smoothed_state_means, smoothed_state_covariances) = kf1.smooth(measurements)

# #second run adds observation covariance matrix that smooths out the data even more, increase the multiplication factor to increase smoothing
# kf2 = KalmanFilter(transition_matrices = transition_matrix,
#                   observation_matrices = observation_matrix,
#                   initial_state_mean = initial_state_mean,
#                   observation_covariance = 50*kf1.observation_covariance,
#                   em_vars=['transition_covariance', 'initial_state_covariance'])

# kf2 = kf2.em(measurements, n_iter=5)
# (smoothed_state_means, smoothed_state_covariances)  = kf2.smooth(measurements)