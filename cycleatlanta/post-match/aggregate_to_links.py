# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 18:39:32 2023

@author: tpassmore6
"""

import pandas as pd
import geopandas as gpd
import pickle
import datetime
from pathlib import Path
from tqdm import tqdm
import itertools

#export filepath
export_fp = Path.home() / 'Downloads/cleaned_trips'

#links
network_fp = r"C:\Users\tpassmore6\Documents\TransitSimData\networks\final_network.gpkg"
edges = gpd.read_file(network_fp,layer="links")
edges['linkid'] = list(zip(edges['A'],edges['B']))


#%%import matched traces
with (export_fp/'matched_traces.pkl').open('rb') as fh:
     matched_traces = pickle.load(fh)

#export into same spot?
trips_df = pd.read_csv(export_fp/'trips.csv')

#only keep the fully matched ones
trips_df = trips_df[trips_df['match_ratio']>0.95]

#subset the dict
complete_traces = {key:item for key,item in matched_traces.items() if key in set(trips_df['tripid'].tolist())}

#%% import trip info
trip = pd.read_csv(export_fp/"trip.csv", header = None)
col_names = ['tripid','userid','trip_type','description','starttime','endtime','notsure']
trip.columns = col_names

#not sure what to do with the notes
#note = pd.read_csv(export_fp/'note.csv',header=None)

# import user info and filter columns
user = pd.read_csv(export_fp/"user.csv", header=None)
user_col = ['userid','created_date','device','email','age','gender','income','ethnicity','homeZIP','schoolZip','workZip','cyclingfreq','rider_history','rider_type','app_version']
user.columns = user_col
user.drop(columns=['device','app_version','app_version','email'], inplace=True)

# merge trip and users
#join the user information with trip information
trip_and_user = pd.merge(trip,user,on='userid')

#%%

age = {
0: "no data",
1: "Less than 18" ,
2: "18-24" ,
3: "25-34" ,
4: "35-44" ,
5: "45-54" ,
6: "55-64" ,
7: "65+" 
}

gender = {
0: "no data" ,
1: "Female" ,
2: "Male"
}

ethnicity = {
0: "no data" ,
1: "White" ,
2: "African American" ,
3: "Asian" ,
4: "Native American" ,
5: "Pacific Islander" ,
6: "Multi-racial" ,
7: "Hispanic / Mexican / Latino" ,
8: "Other" 
}

income = {
0: "no data" ,
1: "Less than $20:000" ,
2: "$20:000 to $39:999" ,
3: "$40:000 to $59:999" ,
4: "$60:000 to $74:999" ,
5: "$75:000 to $99:999" ,
6: "$100:000 or greater" 
}

cycling_freq = {
0: "no data" ,
1: "Less than once a month" ,
2: "Several times per month", 
3: "Several times per week" ,
4: "Daily" 
}

rider_type = {
0: "no data" ,
1: "Strong & fearless" ,
2: "Enthused & confident" ,
3: "Comfortable: but cautious" ,
4: "Interested: but concerned" 
}

rider_history = {
0: "no data" ,
1: "Since childhood" ,
2: "Several years" ,
3: "One year or less" ,
4: "Just trying it out / just started" 
}

note_type = {
0: 'Pavement issue',
1: 'Traffic signal',
2: 'Enforcement',
3: 'Bike parking',
4: 'Bike lane issue',
5: 'Note this issue',
6: 'Bike parking',
7: 'Bike shops',
8: 'Public restrooms',
9: 'Secret passage',
10: 'Water fountain',
11: 'Note this asset'
}

#%%

#aggregate by female
only_female = trip_and_user[trip_and_user['gender']==1]['tripid'].to_list()

'''
Rather than just counting the number of times an occurance happens, need
to figure out how to give it meaning.

I want to know if female rides on different streets than males so look at % female on that link compared to all observed counts
'''

#aggregate by user type
strong_and_fearless = trip_and_user[trip_and_user['rider_type']==1]['tripid'].to_list()
ethused_and_confident = trip_and_user[trip_and_user['rider_type']==2]['tripid'].to_list()
comfortable_but_cautious = trip_and_user[trip_and_user['rider_type']==3]['tripid'].to_list()
interested_but_concerned = trip_and_user[trip_and_user['rider_type']==4]['tripid'].to_list()


def aggregate_trips_to_links(links,matched_traces,tripids,name):
    if tripids is not None:
        #use list of tripids to subset dict of matched traces
        filtered_dict = {key:item for key,item in matched_traces.items() if key in set(tripids)}
    else:
        filtered_dict = matched_traces
    #make one large series
    list_of_links = [item['edges'] for key, item in filtered_dict.items()]
    list_of_links = list(itertools.chain(*list_of_links))
    series_of_links = pd.Series(list_of_links)
    links[name] = links['linkid'].map(series_of_links.value_counts())
    return links
    
edges = aggregate_trips_to_links(edges,complete_traces,None,'total')
edges = aggregate_trips_to_links(edges,complete_traces,only_female,'female')
edges['female_pct'] = edges['female'] / edges['total'] * 100 

edges = aggregate_trips_to_links(edges,complete_traces,strong_and_fearless,'strong_and_fearless')
edges = aggregate_trips_to_links(edges,complete_traces,ethused_and_confident,'ethused_and_confident')
edges = aggregate_trips_to_links(edges,complete_traces,comfortable_but_cautious,'comfortable_but_cautious')
edges = aggregate_trips_to_links(edges,complete_traces,interested_but_concerned,'interested_but_concerned')

edges.drop(columns=['linkid']).to_file(export_fp/'aggregated_data.gpkg',layer='links')

'''
Need to remove reverse links from network either at this stage or before data will be masked otherwise
'''

#%%

#find wrongway