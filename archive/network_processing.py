#!/usr/bin/env python
# coding: utf-8




# ### Links and Nodes to Conflate

# In[13]:
  
import os
from pathlib import Path
import time
import geopandas as gpd
import pickle

#make directory/pathing more intuitive later
file_dir = r"C:\Users\tpassmore6\Documents\BikewaySimData" #directory of bikewaysim network processing code

#change this to where you stored this folder
os.chdir(file_dir)


base_name = "abm"
join_name = "here"

#road layers
base_links = gpd.read_file(r"processed_shapefiles/abm/abm_bikewaysim_road_links.geojson")
base_nodes = gpd.read_file(r"processed_shapefiles/abm/abm_bikewaysim_road_nodes.geojson")
join_links = gpd.read_file(r"processed_shapefiles/here/here_bikewaysim_road_links.geojson")
join_nodes = gpd.read_file(r"processed_shapefiles/here/here_bikewaysim_road_nodes.geojson")

base_links, base_nodes = cleaning_process(base_links,base_nodes,base_name)
join_links, join_nodes = cleaning_process(join_links,join_nodes,join_name)

# In[15]:


#first match the nodes, can repeat this by adding in previously matched_nodes
tolerance_ft = 25
matched_nodes, unmatched_base_nodes, unmatched_join_nodes = match_nodes(base_nodes,
                                                                        base_name,
                                                                        join_nodes,
                                                                        join_name,
                                                                        tolerance_ft,
                                                                        prev_matched_nodes=None
                                                                       )



#%%

#join the matched nodes to the base nodes once done with matching
matched_nodes_final = pd.merge(base_nodes, matched_nodes, on = f'{base_name}_ID', how = "left")


# In[19]:


unmatched_join_nodes.head()


# In[20]:


#create new node and lines from the base links by splitting lines can repeat after the add_new_links_nodes function
tolerance_ft = 25
split_lines, split_nodes, unmatched_join_nodes = split_lines_create_points(unmatched_join_nodes,
                                                                           join_name,
                                                                           base_links,
                                                                           base_name,
                                                                           tolerance_ft,
                                                                           export = False)
split_lines.head()


# In[21]:


#add new links and nodes to the base links and nodes created from split_lines_create_points function
new_links, new_nodes = add_new_links_nodes(base_links, matched_nodes_final, split_lines, split_nodes, base_name)
new_links.head()


# ### Attribute Transfer
# In the previous steps, we found geometric commonalties between the networks. In this step, we want to transfer attribute information from the join network into the base network. Link attributes are based on a link's reference ids, but the current set of links may not have reference ids that correspond to a join network link.
# 
# To address this, we buffer the base links and intersect them with the join links. We then measure the length of the resulting linestrings. The attribute information from the join links that have the maximum length (i.e. the maximum amount of overlap with the base link) is tranferred. This ensures that each base link is associated with only one join link's attributes.
# 
# ### NOTE: The buffer here needs to be smaller
# If it's larger, then a longer join node could be selected as the join link with most overlap.
# 
# This process will likely change in the future. A different approach might be to look at all the base links with at least one join node in the reference id column, and then look up all the links in the join network associated with that node (there should only be a few). Using other reference node that doesn't have a join node id, the nearest node in that lookup table could be found.

# In[22]:


#match attribute information with greatest overlap from joining links
buffer_ft = 30
new_base_links_w_attr = add_attributes(new_links, base_name, join_links, join_name, buffer_ft)
new_base_links_w_attr.head()


# ### Add rest of features
# Now that we've settled the geometric and attribute commonalities between the base and join networks, we can add in the join network features that aren't represented in the base network. This is done using a buffer. If a join link is covered at least 95% by a base link, then it is left out.

# In[23]:


#add unrepresented features from joining by looking at the attributes added in previous step for links and the list of matched nodes
added_base_links, added_base_nodes = add_rest_of_features(new_base_links_w_attr,new_nodes,base_name,join_links,join_nodes,join_name)


# ### Save as pickle, this is more of a progress save

# In[24]:


pickle.dump(added_base_links, open("processed_shapefiles/conflation/inter/abm_here_road.p","wb"))
pickle.dump(added_base_nodes, open("processed_shapefiles/conflation/inter/abm_here_road.p","wb"))


# ### Repeat for OSM
# Now that we've resovled ABM and HERE, we can add the second join network.

# In[25]:

#create new column
base_links = added_base_links
base_links.rename(columns={'abm_line_geo':'abmhere_line_geo'},inplace=True)
base_links.set_geometry('abmhere_line_geo',inplace=True)
base_links['abmhere_A_B'] = np.nan
base_links['abmhere_A_B'] = base_links['abmhere_A_B'].fillna(base_links['abm_A_B'])
base_links['abmhere_A_B'] = base_links['abmhere_A_B'].fillna(base_links['here_A_B'])

base_nodes = added_base_nodes
base_nodes.rename(columns={'abm_point_geo':'abmhere_point_geo'},inplace=True)
base_nodes.set_geometry('abmhere_point_geo',inplace=True)
base_nodes['abmhere_ID'] = np.nan
base_nodes['abmhere_ID'] = base_nodes['abmhere_ID'].fillna(base_nodes['abm_ID'])
base_nodes['abmhere_ID'] = base_nodes['abmhere_ID'].fillna(base_nodes['here_ID'])


base_name = "abmhere"
base_links = base_links.reset_index(drop=True)
base_nodes = base_nodes.reset_index(drop=True)

join_name = "osm"
join_links = gpd.read_file(r"processed_shapefiles/osm/osm_bikewaysim_road_links.geojson")
join_nodes = gpd.read_file(r"processed_shapefiles/osm/osm_bikewaysim_road_nodes.geojson")


# In[26]:


#clean join links (no need to clean base links)
join_links, join_nodes = cleaning_process(join_links,join_nodes,join_name)


# In[27]:


#first match the nodes, can repeat this by adding in previously matched_nodes
tolerance_ft = 25
matched_nodes, unmatched_base_nodes, unmatched_join_nodes = match_nodes(base_nodes, base_name, join_nodes, join_name, tolerance_ft, prev_matched_nodes=None)

#join the matched nodes to the base nodes once done with matching
matched_nodes_final = pd.merge(base_nodes, matched_nodes, on = f'{base_name}_ID', how = "left")


# In[28]:


unmatched_join_nodes.head()


# In[29]:


#create new node and lines from the base links by splitting lines can repeat after the add_new_links_nodes function
tolerance_ft = 25
split_lines, split_nodes, unmatched_join_nodes = split_lines_create_points(unmatched_join_nodes,
                                                                           join_name,
                                                                           base_links,
                                                                           base_name,
                                                                           tolerance_ft,
                                                                           export = False)
split_lines.head()

#%%





# In[ ]:


#add new links and nodes to the base links and nodes created from split_lines_create_points function
new_links, new_nodes = add_new_links_nodes(base_links, matched_nodes_final, split_lines, split_nodes, base_name)
new_links.head()


# In[ ]:


#match attribute information with greatest overlap from joining links
buffer_ft = 30
new_base_links_w_attr = add_attributes(new_links, base_name, join_links, join_name, buffer_ft)
new_base_links_w_attr.head()


# In[ ]:


#add unrepresented features from joining by looking at the attributes added in previous step for links and the list of matched nodes
added_base_links, added_base_nodes = add_rest_of_features(new_base_links_w_attr,new_nodes,base_name,join_links,join_nodes,join_name)


# # Bike Subnetworks

# In[ ]:


#bike layers
bike_links = gpd.read_file(r'processed_shapefiles/here/here_bikewaysim_bike_links.geojson')
bike_nodes = gpd.read_file(r'processed_shapefiles/here/here_bikewaysim_bike_nodes.geojson')
bike_name = 'here'


# In[ ]:


#clean excess columns
bike_links, bike_nodes = cleaning_process(bike_links,bike_nodes,bike_name)


# ### Merge with other networks

# In[ ]:


tolerance_ft = 25
merged_links, merged_nodes = merge_diff_networks(added_base_links, added_base_nodes, 'road', bike_links, bike_nodes, 'bike', tolerance_ft)


# ### Add reference IDs

# In[ ]:


# match reference IDs based on all the id in the nodes
refid_base_links = add_reference_ids(merged_links, merged_nodes)


# In[ ]:


refid_base_links.head()


# In[ ]:





# ### Export

# In[ ]:


refid_base_links.to_file(r'processed_shapefiles\conflation\final_links.geojson', driver = 'GeoJSON')
merged_nodes.to_file(r'processed_shapefiles\conflation\final_nodes.geojson', driver = 'GeoJSON')


# ## Convert for use in BikewaySim
# 
# This last section focusses on making sure that the conflated network is readable by BikewaySim. After this is completed, you can run the Running BikwaySim notebook.

# In[ ]:


import os
from pathlib import Path
import time
import pandas as pd
import geopandas as gpd
import pickle

#make directory/pathing more intuitive later
file_dir = r"C:\Users\tpassmore6\Documents\BikewaySimData" #directory of bikewaysim network processing code

#change this to where you stored this folder
os.chdir(file_dir)


# ### Specify filepaths

# In[ ]:


#filepath for just OSM network
conflated_linksfp
conflated_nodesfp

#filepath for conflated network
#conflated_linksfp = r'processed_shapefiles\conflation\final_links.geojson'
#conflated_nodesfp = r'processed_shapefiles\conflation\final_nodes.geojson'

#filepaths for network attribute data (doesn't have to be a shapefile)
abm_linksfp = r'processed_shapefiles\abm\abm_bikewaysim_base_links.geojson'
here_linksfp = r'processed_shapefiles\here\here_bikewaysim_base_links.geojson'
osm_linksfp = r'base_shapefiles\osm\osm_links_attr.p'


# #### Node cleaning and export

# In[ ]:


#import conflated nodes
conflated_nodes = gpd.read_file(conflated_nodesfp)

#drop the num links columns
conflated_nodes = conflated_nodes.drop(columns=['abm_num_links','here_num_links'])

#create an N column that takes the abm_id if avaiable followed by the here_id
func = lambda row: row['here_ID'] if row['abm_ID'] == None else row['abm_ID']
conflated_nodes['N'] = conflated_nodes.apply(func,axis=1)

#create UTM coords columns
conflated_nodes['X'] = conflated_nodes.geometry.x
conflated_nodes['Y'] = conflated_nodes.geometry.y

#reproject and find latlon
conflated_nodes = conflated_nodes.to_crs(epsg=4326)
conflated_nodes['lon'] = conflated_nodes.geometry.x
conflated_nodes['lat'] = conflated_nodes.geometry.y

#filter
conflated_nodes = conflated_nodes[['N','X','Y','lon','lat','geometry']]

#export
conflated_nodes.to_file(r'processed_shapefiles\prepared_network\nodes\nodes.geojson',driver='GeoJSON')
conlfated_nodes = conflated_nodes.drop(columns=['geometry'])
conflated_nodes.to_csv(r'processed_shapefiles\prepared_network\nodes\nodes.csv')


# ### Link cleaning and export

# In[ ]:


#import conflated network
conflated_links = gpd.read_file(conflated_linksfp)


# #### Merging function

# In[ ]:


def merge_network_and_attributes(conflated_links,attr_network,cols_to_keep):
    #find the shared columns between conflated network and attribute network
    shared_cols = list(conflated_links.columns[conflated_links.columns.isin(attr_network.columns)])

    if len(shared_cols) > 2:
        #merge based on shared columns
        conflated_links = pd.merge(conflated_links,attr_network[cols_to_keep + shared_cols],on=shared_cols,how='left')
        print(conflated_links.head(20))
    else:
        print(f'Attr_network columns not in conflated network')
    return conflated_links


# In[ ]:


#import data with attributes, don't bring in geometry
abm_links = gpd.read_file(abm_linksfp,ignore_geometry=True)

#specify which columns you need
cols_to_keep = ['NAME','SPEEDLIMIT','two_way']

#perform the merge
conflated_links = merge_network_and_attributes(conflated_links,abm_links,cols_to_keep)

#delete data with attributes to free up memory
del(abm_links)


# In[ ]:


here_links = gpd.read_file(here_linksfp,ignore_geometry=True)

cols_to_keep = ['ST_NAME','DIR_TRAVEL']

conflated_links = merge_network_and_attributes(conflated_links,here_links,cols_to_keep)
del(here_links)


# In[ ]:


osm_links = pickle.load(open(osm_linksfp,"rb"))

cols_to_keep = ['name']

conflated_links = merge_network_and_attributes(conflated_links,osm_links,cols_to_keep)
del(osm_links)


# In[ ]:


conflated_links.head()


# ### Data Merging
# In this case we're just using street names and speed limit, but this section is dedicated for dealing with duplicate and/or missing data.

# In[ ]:


#streetnames
#if abm name is present use that, else use the HERE name
conflated_links['name'] = conflated_links.apply(lambda row: row['ST_NAME'] if row['NAME'] == None else row['NAME'], axis=1)
#if no streetname exists then put in "placeholder" as the streetname
conflated_links['name'] = conflated_links.apply(lambda row: 'placeholder' if pd.isna(row['name']) else row['name'], axis=1)

#speed limits
#use the ABM speed limit, if none present assume 30mph
conflated_links['speedlimit'] = conflated_links['SPEEDLIMIT'].apply(lambda row: row if row == row else 30)

#drop old columns
conflated_links = conflated_links.drop(columns=['NAME','SPEEDLIMIT','ST_NAME'])


# ### Create A and B column
# If ABM ID in A column then go with that, else go with HERE ID.

# In[ ]:


conflated_links['A'] = conflated_links.apply(lambda row: row['here_A'] if row['abm_A'] == None else row['abm_A'], axis=1)
conflated_links['B'] = conflated_links.apply(lambda row: row['here_B'] if row['abm_B'] == None else row['abm_B'], axis=1)
conflated_links.head()


# ### Create reverse links for two way streets and calculate distance

# In[ ]:


conflated_links_rev = conflated_links.copy().rename(columns={'A':'B','B':'A'})

#filter to those that are two way
conflated_links_rev = conflated_links_rev[(conflated_links_rev['two_way'] != False) &
                                            (conflated_links_rev['DIR_TRAVEL'] != 'F') &
                                            (conflated_links_rev['DIR_TRAVEL'] != 'T')                            
                                            ]

conflated_links = conflated_links.append(conflated_links_rev).reset_index()

#create A_B column
conflated_links['A_B'] = conflated_links['A'] + '_' + conflated_links['B']

#drop uneeded cols
conflated_links = conflated_links.drop(columns=['two_way','DIR_TRAVEL'])

#calculate distance
conflated_links['distance'] = conflated_links.length

conflated_links.head()


# ### Export

# In[ ]:


conflated_links.to_file(r'processed_shapefiles\prepared_network\links\links.geojson',driver='GeoJSON')


# In[ ]:





# In[ ]:




