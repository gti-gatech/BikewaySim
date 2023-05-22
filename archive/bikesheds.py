# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 17:37:50 2022

@author: tpassmore6
"""

#TODO cross check with transitsim repo to see if this code is still needed

import pickle

links = gpd.read_file(r'C:/Users/tpassmore6/Documents/BikewaySimData/to_conflate/marta.gpkg',layer='conflated_links')
nodes = gpd.read_file(r'C:/Users/tpassmore6/Documents/BikewaySimData/to_conflate/marta.gpkg',layer='conflated_nodes')

select_tazs = ['553','411','1071','288']
mode = 'bike'
impedance = 'dist'

if not os.path.exists(rf'Outputs\{mode}_{impedance}'):
    os.makedirs(rf'Outputs\{mode}_{impedance}') 

#import snapped tazs
with open(r'snapped_tazs.pkl','rb') as fh:
    snapped_tazs_dict = pickle.load(fh)

centroids = gpd.read_file('base_layers.gpkg',layer='centroids',driver='GPKG') 

centroids['tazN'] = centroids['FID_1'].map(snapped_tazs_dict)

for taz in select_tazs:
    n = centroids.loc[centroids['FID_1']==taz,'tazN'].item()
    bikeshed, bikeshed_node = make_bikeshed(links,nodes,n,5280*2.5,'dist')

    mode = 'bike'
    #export
    bikeshed.to_file(rf'Outputs\{mode}_{impedance}\{taz}.gpkg',layer=f'{mode}shed')
    
    #do convex hull
    bounds = bikeshed.copy()
    bounds = bounds.dissolve()
    bounds.set_geometry(bounds.convex_hull,inplace=True)
    bounds.to_file(rf'Outputs\{mode}_{impedance}\{taz}.gpkg',layer=f'{mode}shed bounds')
    
for taz in select_tazs:
    n = centroids.loc[centroids['FID_1']==taz,'tazN'].item()
    bikeshed, bikeshed_node = make_bikeshed(links,nodes,n,5280*0.5,'dist')

    mode = 'walk'
    #export
    bikeshed.to_file(rf'Outputs\{mode}_{impedance}\{taz}.gpkg',layer=f'{mode}shed')
    
    #do convex hull
    bounds = bikeshed.copy()
    bounds = bounds.dissolve()
    bounds.set_geometry(bounds.convex_hull,inplace=True)
    bounds.to_file(rf'Outputs\{mode}_{impedance}\{taz}.gpkg',layer=f'{mode}shed bounds')
