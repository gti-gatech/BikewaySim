

from scipy.spatial import cKDTree
import numpy as np
import pandas as pd


#take in two geometry columns and find nearest gdB point from each
#point in gdA. Returns the matching distance too.
#MUST BE PROJECTED COORDINATE SYSTEM
def ckdnearest(gdA, gdB, return_dist=True):  
    
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
    
    if return_dist == False:
        gdf = gdf.drop(columns=['dist'])
    
    return gdf


def snap_to_network(to_snap,network_nodes_raw):
    #record the starting time
    time_start = time.time()
    
    #create copy of network nodes
    network_nodes = network_nodes_raw.copy()
    
    #rename geometry columns
    to_snap.rename(columns={'geometry':'original'},inplace=True)
    to_snap.set_geometry('original',inplace=True)
    network_nodes.rename(columns={'geometry':'snapped'},inplace=True)
    network_nodes.set_geometry('snapped',inplace=True)
    
    #find closest network node from each orig/dest
    snapped_nodes = ckdnearest(to_snap, network_nodes)

    #filter columns
    snapped_nodes = snapped_nodes[to_snap.columns.to_list()+['N','dist']]
        
    #drop geo column
    snapped_nodes.drop(columns=['original'],inplace=True)
    
    print(f'snapping took {round(((time.time() - time_start)/60), 2)} minutes')
    return snapped_nodes
