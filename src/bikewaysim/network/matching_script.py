import string
import geopandas as gpd
import pandas as pd
from pathlib import Path
import numpy as np

def overwrite_check(overwrite_edit_version,confirm):
    if (overwrite_edit_version == False) | (overwrite_edit_version == False):
        print('Overwrite or confirm set to false')
        return False
    elif (overwrite_edit_version) & (confirm == True):
        print('WARNING: OVERWRITE ENABLED')
        return True
    else:
        return False
    
# List of common suffixes and their abbreviated forms
suffixes = {
    'street': ['st'],
    'avenue': ['ave'],
    'boulevard': ['blvd'],
    'drive': ['dr'],
    'lane': ['ln'],
    'road': ['rd'],
    'court': ['ct'],
    'circle': ['cir'],
    'way': ['wy'],
    'place': ['pl'],
    # Add more suffixes and their abbreviations as needed
}

# Function to remove suffixes from street names
def remove_suffix(street_name):
    
    if street_name is None:
        return None
    if isinstance(street_name,float) | isinstance(street_name,int):
        if (np.isnan(street_name)):
            return None
    if street_name == '':
        return None

    # Lowercase evertyhing
    street_name = street_name.lower()

    # remove periods
    street_name = street_name.translate(str.maketrans('','',string.punctuation))

    # Split the street name into words
    words = street_name.split()

    # If mlk is one of those words, expand it back
    if 'mlk' in words:
        words.remove('mlk')
        words = words + ['martin','luther','king']

    # Remove directional indicators
    directions = ['north', 'south', 'east', 'west', 'northeast', 'southeast', 'northwest', 'southwest', 'n', 'e', 's', 'w', 'ne', 'se', 'nw', 'sw','wb']
    for direction in directions:
        words = [word for word in words if word.lower() != direction]

    # Remove suffixes
    for suffix, abbreviations in suffixes.items():
        # Remove full suffix
        if words[-1].lower().endswith(suffix):
            words[-1] = words[-1][:-(len(suffix))]
        # Remove abbreviated suffix
        for abbr in abbreviations:
            if words[-1].lower().endswith(abbr):
                words[-1] = words[-1][:-(len(abbr))]

    # Reconstruct the street name with spaces
    cleaned_street_name = ' '.join(words)
    return cleaned_street_name.strip()  # Remove any leading or trailing whitespace

# # Example list of street names
# street_names = ['Berne St (WB)','Bill Kennedy','Main St', 'Elm Avenue', 'Maple Blvd', 'Oak Dr', 'Pine Ln', 'Northwest 1st St', 'ne 2nd Ave', 'Eagle Row']

# # Remove suffixes from street names
# cleaned_street_names = [remove_suffix(name) for name in street_names]

# # Print cleaned street names
# for name in cleaned_street_names:
#     print(name)

def name_check(name1,name2):
    if (name1 is None) | (name2 is None):
        return False
    name1_words = name1.split()
    name2_words = name2.split()
    #check if any part of the name is right
    check1 = [True for name1_word in name1_words if name1_word in name2_words]
    #check2 = [True if name2_word in name1_words else False for name2_word in name2_words]
    if len(check1) > 0:
        return True
    else:
        return False
    
def suggested_matches(cycleways_osm,other_source,other_name,buffer_ft,max_hausdorff_dist,primary_key):
    #TODO there are some instances where having a name is causing a mis-attribution
    
    # copy to prevent modification of original dataframe
    cycleways_osm_buffered = cycleways_osm.copy()
    other_source = other_source.copy()

    # buffer the osm cycleways
    cycleways_osm_buffered.geometry = cycleways_osm_buffered.buffer(buffer_ft)
    
    # intersect with coa/arc (returns coa/arc linestrings)
    overlap = gpd.overlay(other_source,cycleways_osm_buffered)

    # create new fields
    overlap['auto_match'] = None # automatic matching suggestion
    overlap['manual_match'] = None # column for filling in QGIS
    overlap['notes'] = None # always good to have an extra field for notes
    
    #street name check if for bike lanes / sharrows / cycletracks
    #just needs one part of the name to match up
    overlap['name0'] = overlap['name'].apply(lambda row: remove_suffix(row))
    overlap[f"{other_name}_name0"] = overlap[f"{other_name}_name"].apply(lambda row: remove_suffix(row))
    overlap['name_check'] = overlap.apply(lambda row: name_check(row['name0'],row[f"{other_name}_name0"]),axis=1)

    ###### AUTO REJECTS ######
    # Use basic filters to eliminate matches that don't make sense

    ### DUPLICATE CHECK ###
    # NOTE: this interferes with the geometry aspect
    # reject duplicate matches to the same osm link
    # i.e., same coa/arc feature by attributes but a different coa/arc id
    # overlap.loc[overlap.drop(columns=[f"{other_name}_id",'geometry']).duplicated(),'auto_match'] = False

    ### STREET INFRA MATCHED TO A MULTI USE PATH CHECK ###
    street_infra = ['bike lane','buffered bike lane','sharrow','cycletrack'] # cycletracks are hard to indentify in osm
    off_street_infra = ['multi use path']

    # reject if osm is on street infra and suggested match is a multi use path
    osm_is_street = overlap[["facility_fwd",'facility_rev']].isin(street_infra).any(axis=1)
    other_is_mup = overlap[f"{other_name}_osm_type"].isin(off_street_infra)
    overlap.loc[osm_is_street & other_is_mup,'auto_match'] = False

    # reject if osm is a mup and suggested match is on street infra
    osm_is_mup = overlap[["facility_fwd",'facility_rev']].isin(off_street_infra).any(axis=1)
    other_is_street = overlap[f"{other_name}_osm_type"].isin(street_infra)
    overlap.loc[osm_is_mup & other_is_street,'auto_match'] = False

    # reject if osm is a bike lane of some type and the other is a cycletrack OR vice versa
    overlap.loc[(overlap[f"{other_name}_osm_type"]=='cycletrack') & (overlap['facility'].isin(['bike lane','buffered bike lane'])),'auto_match'] = False
    overlap.loc[(overlap["facility"]=='cycletrack') & (overlap[f"{other_name}_osm_type"].isin(['bike lane','buffered bike lane'])),'auto_match'] = False

    ### STREET NAME CHECK ###
    # reject match if street infrastructure and street names do not match
    # type 2 error if cycletrack does not have a street name so don't include these
    street_infra0 = ['bike lane','buffered bike lane','sharrow']
    osm_is_street = overlap[["facility_fwd",'facility_rev']].isin(street_infra0).any(axis=1)
    other_is_street = overlap[f"{other_name}_osm_type"].isin(street_infra0)
    overlap.loc[osm_is_street & other_is_street & (overlap['name_check']==False),'auto_match'] = False

    ### SHARROW CHECK ###
    # reject if osm says sharrow for both directions and arc/coa is not a sharrow
    overlap.loc[(overlap[['facility_fwd','facility_rev']] == 'sharrow').all(axis=1) & (overlap[f"{other_name}_osm_type"] != 'sharrow'),'auto_match'] = False
    # and reject if coa/arc says sharrow but osm does not
    overlap.loc[(overlap[['facility_fwd','facility_rev']] != 'sharrow').all(axis=1) & (overlap[f"{other_name}_osm_type"] == 'sharrow'),'auto_match'] = False

    ### MARK REMAINING ONE TO MANY MATCHES ###
    # to be a one to many must have multiple unassigned "accept matches"
    overlap['one_to_many'] = overlap.groupby(primary_key)['auto_match'].transform(lambda x: x.isna().sum() > 1)
    # set automatch == false onetomany to false
    overlap.loc[overlap['auto_match']==False,'one_to_many'] = False

    ###### AUTO ACCEPTS ######
    # Use basic filter to accepts existing one-to-one matches and hausdorff distance to get rid of one to many matches
    # After each accept, auto reject the possible matches

    ### NAME CHECK ONE TO ONE ###
    # accept if name check is correct and it's a one to one
    overlap.loc[
        overlap['name_check'] & (overlap['one_to_many']==False) & overlap['auto_match'].isna(),'auto_match'] = True
    one_match = overlap.groupby(primary_key)['auto_match'].transform(lambda x: (x == True).any())
    overlap.loc[one_match & overlap['auto_match'].isna(),'auto_match'] = False

    ### MULTI USE PATH ONE TO ONE ###
    # accept if both are multi-use paths and it's a one to one match
    overlap.loc[
        osm_is_mup & other_is_mup & (overlap['one_to_many']==False) & overlap['auto_match'].isna(),'auto_match'] = True
    one_match = overlap.groupby(primary_key)['auto_match'].transform(lambda x: (x == True).any())
    overlap.loc[one_match & overlap['auto_match'].isna(),'auto_match'] = False

    ### CYCLETRACK ONE TO ONE
    # accept if both are cycletracks and it's a one to one match
    osm_is_cycletrack = overlap[["facility_fwd",'facility_rev']].isin(['cycletrack']).any(axis=1)
    other_is_cycletrack = overlap[f"{other_name}_osm_type"] == 'cycletrack'
    overlap.loc[
        osm_is_cycletrack & other_is_cycletrack & (overlap['one_to_many']==False) & overlap['auto_match'].isna(),'auto_match'] = True
    one_match = overlap.groupby(primary_key)['auto_match'].transform(lambda x: (x == True).any())
    overlap.loc[one_match & overlap['auto_match'].isna(),'auto_match'] = False

    ### HAUSDORFF DISTANCE CHECK ###
    # add osm geometry to compare against arc/coa geometry
    overlap = pd.merge(overlap,cycleways_osm[[primary_key,'geometry']],on=primary_key)
    overlap['hausdorff_dist'] = overlap.apply(lambda row: row['geometry_x'].hausdorff_distance(row['geometry_y']),axis=1)
    overlap.drop(columns=['geometry_x'],inplace=True)
    overlap.rename(columns={'geometry_y':'geometry'},inplace=True)
    
    # replace intersected geometry with the original geometry
    overlap = gpd.GeoDataFrame(overlap,geometry='geometry')

    # TODO examine the current version in QGIS first
    # for auto_match == NULL set it equal to match with the minimum
    # hausdorff distance as long as it doesn't exceed the maximum amount    
    no_match = overlap.groupby(primary_key)['auto_match'].transform(lambda x: (x == True).any() == False)
    min_hausdorff = overlap[overlap['auto_match'].isna()].groupby(primary_key)['hausdorff_dist'].idxmin().tolist()    
    overlap.loc[overlap.index.isin(min_hausdorff) & (overlap['hausdorff_dist']<=max_hausdorff_dist) & no_match & overlap['auto_match'].isna(),'auto_match'] = True
    one_match = overlap.groupby(primary_key)['auto_match'].transform(lambda x: (x == True).any())
    overlap.loc[one_match & overlap['auto_match'].isna(),'auto_match'] = False

    # auto assign anything that wasn't accepted as false
    match = overlap.groupby(primary_key)['auto_match'].transform(lambda x: (x == 1).any())
    overlap.loc[overlap['auto_match'].isna(),'auto_match'] = False

    return overlap