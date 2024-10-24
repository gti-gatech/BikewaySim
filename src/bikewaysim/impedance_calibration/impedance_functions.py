import numpy as np
import pandas as pd

########################################################################################

# Impedance Functions

########################################################################################

'''
Currently works with binary and numeric variables. Categorical data will have to be
cast into a different format for now.

Link impedance is weighted by the length of the link, turns are just the impedance associated
'''


def lowry_impedance_function(betas:np.array,betas_tup:tuple,links:pd.DataFrame,base_impedance_col:str,
                             base_link_col:str=None,trip_specific=None):
    '''
    Inspired by Lowry et al. 2016

    NOTE: attribute values are hard-coded for this one

    Presence of a bike lane or cycletrack 

    '''

    # retrieve col names of the links and the positions in the betas array
    betas_links = [(idx,x['col']) for idx, x in enumerate(betas_tup) if x['type']=='link']

    # calculate the bike accomodation factor
    bike_infra_cols = [x['col'] for x in betas_tup if x in ['bike lane','buffered bike lane','cycletrack']]
    if len(bike_infra_cols) > 0:
        f_bikeaccom = np.sum([betas_links[x] * links[x] for x in bike_infra_cols])
    else:
        f_bikeaccom = 0
    
    # calculate the road stress factor
    # each category gets its own parameter
    roadway_cols = [
        '1lpd','2lpd','3+lpd',
        '[0,30] mph','(30,inf) mph',
        '[0k,4k) aadt','[4k,10k) aadt','[10k,inf) aadt'
        ]
    f_roadway = np.sum([betas_links[x] * links[x] for x in roadway_cols])
    
    # calcualte the grade factor
    slope_cols = [x['col'] for x in betas_tup if x in ['[4,6) grade','[6,inf) grade']]
    if len(slope_cols) > 0:
        f_slope = np.sum([betas_links[x] * links[x] for x in slope_cols])

    # calculate bicycle accomodation adjusted road stress factor
    f_stress = f_roadway * (1 - f_bikeaccom)

    # calculate the final edge weights
    links['link_cost'] = links[base_impedance_col] * (1 + f_slope + f_stress)

    # reset link impedance for links that represent the base case
    # ISSUE can't do this because it overrites the grade impedance which applies no matter what
    # (in this case it should everything that is not a road)
    # base_link_col = 'base_link_col'
    # links[base_link_col] = links['link_type'] != 'road'
    # if base_link_col is not None:
    #     links.loc[base_link_col,'link_cost'] = links.loc[base_link_col,base_impedance_col]
    
    # round link cost
    links['link_cost'] = links['link_cost'].round(8)

def link_impedance_function(betas:np.array,betas_tup:tuple,links:pd.DataFrame,base_impedance_col:str,
                            base_link_col:str=None,trip_specific=None):
    '''
    Default link impedance function. Assumes that link impedance factors are additive
    and increase/decrease link impedance proportional to the link's distance/travel time.

    Rounded to eight decimals because expected units are in travel time

    Modifies the links dataframe inplace, so this function doesn't return anything.

    #TODO change travel time to seconds instead?
    '''
    
    # set up a multiplier of zeros that gets addded/subtracted to for visualization purposes
    multiplier = np.zeros(links.shape[0])

    # retrieve col names of the links and the positions in the betas array
    betas_links = [(idx,x['col'],x.get('bypass_base',False)) for idx, x in enumerate(betas_tup) if x['type']=='link']

    # check to see if there are any link impedances at all
    if len(betas_links) > 0:
        #assumes that these effects are additive
        for idx, col, bypass_base in betas_links:
            # only apply certain impedances such as elevation to the base case
            if bypass_base:
                multiplier = multiplier + (betas[idx] * links[col].values * (links[base_link_col]==False).astype(int))
            else:
                multiplier = multiplier + (betas[idx] * links[col].values)

        #scale the multiplier by a trip specific attribute such as distance or confident/fearless
        if trip_specific is not None:
            multiplier = multiplier * trip_specific
            
        #stores the multiplier
        links['multiplier'] = multiplier
        links['link_cost'] = links[base_impedance_col] * (1 + multiplier)
    else:
        print('No link impedance factors assigned')
        links['link_cost'] = links[base_impedance_col]

    # # reset link impedance for links that represent the base case
    # if base_link_col is not None:
    #     links.loc[base_link_col,'link_cost'] = links.loc[base_link_col,base_impedance_col]

    # round link cost
    links['link_cost'] = links['link_cost'].round(8)

def turn_impedance_function(betas:np.array,betas_tup:tuple,turns:pd.DataFrame):
    '''
    Default turn impedance function. Assumes that turns have zero initial cost.
    Event impedance so it's in the same units as the impedance (time or distance)
    '''
    #initialize a zero turn cost column
    turns['turn_cost'] = 0

    # retrieve col names of the links and the positions in the betas array
    betas_turns = [(idx,x['col']) for idx, x in enumerate(betas_tup) if x['type']=='turn']

    if len(betas_turns) > 0:
        #instance impedance
        for idx, col in betas_turns:
            turns['turn_cost'] = turns['turn_cost'] + (betas[idx] * turns[col])
    # TODO also have the print thing appear here but use the print results setting

    # round turn cost
    turns['turn_cost'] = turns['turn_cost'].round(8)