import numpy as np

'''
This is a Python translation of Rosa's R code:
 https://github.com/U-Shift/Declives-RedeViaria/blob/main/SpeedSlopeFactor/speedfactor.R
'''

def g_factor(slope,length):
    '''
    Assigns a speed factor based on the average slope % and length of
    a street segment. Length units in meters.
    '''

    if (slope > 3) & (slope <= 5) & (length > 120):
        return 6
    elif (slope > 5) & (slope <= 8) & (length > 60):
        return 5
    elif (slope > 8) & (slope <= 10) & (length > 30):
        return 4.5
    elif (slope > 10) & (slope <= 13) & (length > 15):
        return 4
    else:
        return 7

def speedfactor(slope, length):#, convert_ft_to_meters=True):
    '''
    
    '''

    # #print(slope, length)

    # if convert_ft_to_meters:
    #     length = length * 3.28

    g = g_factor(slope,length)

    if slope < -30:
        result_1 = 1.5
    elif slope < 0:
        result_1 = 1 + (0.7 / 13) * 2 * slope + 0.7 / (13**2) * slope**2
    elif slope > 20:
        result_1 = 10
    elif (slope >= 0) & (slope <= 20):
        result_1 = 1 + (slope / g)**2
    elif (slope > 13) & (length > 15):
        result_1 = 10
    else:
        return np.nan

    if result_1 > 10:
        return 10
    elif slope < -30:
        return 1.5
    elif slope < 0:
        return 1 + (0.7 / 13) * 2 * slope + 0.7 / (13**2) * slope**2
    elif slope > 20:
        return 10
    elif (slope >= 0) & (slope <= 20):
        return 1 + (slope / g)**2
    elif (slope > 13) & (length > 15):
        return 10
    else:
        return np.nan

def calculate_adjusted_speed(links,flatspeed_mph):
    #convert mph to kph
    flatspeed_kph = flatspeed_mph * 1.609344
    #convert ft to m
    links['length_m'] = links['length_ft'] / 3.28
    #get the slope factor
    links['slope_factor'] = links.apply(lambda row: 
        speedfactor(row['ascent_grade_%'],row['length_m']),axis=1)

    links['adjusted_speed_kph'] = flatspeed_kph / links['slope_factor']

    links['travel_time_min'] = (links['length_m'] / 1000) / flatspeed_kph * 60
    links['adj_travel_time_min'] = links['travel_time_min']
    links.loc[links['adjusted_speed_kph'].notna(),'adjusted_speed_kph'] = (links['length_m'] / 1000) / links['adjusted_speed_kph'] * 60

    #return links