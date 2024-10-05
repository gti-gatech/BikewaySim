import pandas as pd


def add_osm_attr(links,attr_fp):
    '''
    
    '''
    #bring in attribute data
    attr = pd.read_pickle(attr_fp)

    #drop a/b
    attr.drop(columns=['osm_A','osm_B'],inplace=True)

    #attach attribute data to filtered links
    links = pd.merge(links,attr,on=['osm_linkid'],how='left')

    #turn oneway into boolean
    links['oneway'] = links['oneway'] == 'yes'

    # #new columns
    # columns_to_add = ['bl','pbl','mu','speed_mph','lanes']

    # for col in columns_to_add:
    #     links[col] = 0

    # # bike facils
    # # note that these mostly apply to atlanta and may need to modified for other areas    

    # #find bike lanes
    # #get all bike specific columns (find features with cycle, foot, or bike in tags but not motorcycle)
    # bike_columns = [x for x in links.columns.to_list() if (('cycle' in x) | ('bike' in x)) & ('motorcycle' not in x)]
    # foot_columns = [x for x in links.columns.to_list() if ('foot' in x)]
    # bike_columns = bike_columns + foot_columns
    # bl_cond = ((links[bike_columns] == 'lane').any(axis=1)) & (links['highway'] != 'cycleway')
    # #bl.to_file(Path.home() / 'Downloads/testing.gpkg', layer = 'bls')

    # #anything that contains lane is a bike lane (class ii)
    # links.loc[bl_cond,'bl'] = 1

    # #find protected bike lanes (class iv)
    # if (links.columns == 'highway_1').any():
    #     pbl_cond = (links['highway'] == 'cycleway') | links['highway_1']=='cycleway'#& (links['foot'] == 'no')
    # else:
    #     pbl_cond = (links['highway'] == 'cycleway')
    # links.loc[pbl_cond, 'pbl'] = 1

    # #find multi-use paths (class i)
    # mups = ['path','footway','pedestrian','steps']
    # mup_cond = (links['highway'].isin(mups)) | ((links['highway'] == 'cycleway') & (links['foot'] != 'no'))
    # links.loc[mup_cond, 'mu'] = 1

    # #conflict warning
    # if (links[['mu','pbl','bl']].sum(axis=1) > 1).any():
    #     print('more than one bike facility detected')

    #speed limit (come back to this)
    #change nones to NaN
    links['maxspeed'] = pd.to_numeric(links['maxspeed'],errors='coerce').fillna(links['maxspeed'])
    
    #covert strings to numbers (gets rid of units, but won't work if no numbers present)
    func = lambda x: float(str.split(x,' ')[0]) if isinstance(x,str) else x
    links['speed_mph'] = links['maxspeed'].apply(func)

    # links.loc[(links['maxspeed'] < 25), network+'_<25mph'] = 1
    # links.loc[(links['maxspeed'] >= 25) & (links['maxspeed'] <= 30), network+'_25-30mph'] = 1
    # links.loc[(links['maxspeed'] > 30), network+'_>30mph'] = 1

    #number of lanes
    #convert none to nan
    links['lanes'] = links['lanes'].apply(lambda x: np.nan if x == None else x)

    #make sure numeric
    links['lanes'] = pd.to_numeric(links['lanes'])

    # links.loc[links['lanes'] < 3, network+'_1lpd'] = 1
    # links.loc[(links['lanes'] >= 3) & (links['lanes'] < 6), network+'_2-3lpd'] = 1
    # links.loc[links['lanes'] >= 8, network+'_>4lpd'] = 1

    #OTHER ATTRIBUTES (NOT AVAILABLE FOR ATLANTA)
    # =============================================================================
    # 
    # # road directionality
    # links.loc[(links['oneway']=='-1') | (links['oneway'] is None),'oneway'] = 'NA'
    # 
    # # parking
    # parking_columns = [x for x in links.columns.to_list() if 'parking' in x]
    # parking_vals = ['parallel','marked','diagonal','on_street']
    # links.loc[(links[parking_columns].isin(parking_vals)).any(axis=1),'parking_pres'] = 'yes'
    # links.loc[(links[parking_columns]=='no').any(axis=1),'parking_pres'] = 'yes'
    # links.drop(columns=parking_columns,inplace=True)
    # 
    # # sidewalk presence
    # sidewalks = [x for x in links.sidewalk.unique().tolist() if x not in [None,'no','none']]
    # links['sidewalk_pres'] = 'NA'
    # links.loc[links['sidewalk'].isin(sidewalks),'sidewalk_pres'] = 'yes'
    # links.loc[links['sidewalk'].isin([None,'no','none']),'sidewalk_pres'] = 'no'
    # links.drop(columns=['sidewalk'],inplace=True)
    # =============================================================================

    #final_cols = [] + columns_to_add
    #final_cols = [ network + '_' + x for x in final_cols]
    final_cols = ['osm_A','osm_B','osm_linkid','osmid','link_type','name','highway','oneway','bearing','bridge','tunnel'] + final_cols + ['geometry']

    links = links[final_cols]

    return links

def add_here_attr(links,attr_fp):

    network = 'here'

    #bring in attribute data
    attr = pd.read_pickle(attr_fp)

    #drop the A and B column
    attr.drop(columns=['here_A','here_B'],inplace=True)

    #attach attribute data to filtered links
    links = pd.merge(links,attr,on='here_linkid')
        
    # #init columns
    # columns_to_add = ['<25mph','25-30mph','>30mph','1lpd','2-3lpd','>4lpd']

    # for col in columns_to_add:
    #     links[network + '_' + col] = 0

    # take middle of speed categories
    here_speed_bins = {
        '1': '> 80 MPH',
        '2': '65-80 MPH',
        '3': '55-64 MPH',
        '4': '41-54 MPH',
        '5': '31-40 MPH',
        '6': '21-30 MPH',
        '7': '6-20 MPH',
        '8': '< 6 MPH'
        }
    links['speedlimit_range_mph'] = links['SPEED_CAT'].map(here_speed_bins)
    # links.loc[links['SPEED_CAT']=='< 25', network+'_<25mph'] = 1
    # links.loc[links['SPEED_CAT']=='25-30', network+'_25-30mph'] = 1
    # links.loc[links['SPEED_CAT']=='> 30', network+'_>30mph'] = 1

    # number of lanes per direction not including turn lanes
    here_lane_bins = {
        '1': '1', # 'one lane',
        '2': '2-3', # 'two or three lanes',
        '3': '> 4' # 'four or more'
        }
    links['lanes_per_direction'] = links['LANE_CAT'].map(here_lane_bins)
    # links.loc[links['LANE_CAT']=='1', network+'_1lpd'] = 1
    # links.loc[links['LANE_CAT']=='2-3', network+'_2-3lpd'] = 1
    # links.loc[links['LANE_CAT']=='> 4', network+'_>4lpd'] = 1

    #to/from lanes (when road is not symmetric)

    # road directionality
    here_oneway_bins = {
        'B':'both', # Both Directions
        'F':'oneway', # From Reference Node
        'T':'wrongway', # To Reference Node
        'N': 'NA' # Closed in both directions
        }
    links['DIR_TRAVEL'] = links['DIR_TRAVEL'].map(here_oneway_bins)

    # here functional class (does not correspond to FHWA or HFCS)
    func_class = {
        '1':'highways',
        '2':'major arterials',
        '3':'collectors/minor arterials',
        '4':'collectors/minor atrerials',
        '5':'local'
        }
    links['functional_class'] = links['FUNC_CLASS'].map(func_class)

    columns_to_keep = ['ST_NAME','functional_class','DIR_TRAVEL',
                      'speedlimit_range_mph','lanes_per_direction',
                      'FROM_LANES','TO_LANES']

    links = links[['here_A','here_B','here_linkid','link_type']+columns_to_keep+['geometry']]

    return links

def add_abm_attr(links,attr_fp):

    #bring in attribute data
    attr = pd.read_pickle(attr_fp)

    #attach attribute data to filtered links
    links = pd.merge(links,attr,on=['here_A','here_B','here_linkid'])
    
    #speed
    links['SPEEDLIMIT']

    conditions = [
        (links['SPEEDLIMIT'] > 80),
        (links['SPEEDLIMIT'] >= 65) & (links['SPEEDLIMIT'] < 80),
        (links['SPEEDLIMIT'] >= 55) & (links['SPEEDLIMIT'] < 64),
        (links['SPEEDLIMIT'] >= 41) & (links['SPEEDLIMIT'] < 54),
        (links['SPEEDLIMIT'] >= 31) & (links['SPEEDLIMIT'] < 40),
        (links['SPEEDLIMIT'] >= 21) & (links['SPEEDLIMIT'] < 30),
        (links['SPEEDLIMIT'] >= 6) & (links['SPEEDLIMIT'] < 20),
        (links['SPEEDLIMIT'] < 6)
    ]
    values = ['> 80 MPH', '65-80 MPH', '55-64 MPH', '41-54 MPH', '31-40 MPH', '21-30 MPH', '6-20 MPH', '< 6 MPH']

    links['SPEEDLIMIT'] = np.select(conditions, values)

    #oneway
    #links['oneway'] = links['two_way'] == False
    links = links[['NAME','SPEEDLIMIT']]

    return links

def add_arc_bike(links):

    network = 'arc'

    columns_to_add = ['bl','pbl','mu']

    for col in columns_to_add:
        links[network + '_' + col] = 0

    links.loc[links['facil']=='Protected Bike Lane', network+'_pbl'] = 1
    links.loc[links['facil']=='Bike Lane', network+'_bl'] = 1
    links.loc[(links['facil']=='Multi-Use Path'), network+'_mu'] = 1

    links = links[[network+'_bl',network+'_pbl',network+'_mu','geometry']]

    return links
