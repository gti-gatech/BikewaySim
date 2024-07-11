from shapely.ops import MultiLineString
import pandas as pd
import geopandas as gpd

def route_attributes(tripid,match_dict_entry,edge_col,cols_to_summarize,links,turns_df):
    '''
    Two different types of summarization:

    Instance based (turns, signals, bridges, etc)

    Length based on certain tag (bike facilities)

    Cumulative (length,elevation)

    '''

    summary_attributes = {}

    summary_attributes['tripid'] = tripid

    #get trip date for the bike facility check
    #trip_date_year = match_dict_entry['trip_start_time']#trip_date_year = match_dict_entry['trace'].iloc[0,2].year

    #get route and turns
    route = [tuple(x) for x in match_dict_entry[edge_col].values]
    turns = [(route[i][0],route[i][1],route[i+1][0],route[i+1][1]) for i in range(0,len(route)-1)]
    
    #remove any doubling back (might be some of this in the matched dataset)
    turns = [turns for turns in turns if turns[0] != turns[2]]
    #add turns to dataset for parsing?

    linkids = match_dict_entry[edge_col]['linkid'].tolist()
    reverse_links = match_dict_entry[edge_col]['reverse_link'].tolist()
    linkids_and_reverse = list(zip(linkids,reverse_links))

    #get attributes
    route_w_attr = links.loc[linkids_and_reverse]
    #route_w_attr = edges_w_attr.loc[linkids]
    turns_w_attr = turns_df.loc[turns]

    #add reverse direction
    #route_w_attr['reverse_link'] = reverse_links

    #turn to gdf
    #route_w_attr = gpd.GeoDataFrame(route_w_attr,geometry='geometry',crs=config['projected_crs_epsg'])
    summary_attributes["geometry"] = MultiLineString([list(line.coords) for line in route_w_attr['geometry'].values])

    #flip relevant attributes (no need with directed edges)
    # route_w_attr.loc[route_w_attr['reverse_link']==True,ascent_columns+descent_columns+bike_facils] = \
    #     route_w_attr.loc[route_w_attr['reverse_link']==True,descent_columns+ascent_columns+bike_facils[::-1]].values

    #set no facility_fwd to nan
    route_w_attr.loc[route_w_attr['facility_fwd']=='no facility'] = None
    
    #create copies of the bike facility column to use that doesn't consider date
    #route_w_attr['facility_fwd_nd'] = route_w_attr['facility_fwd']

    #set the bike facility to na if the trip date was before the bike facility  
    #route_w_attr.loc[route_w_attr['year'] > trip_date_year,bike_facils] = np.nan

    #summary columns
    summary_attributes["length_mi"] = route_w_attr['length_mi'].sum()
    summary_attributes["ascent_ft"] = route_w_attr['ascent_ft'].sum()
    summary_attributes["travel_time_min"] = route_w_attr['travel_time_min'].sum()
    #summary_attributes["descent_ft"] = route_w_attr['descent_ft'].sum()

    # # average grade by category (cut offs from broach)
    # # should probably use pd.cut here instead
    # zero_to_two = (route_w_attr['ascent_grade_%'] >= 0) & (route_w_attr['ascent_grade_%'] < 2)
    # two_to_four = (route_w_attr['ascent_grade_%'] >= 2) & (route_w_attr['ascent_grade_%'] < 4)
    # four_to_six = (route_w_attr['ascent_grade_%'] >= 4) & (route_w_attr['ascent_grade_%'] < 6)
    # six_and_beyond = (route_w_attr['ascent_grade_%'] >= 6)
    # summary_attributes["(0,2]_prop"] = (route_w_attr.loc[zero_to_two,'length_mi'].sum() / route_w_attr['length_mi'].sum()).round(2)
    # summary_attributes["(2,4]_prop"] = (route_w_attr.loc[two_to_four,'length_mi'].sum() / route_w_attr['length_mi'].sum()).round(2)
    # summary_attributes["(4,6]_prop"] = (route_w_attr.loc[four_to_six,'length_mi'].sum() / route_w_attr['length_mi'].sum()).round(2)
    # summary_attributes["(6,inf)_prop"] = (route_w_attr.loc[six_and_beyond,'length_mi'].sum() / route_w_attr['length_mi'].sum()).round(2)

    # average grade by category (cut offs from broach)
    # should probably use pd.cut here instead
    zero_to_four = (route_w_attr['ascent_grade_%'] >= 0) & (route_w_attr['ascent_grade_%'] < 4)
    four_to_eight = (route_w_attr['ascent_grade_%'] >= 4) & (route_w_attr['ascent_grade_%'] < 8)
    eight_and_beyond = (route_w_attr['ascent_grade_%'] >= 8)
    summary_attributes["(0,4]_prop"] = (route_w_attr.loc[zero_to_four,'length_mi'].sum() / route_w_attr['length_mi'].sum()).round(2)
    summary_attributes["(4,8]_prop"] = (route_w_attr.loc[four_to_eight,'length_mi'].sum() / route_w_attr['length_mi'].sum()).round(2)
    summary_attributes["(8,inf]_prop"] = (route_w_attr.loc[eight_and_beyond,'length_mi'].sum() / route_w_attr['length_mi'].sum()).round(2)

    for col, item in cols_to_summarize.items():
        if isinstance(item,tuple):
            for breakpoint in item[1]:
                #null values are ignored
                summary_attributes[col+'_'+str(breakpoint)+'_prop'] = (route_w_attr.loc[route_w_attr[col] > breakpoint,'length_mi'].sum() / route_w_attr['length_mi'].sum()).round(2)
        
        if item == "category":
            for unique_val in route_w_attr[col].unique():
                #skip null results
                if (unique_val == None) | (unique_val != unique_val):
                    continue
                #remove decimal values
                if isinstance(unique_val,float):
                    unique_val = str(int(unique_val))
                # if isinstance(unique_val,str) == False:
                #     unique_val = str(unique_val)
                summary_attributes[col+'_'+unique_val+'_prop'] = (route_w_attr.loc[route_w_attr[col]==unique_val,'length_mi'].sum() / route_w_attr['length_mi'].sum()).round(2)
        
        if item == "bool":
            for unique_val in route_w_attr[col].unique():
                #if bool just take the true value
                if unique_val == True:
                    summary_attributes[col+'_prop'] = (route_w_attr.loc[route_w_attr[col]==unique_val,'length_mi'].sum() / route_w_attr['length_mi'].sum()).round(2)

    # dillon version for visual clarity
    # greater_than_four = (route_w_attr['ascent_grade_%'] > 4)
    # summary_attributes["(4,inf)_prop"] = (route_w_attr.loc[greater_than_four,'length_mi'].sum() / route_w_attr['length_mi'].sum()).round(2)

    #TODO add this back in the elevation step and use the same limits?
    #add meters on grade segments (i.e. add all in length along x to x)
    #could possibly be a more accurate represntation of steep roads

    # #instance columns to summarize
    # count_cols = ['bridge','tunnel']
    # for count_col in count_cols:
    #     summary_attributes[count_col] = (route_w_attr[count_col]==True).sum().round(0)

    # length of route columns to summarize
    #cols = #['link_type_new','link_type','highway']#,'speedlimit_range_mph','lanes_per_direction']

    #TODO replace with a dictionary system where the keys are column names and the values descibe how to aggregate

    # for col in cols_to_summarize:
    #     #make a summary column for every unique value in that column
    #     for unique_val in route_w_attr[col].unique():
    #         if isinstance(unique_val,str):
    #             summary_attributes[col+'_'+unique_val+'_prop'] = (route_w_attr.loc[route_w_attr[col]==unique_val,'length_mi'].sum() / route_w_attr['length_mi'].sum()).round(2)
    #         elif isinstance(unique_val,bool):
    #             #if bool just take the true value
    #             if unique_val == True:
    #                 summary_attributes[col+'_prop'] = (route_w_attr.loc[route_w_attr[col]==unique_val,'length_mi'].sum() / route_w_attr['length_mi'].sum()).round(2)
    #         elif isinstance(unique_val,float) & (unique_val==unique_val):
    #             new_unique_val = str(int(unique_val))
    #             summary_attributes[col+'_'+new_unique_val+'_prop'] = (route_w_attr.loc[route_w_attr[col]==unique_val,'length_mi'].sum() / route_w_attr['length_mi'].sum()).round(2)

    # turns
    summary_attributes.update(
        (turns_w_attr['turn_type'].value_counts() / route_w_attr['length_mi'].sum()).round(1).to_dict()
    )

    #signals
    summary_attributes['signalized'] = (turns_w_attr['signalized'].sum() / route_w_attr['length_mi'].sum()).round(1)

    return summary_attributes

def procees_summary_results(matched_summary,crs):
    matched_summary = pd.DataFrame.from_records(matched_summary)
    matched_summary = gpd.GeoDataFrame(matched_summary,crs=crs)
    matched_summary.fillna(0,inplace=True)
    first_columns = ["tripid","length_mi","travel_time_min","ascent_ft"]
    last_columns = ["geometry"]
    remaining_columns = sorted([col for col in matched_summary.columns if col not in first_columns+last_columns])
    new_column_order = first_columns + remaining_columns + last_columns
    matched_summary = matched_summary[new_column_order]
    return matched_summary