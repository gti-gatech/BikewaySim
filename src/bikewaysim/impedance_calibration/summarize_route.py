from shapely.ops import LineString
import pandas as pd
import geopandas as gpd

def route_attributes1(tripid,edge_list,links,turns_df):
    '''
    Aggregates link attributes to the route level. There are several types of aggregation
    available.
    
    Three different types of summarization:
        Instance based (turns, signals, bridges, etc)
        Length based on certain tag (bike facilities)
        Cumulative (length,elevation)

    '''
    
    #empty contructor dict
    summary_attributes = {}

    summary_attributes['tripid'] = tripid

    #get route and turns
    route = [tuple(x) for x in edge_list]
    turns = [(route[i][0],route[i][1],route[i+1][0],route[i+1][1]) for i in range(0,len(route)-1)]
    
    #remove any doubling back (might be some of this in the matched dataset)
    turns = [turns for turns in turns if turns[0] != turns[2]]

    #get route attributes for link and turn dataframes
    links_w_attr = links.loc[route]
    turns_w_attr = turns_df.loc[turns]

    # Start with the basics
    total_length = links_w_attr.length.sum()
    attrs = {
        'multi use path': links_w_attr.loc[links_w_attr['multi use path']==1].length.sum() / total_length,
        'bike lane': links_w_attr.loc[links_w_attr['bike lane']==1].length.sum() / total_length,
        'above_4': links_w_attr.loc[links_w_attr['above_4']==True].length.sum() / total_length,
        'lane_0': links_w_attr.loc[links_w_attr['lanes']==0].length.sum() / total_length,
        'lane_1': links_w_attr.loc[links_w_attr['lanes']==1].length.sum() / total_length,
        'lane_2': links_w_attr.loc[links_w_attr['lanes']==2].length.sum() / total_length,
        'lane_3': links_w_attr.loc[links_w_attr['lanes']==3].length.sum() / total_length,
        'unsig_major_road_crossing': (turns_w_attr['unsig_major_road_crossing']==True).sum() / total_length
    }
    return attrs

    # # average grade by category (cut offs from broach)
    # # should probably use pd.cut here instead
    # zero_to_four = (route_w_attr['ascent_grade_%'] >= 0) & (route_w_attr['ascent_grade_%'] < 4)
    # four_to_eight = (route_w_attr['ascent_grade_%'] >= 4) & (route_w_attr['ascent_grade_%'] < 8)
    # eight_and_beyond = (route_w_attr['ascent_grade_%'] >= 8)
    # summary_attributes["(0,4]_prop"] = (route_w_attr.loc[zero_to_four,'length_mi'].sum() / route_w_attr['length_mi'].sum()).round(2)
    # summary_attributes["(4,8]_prop"] = (route_w_attr.loc[four_to_eight,'length_mi'].sum() / route_w_attr['length_mi'].sum()).round(2)
    # summary_attributes["(8,inf]_prop"] = (route_w_attr.loc[eight_and_beyond,'length_mi'].sum() / route_w_attr['length_mi'].sum()).round(2)

    # for col, item in cols_to_summarize.items():
    #     if isinstance(item,tuple):
    #         if item[0] == "threshold":                
    #             for breakpoint in item[1]:
    #                 #null values are ignored
    #                 summary_attributes[col+'_'+str(breakpoint)+'_prop'] = (route_w_attr.loc[route_w_attr[col] > breakpoint,'length_mi'].sum() / route_w_attr['length_mi'].sum()).round(2)
        
    #     if item == "category":
    #         for unique_val in route_w_attr[col].unique():
    #             #skip null results
    #             if (unique_val == None) | (unique_val != unique_val):
    #                 continue
    #             #remove decimal values
    #             if isinstance(unique_val,float):
    #                 unique_val = str(int(unique_val))
    #             # if isinstance(unique_val,str) == False:
    #             #     unique_val = str(unique_val)
    #             summary_attributes[col+'_'+unique_val+'_prop'] = (route_w_attr.loc[route_w_attr[col]==unique_val,'length_mi'].sum() / route_w_attr['length_mi'].sum()).round(2)
        
    #     if item == "bool":
    #         for unique_val in route_w_attr[col].unique():
    #             #if bool just take the true value
    #             if unique_val == True:
    #                 summary_attributes[col+'_prop'] = (route_w_attr.loc[route_w_attr[col]==unique_val,'length_mi'].sum() / route_w_attr['length_mi'].sum()).round(2)

    # # turns
    # summary_attributes.update(
    #     (turns_w_attr['turn_type'].value_counts() / route_w_attr['length_mi'].sum()).round(1).to_dict()
    # )

    # #signals
    # summary_attributes['signalized'] = (turns_w_attr['signalized'].sum() / route_w_attr['length_mi'].sum()).round(1)

def route_attributes(tripid,match_dict_entry,link_cols,turn_cols,links,turns_df,restr_stats):
    '''
    Two different types of summarization:

    Instance based (turns, signals, bridges, etc)
    Length based on certain tag (bike facilities)

    Cumulative (length,elevation)

    '''

    summary_attributes = {}
    summary_attributes['tripid'] = tripid

    #get trip date for the bike facility check (later)
    #trip_date_year = match_dict_entry['trip_start_time']#trip_date_year = match_dict_entry['trace'].iloc[0,2].year

    #get route and turns
    route = [tuple(x) for x in match_dict_entry.values]
    turns = [(route[i][0],route[i][1],route[i+1][0],route[i+1][1]) for i in range(0,len(route)-1)]
    
    #remove any doubling back (might be some of this in the matched dataset)
    # turns = [turns for turns in turns if turns[0] != turns[2]]
    #add turns to dataset for parsing?

    linkids = match_dict_entry['linkid'].tolist()
    reverse_links = match_dict_entry['reverse_link'].tolist()
    linkids_and_reverse = list(zip(linkids,reverse_links))

    #TODO change this cuz it's what's taking forever
    #get attributes
    route_w_attr = links.loc[linkids_and_reverse]
    lines = [list(x.coords) for x in route_w_attr.geometry.tolist()]
    route_geo = lines[0]
    for line in lines[1:]:
        route_geo.extend(line[1:])
    summary_attributes['geometry'] = LineString(route_geo)

    # turns_w_attr = turns_df.loc[turns]
    
    for col, item in link_cols.items():
        if isinstance(item,tuple):
            for breakpoint in item[1]:
                #null values are ignored
                summary_attributes[col+'_'+str(breakpoint)+'_pct'] = (route_w_attr.loc[route_w_attr[col] > breakpoint,'length_mi'].sum() / route_w_attr['length_mi'].sum() * 100).round(2)
        
        if item == "sum":
            summary_attributes[col] = round(route_w_attr.loc[:,col].sum(),2)
        
        if item == "category":
            for unique_val in route_w_attr[col].unique():
                #skip null results
                if (unique_val == None) | (unique_val != unique_val):
                    continue
                #remove decimal values
                if isinstance(unique_val,float):
                    unique_val = str(int(unique_val))
                summary_attributes[col+'_'+unique_val+'_pct'] = (route_w_attr.loc[route_w_attr[col]==unique_val,'length_mi'].sum() / route_w_attr['length_mi'].sum() * 100).round(2)
        
        if item == "bool":
            if col == 'on_eastside_beltline':
                summary_attributes[col+'_pct'] = \
                        (route_w_attr.loc[(route_w_attr[col]==True) & (route_w_attr['multi use path']==True),'length_mi'].sum() / route_w_attr.loc[route_w_attr['multi use path']==True,'length_mi'].sum() * 100).round(2)
            
            if col in restr_stats:
                for unique_val in route_w_attr[col].unique():
                    #if bool just take the true value
                    #TODO replace this with just True/False
                    if unique_val == True:
                        summary_attributes[col+'_pct'] = \
                        (route_w_attr.loc[(route_w_attr[col]==unique_val) & (route_w_attr['link_type']=='road'),'length_mi'].sum() / route_w_attr.loc[route_w_attr['link_type']=='road','length_mi'].sum() * 100).round(2)
            else:
                for unique_val in route_w_attr[col].unique():
                    #if bool just take the true value
                    if unique_val == True:
                        summary_attributes[col+'_pct'] = \
                            (route_w_attr.loc[route_w_attr[col]==unique_val,'length_mi'].sum() / route_w_attr['length_mi'].sum() * 100).round(2)

    # TODO indexing error here
    # for col, item in turn_cols.items():
    #     if item == 'bool':
    #         for unique_val in turns_w_attr[col].unique():
    #             #if bool just take the true value
    #             if unique_val == True:
    #                 summary_attributes[col] = (turns_w_attr[col]==unique_val).sum()#,'length_mi'].sum() #/ route_w_attr['length_mi'].sum()).round(2)
        
    #     if item == "category":
    #         for unique_val in turns_w_attr[col].unique():
    #             #skip null results
    #             if (unique_val == None) | (unique_val != unique_val):
    #                 continue
    #             #remove decimal values
    #             if isinstance(unique_val,float):
    #                 unique_val = str(int(unique_val))
    #             summary_attributes[col+'_'+unique_val] = (turns_w_attr[col]==unique_val).sum()# .sum() / route_w_attr['length_mi'].sum()).round(2)
        

        # # turns
    # summary_attributes.update(
    #     (turns_w_attr['turn_type'].value_counts() / route_w_attr['length_mi'].sum()).round(1).to_dict()
    # )

    #signals
    # summary_attributes['signalized'] = (turns_w_attr['signalized'].sum() / route_w_attr['length_mi'].sum()).round(1)



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