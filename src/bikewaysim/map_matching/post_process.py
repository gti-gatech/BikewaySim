import pandas as pd

def mapmatch_results(match_dict,cutoff):
    '''
    Prints the map match results and returns lists of failed and incomplete matches
    '''

    #get a series with the match ratios
    match_ratios = {tripid:item['match_ratio'] for tripid, item in match_dict.items() if isinstance(item,str)==False}
    match_ratios = pd.Series(match_ratios)
    
    # get counts and print them
    above_threshold = match_ratios[match_ratios > cutoff].index.tolist()
    below_threshold = match_ratios[match_ratios <= cutoff].index.tolist()
    failed_matches = [tripid for tripid, item in match_dict.items() if isinstance(item,str)==True]

    print(len(above_threshold),'/',len(match_dict),f"({round(len(above_threshold)/len(match_dict)*100)}%) successful matches")
    print(len(below_threshold),'/',len(match_dict),f"({round(len(below_threshold)/len(match_dict)*100)}%) partial matches")
    print(len(failed_matches),'/',len(match_dict),f"({round(len(failed_matches)/len(match_dict)*100)}%) failed matches")

    return above_threshold, below_threshold, failed_matches, match_ratios

def get_ods_from_match_dict(match_dict,links):
    '''
    Get the origin and destination nodes from the matching dict
    '''
    start_edges = [tuple(match_dict[tripid]['edges'].iloc[0,:].values) for tripid in match_dict.keys()]
    end_edges = [tuple(match_dict[tripid]['edges'].iloc[-1,:].values) for tripid in match_dict.keys()]
    starts = links.loc[start_edges,'A'].tolist()
    ends = links.loc[end_edges,'B'].tolist()
    return starts, ends

