import pandas as pd
import pickle

from bikewaysim.paths import config

def combine_results():
    fps = config['matching_fp'].glob("match_dict_*.pkl")

    with (config['matching_fp'] / 'match_settings.pkl').open('rb') as fh:
        matching_index, _ = pickle.load(fh)
    print(matching_index)
    match_dict = {}
    i = 0
    for fp in fps:
        if fp.parts[-1] == 'match_dict_full.pkl':
            continue
        with fp.open('rb') as fh:
            small_match_dict = pickle.load(fh)
        match_dict.update(small_match_dict)
        i += len(small_match_dict)
        del small_match_dict

    with (config['matching_fp'] / f'match_dict_full_{matching_index}.pkl').open('wb') as fh:
        pickle.dump(match_dict,fh)

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

