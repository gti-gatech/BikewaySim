import pickle
import pandas as pd

from bikewaysim.paths import config

def privacy_distance(df,privacy_dist=500):
    first_point = df['geometry'].iloc[0].buffer(privacy_dist)
    last_point = df['geometry'].iloc[-1].buffer(privacy_dist)
    double_buffer = df['geometry'].iloc[0].buffer(privacy_dist*2)
    if df['geometry'].intersects(double_buffer).all():
        return
    else:
        first_cut = df['geometry'].intersects(first_point).idxmin() # find the first point where it's false
        last_cut = df['geometry'].intersects(last_point).idxmax() - 1
        if df.loc[first_cut:last_cut,:].shape[0] == 0:
            return
        else:
            return df.loc[first_cut:last_cut,:]
        
def update_matching_settings(matching_settings):
    """
    Manages the matching settings for map matching.

    Exports a current version of the matching settings to a pickle file
    and updates the matching settings DataFrame with the new settings.
    Useful for keeping track of different map matching settings used.
    """

    # The matching setting dictionary stores all of the settings used for map matching, so they can be retrieved later for study
    if (config['matching_fp'] / 'matching_settings_df.pkl').exists():
        with (config['matching_fp'] / 'matching_settings_df.pkl').open('rb') as fh:
            matching_settings_df = pickle.load(fh)
    else:
        matching_settings_df = pd.DataFrame()

    #add to matching_settings_tuple if contents are unique
    row = pd.DataFrame([matching_settings])
    matching_settings_df = pd.concat([matching_settings_df,row],ignore_index=True)
    if matching_settings_df.duplicated().any():
        print('Settings have been used before')
    matching_settings_df.drop_duplicates(inplace=True)
    matching_index = matching_settings_df[(matching_settings_df == tuple(row.loc[0,:])).all(axis=1)].index.item()

    # exports only the current matching settings
    with (config['matching_fp']/'match_settings.pkl').open('wb') as fh:
        pickle.dump((matching_index,matching_settings),fh)

    # export the matching settings tested
    with (config['matching_fp']/'matching_settings_df.pkl').open('wb') as fh:
        pickle.dump(matching_settings_df,fh)