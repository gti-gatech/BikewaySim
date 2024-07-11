import json
from pathlib import Path

'''
Function used to add filepaths to config json
'''

def filepaths():

    config = json.load((Path.cwd().parent / 'config.json').open('rb'))
    
    for key, item in config.items():
        if "_fp" in key:
            config[key] = Path(item)

    bicycle_facilities_fp = Path(config['project_fp']) / 'Bicycle_Facilities'
    bikewaysim_fp = Path(config['project_fp']) / 'BikewaySim'
    calibration_fp = Path(config['project_fp']) / 'Calibration'
    cycleatl_fp = Path(config['project_fp']) / 'CycleAtlanta'
    matching_fp = Path(config['project_fp']) / 'Map_Matching'
    network_fp = Path(config['project_fp']) / 'Network'
    osmdwnld_fp = Path(config['project_fp']) / "OSM_Download"

    # create if it doesn't exist
    if bicycle_facilities_fp.exists() == False:
        bicycle_facilities_fp.mkdir()
    if bikewaysim_fp.exists() == False:
        bikewaysim_fp.mkdir()
    if calibration_fp.exists() == False:
        calibration_fp.mkdir()
    if cycleatl_fp.exists() == False:
        cycleatl_fp.mkdir()
    if matching_fp.exists() == False:
        matching_fp.mkdir()
    if network_fp.exists() == False:
        network_fp.mkdir()
    if osmdwnld_fp.exists() == False:
        osmdwnld_fp.mkdir()

    #add to config
    config["bicycle_facilities_fp"] = bicycle_facilities_fp
    config["bikewaysim_fp"] = bikewaysim_fp
    config["calibration_fp"] = calibration_fp
    config["cycleatl_fp"] = cycleatl_fp
    config["matching_fp"] = matching_fp
    config["network_fp"] = network_fp
    config["osmdwnld_fp"] = osmdwnld_fp

    return config