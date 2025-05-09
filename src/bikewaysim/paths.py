import json
from pathlib import Path
import xyzservices.providers as xyz
from pyprojroot import here
root = here()

#import the config.json file
config = json.load((root / 'config.json').open('rb'))

# convert from text pathlib object
for key, item in config.items():
    if "_fp" in key:
        config[key] = Path(item)
        if (config[key].exists() == False) & (config[key].suffix == ''):
            config[key].mkdir(parents=True)

# bikewaysim data directories
bicycle_facilities_fp = config['project_fp'] / 'Bicycle_Facilities'
bikewaysim_fp = config['project_fp'] / 'BikewaySim'
calibration_fp = config['project_fp'] / 'Calibration'
cycleatl_fp = config['project_fp'] / 'CycleAtlanta'
matching_fp = config['project_fp'] / 'Map_Matching'
network_fp = config['project_fp'] / 'Network'
osmdwnld_fp = config['project_fp'] / "OSM_Download"

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

if config['scratch_fp'].exists() == False:
    config['scratch_fp'].mkdir()
if config['figures_fp'].exists() == False:
    config['figures_fp'].mkdir()

#add to config
config["bicycle_facilities_fp"] = bicycle_facilities_fp
config["bikewaysim_fp"] = bikewaysim_fp
config["calibration_fp"] = calibration_fp
config["cycleatl_fp"] = cycleatl_fp
config["matching_fp"] = matching_fp
config["network_fp"] = network_fp

# TODO figure out how to deal with base maps
# #tiles config for folium
# stadia_toner = {
#     "tiles": 'https://tiles.stadiamaps.com/tiles/stamen_toner_lite/{z}/{x}/{y}{r}.png' + f"?api_key={config['stadiaapikey']}",
#     "name": 'Stamen Toner',
#     "attr": xyz.Stadia.StamenToner.attribution
# }
# maptiler_streets = {
#     "tiles": xyz.MapTiler.Streets.build_url(key=config['maptilerapikey']),
#     "name": str.replace(xyz.MapTiler.Streets.name,'.',' '),
#     "attr": xyz.MapTiler.Streets.attribution
# }