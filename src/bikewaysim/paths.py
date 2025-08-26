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
data_directories = {
    'bicycle_facilities_fp': config['project_fp'] / 'Bicycle_Facilities',
    'bikewaysim_fp': config['project_fp'] / 'BikewaySim',
    'calibration_fp': config['project_fp'] / 'Calibration',
    'cycleatl_fp': config['project_fp'] / 'CycleAtlanta',
    'matching_fp': config['project_fp'] / 'Map_Matching',
    'network_fp': config['project_fp'] / 'Network',
    'osmdwnld_fp': config['project_fp'] / "OSM_Download",
    'scratch_fp': config['project_fp'] / 'scratch',
    'figures_fp': config['project_fp'] / 'figures',
    'geofabrik_fp': config['raw_data_fp'] / 'geofabrik'
}
for key, data_dir in data_directories.items():
    if data_dir.exists() == False:
        data_dir.mkdir()

#add to config
config.update(data_directories)

#add study area to config
config['studyarea_fp'] = config['project_fp'] / 'studyarea.geojson'

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