# BikewaySim (Development Version)
BikewaySim is a data science project used for assessing the impedance reduction from new cycling infrastructure. It is part of Reid Passmore's Ph.D. dissertation. BikewaySim is currently intended to work in the continental United States.

This is the development version of BikewaySim, for the production version access: https://github.com/gti-gatech/BikewaySim


## Main functionalities:
1. Downloading and processing OpenStreetMap network data for bicycle routing using Geofabrik and OSMnx
1. Calibrating link and turn impedance functions for cycling using bicycling GPS traces or count data using stochopy
1. Finding the shortest path for selected or all-to-all O-D pairs given custom link and turn impedance functions
1. Processing shortest path results to generate several metrics for assessing the impacts of planned cycling infrastructure

## Install instructions:
- Clone the repository into your desired directory
```
git clone https://github.com/reidx19/BikewayDev
```
- Install conda to set up a Python virtual environment by following the [instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
- Using conda's CLI, install the invluded environment named `environment.yml` and name it `bikewaysim`
```
conda create env -n bikewaysim environment.yml
```
- Activate the 'bikewaysim' environment
```
conda activate bikewaysim
```
- Navigate to the `BikewaySim/src` folder on conda CLI and install the development package `bikewaysim`
```
pip install -e .
```
- Great! Everything should be installed!
- Most of the code is excuted through Jupyter Notebooks which can be opened/run/edited through VS Code or Jupyter Notebook, just make sure your conda environment is activated.
## Running Instructions

1. 

## Quick Start for Bike Routing
1. Open [Downloading_OSM.ipynb](./osm_download/Downloading_OSM.ipynb) and follow the instructions to download OpenStreetMap network data
1. Open [Step_1 Network_Filtering_and_Processing.ipynb](./network/Step_1_Network_Filtering.ipynb) to process OSM into network graph format and identify link types
1. Open [Step_2_Network_Reconciliation.ipynb](./network/Step_2_Network_Reconciliation.ipynb) to finalize the network for routing
1. Open [Run_BikewaySim.ipynb](./bikewaysim/Run_BikewaySim.ipynb) to perform aggregated shortest path routing and output Geopackage files to visualize in GIS software

## Quick Start Bike-Transit Routing
1. Open [Simulating%20Bike-Transit%20Trips.ipynb](./bike_transit/Simulating%20Bike-Transit%20Trips.ipynb) and walk through the steps until a study area is generated
1. Open [Downloading_OSM.ipynb](./osm_download/Downloading_OSM.ipynb) and use the generated study area to download OpenStreetMap network data
1. Continue with [Simulating%20Bike-Transit%20Trips.ipynb](./bike_transit/Simulating%20Bike-Transit%20Trips.ipynb)

### List of Modules

| Module                 | Description                | Status |
|------------------------|----------------------------|---|
| add_elevation_data     | Add USGS elevatoin data to a network | working  |
| bike-transit           | Run aggregated bike-transit shortest path calculations | working |                    
| run_bikewaysim         | Run aggregated cycling shortest path calculations | working |
| gps_processing         | Process and map-match GPS trace data | in development  |   
| impedance_calibration  | Use map-matched GPS traces to calirbate link impedance functions | in development |
| network                | Process network data for shortest path routing          | working
| osm_download           | Download OpenStreetMap network data             | working


This Jupyter Notebook uses code from "osm_download_functions.py" to retrieve and export to file OSM geometry and attribute data for a GeoDataFrame's boundingbox. It also contains templates for examining the attribute completion of the OSM data and for visualizing where attribute data is complete. Once run, the data can be used in the network module.

## gps_processing
This module contains code for 

## 

## Assessing Bike-Transit Accessibility




This version of the code was used for the NCST Report on "Simulating Bike-Transit Trips Using BikewaySim and TransitSim." This repository is composed of several scripts and jupyter notebooks necessary for running the analyses performed in that report. This readme will outline the steps needed to repeat the analyses done in that report and go over the contents of each file. For more detailed documentation, open up the file of interest to read through line comments.

__Note 1: The version of the code published here might not be the most current version. 
Please use the latest version of the code published in the GitHub repository:
https://github.com/gti-gatech/BikewaySim__

__Note 2: This repository does not contain any of the network data required to run BikewaySim. This data must be provided by the user. The data used in the NCST report came from OpenStreetMap, the Atlanta Regional Commission, and HERE. This repository does contain code for retrieving OpenStreetMap data. ARC data can be provided upon request from the ARC. HERE data must be licensed from HERE.__

# Repeating NCST Report Analyses

## Step 0 Downloading OSM.ipynb
This Jupyter Notebook uses code from "osm_dwnlod.py" to retrieve and export to file OSM geometry and attribute data for a GeoDataFrame's perimeter or boundingbox. It also contains templates for examining the attribute completion of the OSM data and for visualizing where attribute data is complete.

## Step 1 Network Filtering and processing.ipynb
This Jupyter notebook takes in spatial data on links and nodes from a provided street network (the code will create nodes if none are provided) in GeoJSON/Geopackage/Geodatabase/Shapefile format and the same study area GeoDataFrame used in Step 0. It filters the street network into five different layers (raw, general, road, bike, and service) and formats the street network into a routable network graph. The supporting script for this notebook is "prepare_network.py.". This script contains hard-coded filters for each network defined. If a network that is not HERE, ABM, or OSM is provided, new filters will need to be defined.

See the notebook for a detailed explanation on the five different layers but in general:

- The raw layer contains all the features that are within the study area but removes duplicate links intended to represent two way streets
- The general layer removes all interstates and interstate ramps and sidewalks where bicycles are not permitted
- The road layer contains all publicly owned roads with vehicular access
- The bike layer contains all multi-use paths, bike paths, and sidewalks where bikes have been explicitly permitted
- The service layer contains all private driveways and parking lot roads
- Attribute data for each network is stored in a ".pkl" file to reduce memory burden

## Step 2 Network Reconciliation.ipynb
This Jupyter notebook is a workspace for using the reconciliation functions in the "conflation_tools.py" and "network_reconcile.py" scripts. Unlike the previous notebooks, this notebook isn't meant to be run in series. 

## Step 3 Finalize Network.ipynb
The Jupyter notebook generates link impedances and prepares the reconciled network for routing for Step 4. This step is dependent on the available networks, network attribute data, and desired link impedances.

## Step 4 Run BikewaySim.ipynb
This final notebook runs the Dijkstra shortest path routing algorithm for the given OD pairs. Results are exported to file and processed to make the visualizations and metrics.

### Authors
- **Reid Passmore**: Ph.D. Canddiate at the Georgia Institute of Technology
- **Ziyi Dai** 
- **Fizzy Fan**
