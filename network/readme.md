# Network Module
This module is used to process network data into a routable network graph format. If no network data is available, then use the "osm_download" module to download OSM network data. Once network data is available follow the following steps:

## Step 1 Network Filtering and processing.ipynb
This Jupyter notebook takes in spatial data on links and nodes from a provided street network (the code will create nodes if none are provided) in GeoJSON/Geopackage/Geodatabase/Shapefile format and the same study area GeoDataFrame used in Step 0. It filters the street network into five different layers (raw, general, road, bike, and service) and formats the street network into a routable network graph. The supporting script for this notebook is "prepare_network.py". The filters required for distinguqshing the five layers are defined in the notebook. If a network that is not HERE, ABM, or OSM is provided, new filters will need to be defined.

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
The Jupyter notebook generates link impedances and prepares the reconciled network for routing. This step is dependent on the available networks, network attribute data, and desired link impedances.
