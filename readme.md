# BikewaySimDev

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
