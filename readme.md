# BikewaySim
BikewaySim is a collection of open-source scripts and notebooks for helping cities, DOTs, MPOs, and/or advocacy groups assess the connectivity impacts of new cycling facilities (bike lanes and cycletracks) through change in cycling impedance.
 
Cycling impedance measures the relative difficulty of cycling from point A to point B considering travel time, elevation/hills, exposure to automobile traffic, the presence of bicycle facilities, and other preferences that cyclists have for road attributes.

BikewaySim's cycling impedance model is trained on recorded cycling trips. The cycling impedance model can then be used to find the least cycling impedance route (Figure 1) between any two places (home to grocery, school to park, etc.). On its own, this model can be used by cities to help citizens find better cycling routes and encourage cycling.

<figure>
    <img src="resources\impedance_example.jpeg" width=350>
    <figcaption><b>Figure 1:</b> Example of BikewaySim impedance routing (purple) and travel time only routing (black)</figcaption>
</figure>

When paired with BikewaySim's bicycle facility assessment framework (Figure 2), the cycling impedance model can be used to evaluate proposed cycling facilities using five metrics/visuals using the change in simulated impedance from the bicycle facilities. Cities, DOTs, MPOs, and/or advocacy groups can use the outputs of the framework to prioritize new cycling facilities accordingly.

<figure>
    <img src="resources\framework_workflow.jpeg" width=350>
    <figcaption><b>Figure 2:</b> BikewaySim bicycle facility assessment framework</figcaption>
</figure>

<!-- Add descriptions for each metric -->
<!-- | Metric | Description | Example
|-|-|-|
| Trip Impedance Reduction | Impedance reduction shows TAZs that | <img src="resources/origin weighted impedance change.jpeg" width=200> |
| Percent Detour | | <img src="resources/percent detour.jpeg" width=200> |
| Change in Link Betweenness Centrality | | <img src="resources/lbc future - current.jpeg" width=200> |
| Improvement Impedance Reduction | | <img src="resources/impedance contribution.jpeg" width=200> |
| Bikesheds | | <img src="resources/bikeshed add remove.jpeg" width=200> | -->

## Main functionalities:
1. Downloading and processing OpenStreetMap network data for bicycle routing using [Geofabrik](https://www.geofabrik.de) and [OSMnx](https://github.com/gboeing/osmnx)
1. Conflating and reconciling attributes from other network sources to OSM
1. Map matching cycling GPS traces to OSM using [Leuven Map Matching](https://github.com/wannesm/LeuvenMapMatching)
1. Calibrating link and turn impedance functions for cycling using bicycling GPS traces or count data using [stochopy](https://github.com/keurfonluu/stochopy)
1. Finding the least impedance route for selected or all-to-all O-D pairs given calibrated or custom link and turn impedance functions using [NetworkX](https://networkx.org) and [rustworkx](https://www.rustworkx.org)
1. Using least impedance routes to generate five metrics for assessing the impacts of planned cycling facilities

## Data Requirements
BikewaySim is intended to work in the United States with minimal data as it pulls network data from OpenStreetMap and elevation data from USGS. However, it is highly recommended that you provide more detailed network attribute data and cycling GPS traces so that cycling impedances can be calibrated for your study area (as the default ones may not be applicable to your study area since they were calibrated for Atlanta, GA).

Required:
- A study area in shapefile, geojson, geodatabase, or geopackage format (bounding boxes also work)
- Cycling infrastructure improvements in shapefile, geojson, geodatabase, or geopackage format

Optional:
- Supplemental network attribute data from government, public, or private sources to be conflated to OpenStreetMap data (see Data Used for examples)
- Cycling GPS traces for calibrating cycling impedances 

## Installation:
- Assuming that you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) and [git](https://git-scm.com/downloads) installed for your machine already
- Clone the repository (and the transit-routing submodule if desired) into your desired directory
```
# without TransitSim
git clone https://github.com/reidx19/BikewayDev 

# with TransitSim
git clone --recurse-submodules -j8 https://github.com/reidx19/BikewayDev 
```
- Create a new conda environment named `bikewaysim` using the `environment.yml` file (make sure that you're in the directory with the .yml file)
```
conda env create -f environment.yml
```
- Activate the 'bikewaysim' environment
```
conda activate bikewaysim
```
- Change into the `BikewaySim/src` folder on command line and install the development package `bikewaysim`
```
pip install -e .
```
- Great! The main things should be installed! If you're having troubles with creating the environment, open the `environment.yml` and try installing the packages one at a time instead in a new environment instead.
- Most of the code is executed through Jupyter Notebooks which can be opened/run/edited through [VS Code](https://code.visualstudio.com/) or [Jupyter Notebook](https://anaconda.org/anaconda/jupyter), just make sure the `bikewaysim` environment is activated.

## How to Run
1. Create a [config.json](config.json) file to define the project directory and various settings (e.g., desired projected coordinate system, supplemental data directories, etc.). See `example_config.json` for a template.
1. Create a new study area using GIS or [bounding box](https://boundingbox.klokantech.com) or provide an existing one. Must be in `.geojson`, `.gpkg`, or `.shp` format.
1. Download OpenStreetMap network data using [Step_0_Process_OSM.ipynb](./network/Step_0_Process_OSM.ipynb).

## Quick Start Bike Routing
1. Continue running through each .ipynb file in the network module in order
1. Use the bicycle facilities module to add in bicycle facilities from other data sources and OpenStreetMap
1. Run the notebooks in the [bikewaysim_framework](./bikewaysim_framework/) module to route cycling trips using default or custom impedance factors

## Quick Start Bike-Transit Routing
1. Clone the [transit-routing](https://github.com/reidx19/transit-routing) submodule
1. Open [Simulating Bike-Transit Trips.ipynb](./bike_transit/Simulating%20Bike-Transit%20Trips.ipynb) and walk through the steps until a study area is generated
1. Open [Downloading_OSM.ipynb](./osm_download/Downloading_OSM.ipynb) and use the generated study area to download OpenStreetMap network data
1. Continue with [Simulating Bike-Transit Trips.ipynb](./bike_transit/Simulating%20Bike-Transit%20Trips.ipynb)

## BikewaySim Facility 

1. Open  and follow the instructions to download OpenStreetMap network data
1. Open [Step_1 Network_Filtering_and_Processing.ipynb](./network/Step_1_Network_Filtering.ipynb) to process OSM into network graph format and identify link types
1. Open [Step_2_Network_Reconciliation.ipynb](./network/Step_2_Network_Reconciliation.ipynb) to finalize the network for routing
1. Open [Run_BikewaySim.ipynb](./bikewaysim/Run_BikewaySim.ipynb) to perform aggregated shortest path routing and output Geopackage files to visualize in GIS software

### List of Modules

| Module | Description | Status |
|-|-|-|
| bicycle_facilities | Add supplemental bicycle facility data | working |
| bike-transit| Run aggregated bike-transit shortest path calculations | working |                    
| bikewaysim_framework | Run aggregated cycling shortest path calculations and create  | working |
| map_matching | Map-match GPS trace data | working  |   
| impedance_calibration  | Use map-matched GPS traces to calibrate link impedance functions | in development |
| network | Download OSM data and process for shortest path routing | working

<!-- ## Assessing Bike-Transit Accessibility
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
This final notebook runs the Dijkstra shortest path routing algorithm for the given OD pairs. Results are exported to file and processed to make the visualizations and metrics. -->

## Data Used

## Publications
1. Passmore, R., K., Watkins, and R. Guensler (2024). Using Shortest Path Routing to Assess Cycling Networks. *Journal of Transportation Geography*. DOI: [10.1016/j.jtrangeo.2024.103864](https://doi.org/10.1016/j.jtrangeo.2024.103864)
1. Passmore, R., K. Watkins, and R. Guensler (2024). Assessing Bike-Transit Accessibility. *Transportation Research Record*. DOI: [10.1177/03611981241234902](https://doi.org/10.1177/03611981241234902)
1. Passmore, R., K. Watkins, and R. Guensler (2024). Siumulating Bike-Transit Trips Through BikewaySim and TransitSim. *National Center for Sustainable Transportation*. DOI: [10.7922/G22R3Q0B](http://dx.doi.org/10.7922/G22R3Q0B)
1. Passmore, R., K. Watkins, and R. Guensler (2021). BikewaySim Technology Transfer: City of Atlanta Georgia. *National Center for Sustainable Transportation*. DOI: [10.7922/G2CF9NDV](https://dx.doi.org/10.7922/G2CF9NDV)

## Acknowledgements
BikewaySim was developed as part of Reid Passmore's Ph.D. dissertation. BikewaySim was developed with funding from the National Center for Sustainable Transportation and the Georgia Department of Transportation. 

## Authors
**Reid Passmore**, Ph.D. Candidate<sup>1</sup><br />
**Fizzy Fan**, Ph.D. Candidate<sup>1</sup><br />
**Dr. Ziyi Dai**,<br />
**Dr. Randall Guensler**, Professor<sup>1</sup><br />
**Dr. Kari Watkins**, Associate Professor<sup>2</sup><br />

<sup>1</sup>School of Civil and Environmental Engineering, Georgia Institute of Technology<br />
<sup>2</sup>School of Civil and Environmental Engineering, University of California, Davis<br />