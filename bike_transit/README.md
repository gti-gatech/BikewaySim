# TransitSim for Bike + Transit Routing

This folder contains the code needed to re-create the analyses in the NCST Report, ["Simulating Bike-Transit Trips through BikewaySim"](https://ncst.ucdavis.edu/project/simulating-bike-transit-trips-through-bikewaysim-and-transitsim) and a paper accepted for publication in the Transportation Research Record, "Assessing Bike-Transit Accessibility."

Before proceeding make sure to install these additional dependencies:
1. 
```
pip install partridge
```




NOTE: This code uses a forked version of the [transit-routing](https://github.com/transnetlab/transit-routing) repository. The [forked repository](https://github.com/reidx19/transit-routing) adds the pareto set (i.e., the optimal transit routes given a certain number of transfers) as an output to the funciton "RAPTOR.raptor_functions.post_processing." This output is used for creating spatial data later on and creating the visuals. The forked repository also contains a pre-processed set of MARTA GTFS data that was used in the study. 

The forked repository may not be up-to-date with the main [transit-routing](https://github.com/transnetlab/transit-routing).




## Code Directory

1. transitsim_processing.py
    - create_study_area
        - Buffers all transit stops according to given threshold
        - Intersects this buffered area with TAZs
        - Exports study area and TAZ centroids
    - add_new_routeid
        - Transit-Routing takes the GTFS data processed with transit-routing and matches the new routeids and stopids assigned in transit-routing with the GTFS ones to create a new transit stops shapefile that has the new stop ids
    - gtfs_shapefiles
        - Uses [partridge](https://github.com/remix/partridge) to create a transit routes shapefile from zipped GTFS data
1. find_candidate_stops.py
    - get_candidate_stops
        - Finds transit stops that are within the specified access threshold of TAZs
    - ckdnearest
        - Helper function used for snapping transit stops and TAZs to nearest network node
    - snap_to_network
        - Matches points to network nodes
    - create_graph
        - Turns an edge list into a networkx directed graph
        - Impedance or link costs column must be specified
    - find_shortest_bike
        - For performing shortest path routing between TAZs
    - find_shortest
        - For performing shortets path routing from TAZs to candidate transit stops AND from candidate transit stops to TAZs
    - get _path_geo2
        - Gets the geometric length of the shortest paths
        - If the impedance is distance then they will be the same
    - create_raptor_inputs
        - Create a parquet file of dataframe with six columns: start_taz, end_taz, start_transit, end_transit, distance_to, distance_from, impedance_to, impedance_from
    - export files
        - Exports the shortest paths results from find_shortest
    - bikewalksheds
        - Export TAZs that are bikeable/walkable
    - raptor_preprocessing
        - Master funciton for running the above functions
1. transitsim_raptor_routing.py
    - get_times
        - Used for generating departure times for RAPTOR
    - check_type
        - Makes sure the TAZ id is a string
    - run_raptor
        - Runs the raptor algorithm given the files created in the previous script
1. transitsim_raptor_mapping.py
    - check_type
        - Duplicate need to remove
    - get_path_geo
        - Turns a dict of edge lists into linestrings
    - map_routes
        - Take the outputs from RAPTOR and map out the trip as a collection of linestrings that show all legs of the trip
    - NOTE, appears to have duplicates with the preprocessing stage
1. viz_and_metrics.py
    - For creating visuals from the outputted raptor files
    - access_map
        - Output all the accessible TAZs as a geopackage
    - mode_map
        - Output the mode used to reach each TAZ
    - dot_map
        - Export a geopackage of TAZs with the total travel time
    - transit_shed
        - Dissolve all the transit lines taken and export as geopackage
1. quick conflation.py
    - Code used for adding HERE attributes to OSM network