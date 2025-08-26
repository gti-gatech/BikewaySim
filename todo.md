## Propel Action Plan (08/21/2025)
- start with the small study area for speed
- create/filter the network
- export the network in the correct format
    - might need to roll back to a previous commit sometime in november when i had everything working
- hard code the weights as the default condition
- run the analysis with LODES first

### 08/21/2025
- Network module is up and running

## Next Steps
- Go back to an earlier version to better understand how the calibration_network.pkl was made

Network Files Explainer

| Name | Description |
| - | - |
| directed_links.parquet | contains directional attributes like incline and bike facility |
| final_network_edges.pkl | Non-directed edges with all other attributes and the edge geometries |
| final_network.gpkg, nodes | Geopackage with the nodes, just IDs and geometry |
| turns_df.parquet | table of turns with turn attributes like left turn/right turn/unig_crossing/etc. |




# Have a network sync function
- objective is to allow people to skip map matching and calibration if they don't have GPS traces
- in map matching
    - we allow for the removal of certain links in order to get better map matching results
- in calibration
    - we allow for the removal of certain links that are not useful for routing or that could impact the results
- when people skip straight to the framework
    - i want to be able to provide default impedances that people can use
    - so they'd do a direct import from the network section instead of doing it in the calibration section
- recommendations
    - move prepare for calibration out to network where it makes sense


## Map Matching
- Combine Map_Matching.ipynb and Step_0
- Figure out how to sort the interactive components from the scripted ones
- Process that can sort through the different match results to retrieve the best one found
- Current version keeps all the settings tested and then creates a new file for each setting tested. Not sure if there's a check for excluded links


## Impedance Calibration
- decide if certain parts of step_0 need to be placed in map_matching because it's unclear

## BikewaySim Framework
- figure out where the different network exports went
- script certain parts
- make maplibre dashboard style report for propel

## New module that focusses on aggregated statistics on links?



# find where I used to create/export these networks
    networks = [
        config['bikewaysim_fp']/'current_traveltime.pkl',
        config['bikewaysim_fp']/'current_impedance.pkl',
        config['bikewaysim_fp']/'future_impedance.pkl'
    ]

# Small changes
subsets.pkl to trips_for_calibration.pkl