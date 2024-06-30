# Running BikewaySim

Components:
1. Impedance routing for all possible TAZ/census block pairs OR specified ODs on existing network
1. Impedance routing for all possible TAZ/census block pairs OR specified ODs on future network
1. Weighting OD pairs by (employments + population)
1. Calculate and report difference in impedances across all ODs
1. Report impedance reduction from each improved link (via whether the improved link was utilized or not)
    - Essentially a betweenness centrality reduction
1. Rank projects
1. Visualize impedance reduction by block/taz

Optional Components:
1. Visualize before/after routes
1. Inspect percent detour