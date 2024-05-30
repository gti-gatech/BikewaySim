# Elevation Sampling
This readme describes the process for adding elevation data to an OSM navigable network.

## DEM and Lidar Download
Use

## Sampling
Elevation data for each OSM link is sampled from the intersecting DEM. User must specify a sampling distance, which will specify how frequently elevation is sampled along a link. Recomended value is 10 meters. Elevation will be sampled up until the last point (not including the last point unless it's within the sampling distance). For links that are 10 meters or less, only the elevation at first and last point are sampled.

Cut and fill also not considered (new developments with roads that weren't there when the DEM data were recorded)

## Link Labelling
Using OSM tags, we can pre-identify a few examples in which there will likely be extreme grade changes:

- Bridges
- Underpasses (intersect with bridges)
- Tunnels (we can't sample new data for these)
- Highway tag
    - Pedestrian links include un-paved hiking trails that can easily exceed certain grades

## Point Knockout
As mentioned in Liu (2018), sampling straight from a DEM will produce errors. DEM is representation of the ground, meaning that bridges and other elevated features won't carry their actual elevation. In addition, there's no guarantee that link points are going to line up correctly with the DEM. The default sampling method used is to assign the elevation value of the DEM pixel that the interpolated point falls into.

The general percent grade formula is: $ rise / run * 100 $. The $run$ in this case is the sampling distance (default: 10m) and the rise would be: $elevation_{x} - elevation_{x+1}$. Roads have standards for how steep they can be. In Liu (2018), highways and Interstates had a max grade of 8% and local roads a max grade of 15%. It is not specifed what segment length is used.

In Liu (2018), segment grades are calculated, the first segement exceeding the specified threshold is selected and a backwards search is initiated starting 30m, 100m, and 150m to identify the last point of the erroneous elevation sampling. Points between these two points are removed and the process continues past the search bracket.

This code instead removes all segements that exceed the threshold until there are no longer segments that exceed the threshold. In some cases, this will remove too many points, in which case the link is flagged for further inspection. These links either need new sampled points from bridge lidar, are actually the grade specified, or might be too short of a link for the specified approach to work.

Note, some roads are simply steep, and using this point knockout method could incorrectly knock out reasonable segments, so point knockout should really start with using a **high grade threshold > 30%** to prevent the removal of accurate segments.

## Bridge Re-Sampling
After knocking out errouneous points, resample links that are identified as bridges from the lidar data. Then re-run the point knockout code 