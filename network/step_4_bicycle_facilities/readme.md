# Bike Infrastructure Dating
This module was used for assigning approximate installation/ribbon dates to the various bicycle facilities in and around Atlanta, GA for impedance calibration and trip analysis purposes.

These facility types and dates are attached to the [2023-01-01 Geofabrik PBF extract of Georgia, USA](https://download.geofabrik.de/north-america/us/georgia-230101.osm.pbf). To re-create this network, follow the steps in the `osm_download` module.

## The cities that are at least partially within I-285 that we need bike facility data for:
- Atlanta
- Sandy Springs
- Doraville
- Chamblee
- Decatur
- East Point
- College Park
- Smyrna
- Dunwoody
- Marietta

## Considered Infrastructure:
- **Class I Bike Facilities / Multi-Use Paths / Shared-Use Paths / Off-Street Paths:**
    - Side Paths / Wide Sidewalks DESIGNATED for Bicycle Use
    - PATH Foundation Trails (Stone Mountain Path, Silver Comet Connectors)
    - The Atlanta BeltLine (Eastside/Northside Trails)

- **Class II Bike Facilities / Bike lanes**
    - Traditional painted bike lanes
    - Climbing lanes (United Ave. and 10th St NE between Juniper and Peachstree St)
    - Buffered lanes (Ponce de Leon)

- **Class III Bike Facilities / Sharrows / Signed Bike Routes**
    - Not in consideration for this project

- **Class IV Bike Facilities**
    - Anything on the street and with some form of physical barrier
    - Flex post seperated bicycle lanes (10th St NE and Westview St)
    - I don't think there was anything concrete seperated before 2016 but figure out when the Portman path was opened

## Non-considered
- **Non-motorized paths that are not PATH Foundations/BeltLine funded**
    - Not really sure how to treat these yet as variables
    - Includes Piedmont/Central/Grant Park and Georgia Tech/Emory Paths
    - Assume that these have remained constant over time

- **Intersection Treatments**
    - Intersection treatments like bike boxes, bike signals, and left turn boxes
    - Scramble crossings
    - Not enough of (or any of) these to matter prior to 2017

## Available data:
- OpenStreetMap (no construction dates)
- Atlanta Regional Commission Bicycle Inventory 2024 (has construction dates)
- Mike Garber's inventory that was confirmed through Google StreetView (some construction dates)
- City of Atlanta's inventory acquired in 2021? (has construction dates)

## People to contact to confirm installation dates?:
- Becky Katz, person who followed?
- Rebecca Serna
- People currently in the city?

# Dating Process
This is a semi-automated process. Suggested matches are prepared using scripting but final results are confirmed in QGIS/ArcGIS.

## Steps:
1. Process OSM and **ARC/CoA/Garber** data
    1. Assign a standardized facility type: sharrow, bike lane, buffered bike lane, cycletrack, multi use path

1. Suggested matches from **ARC/CoA/Garber** to **2023 OSM bicycle network**
    1. Buffer **2023 OSM bicycle network**
    1. Intersect with other data source (default = 100 feet)
    1. Check street name where applicable to eliminate one-to-many matches
    1. Check if the bicycle infrastructure matches make sense (e.g. a multi use path matching to a bike lane)
    1. Use street name to accept/eliminate most suggested matches (eliminates a lot of work)
    1. Compare the OSM geometry to the intersected ARC/CoA feature and calculate the Hausdorff distance
    1. Export and finish the rest of the matches in QGIS?
    
    
    1. Replace intersect geometry with **2023 OSM bicycle network** geometry
    1. If bike lane or sharrow, use street name to pre-accept/reject matches
    
    
    
    1. Visually inspect remaining matches in QGIS and determine whether to accept or reject them
    1. Once all matches are one-to-one and appear correct, join to the final **2023 OSM network**

Intersection will include both per one intersect with a feature,
need to think about what it means 


1. Missing infrastructure that is in **ARC/CoA/Garber** but not the **2023 OSM bicycle network** (ID method)
    1. Subset **ARC/CoA/Garber** to features not matched in the previous step
    1. Examine features in QGIS and add suggested matches to OSM network


     that weren't included and decide if they need to be added or if they are already included.


1. Missing infrastructure that is in **ARC/CoA/Garber** but not the **2023 OSM bicycle network** (Geometric Difference)
    1. Buffer **2023 OSM bicycle network** (default = 100 feet)
    1. Find features from other data source that do not intersect with the OSM buffer (difference)
    1. Export and examine in QGIS to determine if it's missing **(MANUAL)**
        1. Mark feature after it has been examined
        1. If missing, add feature id to an editing copy of the **2023 OSM network**
    1. Once all potentially missing features have been addressed, join to the final **2023 OSM network**

1. Finalizing **(MANUAL)**
    1. Join the following datasets to a copy of the **2023 OSM network**
        - **ARC/CoA/Garber** matches
        - Missing infrastructure from **ARC/CoA/Garber**
    1. Desired fields:
        - Facility type by direction
        - Ribbon/construction date
    1. For each feature with a non-null **ARC/CoA/Garber** match,
        - Determine the final facility type by direction and year to use
    1. Once each feature has been examined, export this as the new **2023 OSM bicycle network**
    1. Tag remaining on-road bicycle facilites as post 2016
    1. Leave multi-use trails with no construction date alone
    
1. Notes
    - Nancy Creek Trail does not exist

1. Timing
    - Took me 30 minutes to do 45 edits
    - So 11-ish hours at the current pace

it seems like the name check is a good way to cross off a bunch that we need to confirm

also, it prolly makes sense to edit OSM as i go? there were a few features where i noticed that arc was prolly right

there are going to be some misclassifications, it's inevitable