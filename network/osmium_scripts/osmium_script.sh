#!/bin/bash

# Set pattern name
pattern="*.osm.pbf"

# Create an array to hold the names of the filtered .osm files
osm_files=()

# Loop through files matching the pattern
for f in $pattern; do
    # Extract the file name without extension
    name=$(basename "$f" .osm.pbf)

    # Clip to studyarea
    osmium extract -p studyarea.geojson "$f" -o studyarea.osm

    # Filter to only highway tags
    osmium tags-filter studyarea.osm w/highway -o "${name}.osm"

    # Add the filtered .osm file to the array
    osm_files+=("${name}.osm")

    # Export as GeoJSON
    # osmium export "${name}.osm" -o "${name}.geojson" -c export-config.json

    # Delete intermediate files
    rm studyarea.osm
done

# Merge all the filtered .osm files into one
osmium merge "${osm_files[@]}" -o merged.osm

# Export the merged file as GeoJSON
osmium export merged.osm -o merged.geojson -c export-config.json

# Delete intermediate .osm files
rm "${osm_files[@]}"