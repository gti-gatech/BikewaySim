@echo off
:: Activate the Conda environment named "osmium"
call conda activate osmium

:: Enable delayed expansion
setlocal enabledelayedexpansion

:: Set pattern name
set "pattern=*.osm.pbf"

:: Loop through files matching the pattern
for %%f in (%pattern%) do (
    :: Extract the file name without extension
    for %%a in ("%%~nf") do set "name=%%~na"

    :: Clip to studyarea
    osmium extract -p studyarea.geojson "%%f" -o studyarea.osm

    :: Filter to only highway tags
    osmium tags-filter studyarea.osm w/highway -o "!name!.osm"

    :: Export as GeoJSON
    osmium export "!name!.osm" -o "!name!.geojson" -c export-config.json

    :: Delete intermediate files
    del studyarea.osm
)

:: Deactivate the Conda environment
call conda deactivate
