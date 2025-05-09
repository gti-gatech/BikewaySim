import numpy as np
from shapely.ops import LineString, Point
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import contextily as cx
import rasterio
from shapely.geometry import box, mapping
import requests
from tqdm import tqdm
import laspy
import time

#code credit
#https://github.com/geopandas/geopandas/issues/2279
#https://stackoverflow.com/questions/8247973/how-do-i-specify-an-arrow-like-linestyle-in-matplotlib
from IPython.display import display, clear_output

def sample_lidar(bridge_linkids,interpolated_points_dict,lidar_points,dem_crs,buffer_m=20):
    #create spatial index
    spatial_index = lidar_points.sindex

    for linkid in tqdm(bridge_linkids):
        
        item = interpolated_points_dict.get(linkid)

        geometry = [Point(x,y) for x,y in item['geometry']]
        gdf = gpd.GeoDataFrame({'geometry':geometry},crs=dem_crs)

        #buffer the data
        gdf.geometry = gdf.buffer(buffer_m)

        #get the gdf bounding box
        polygon = gdf.geometry.unary_union.convex_hull
        
        #use spatial index to only select a small number of points
        possible_matches_index = list(spatial_index.intersection(polygon.bounds))
        possible_matches = lidar_points.iloc[possible_matches_index]
        
        #add an index column for the overlay part
        gdf.reset_index(inplace=True)
        precise_matches = gpd.overlay(possible_matches,gdf,how='intersection')

        #take average of all nearby values
        lidar_values = precise_matches.groupby('index')['elevation_m'].mean()
        gdf['new_elevation_m'] = gdf['index'].map(lidar_values)
        lidar_values = np.array(gdf['new_elevation_m'])

        #use nanmax (do this)
        #new_elevations = np.nanmax([lidar_values,item['elevations']],axis = 0)

        #output = elevation_tools.elevation_stats(item['distances'],lidar_values,grade_threshold)

        #replace existing values
        #interpolated_points_dict[linkid]['elevations'] = new_elevations
        interpolated_points_dict[linkid].update({'lidar':lidar_values})

def replace_with_lidar(interpolated_points_dict):
    for linkid, item in interpolated_points_dict.items():
        if item.get('lidar_values',0) != 0:
            new_elevations = np.nanmax([item['lidar'],item['elevations']],axis = 0)
            interpolated_points_dict[linkid]['elevations'] = new_elevations


def download_with_retry(url,MAX_RETRIES,RETRY_DELAY):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = requests.get(url, timeout=10)  # Set your desired timeout
            response.raise_for_status()  # Raise an HTTPError for bad responses
            return response
        except requests.exceptions.Timeout:
            print(f"Timeout error for {url}. Retrying...")
            retries += 1
            time.sleep(RETRY_DELAY)
        except requests.exceptions.RequestException as e:
            print(f"Error for {url}: {e}")
            break

    print(f"Failed to download {url} after {MAX_RETRIES} retries.")
    return None


def get_bridge_decks(lidar_urls):
    '''
    Processes a list of LiDAR URLs to identify and extract bridge deck points.
    This function downloads LiDAR data from the provided URLs, checks for the presence
    of bridge deck points (classification code 17), and compiles these points into a 
    GeoDataFrame with their corresponding elevations.
    
    Parameters:
    lidar_urls (list of str): List of URLs pointing to LiDAR data files.
    
    Returns:
    geopandas.GeoDataFrame or None: A GeoDataFrame containing the geometry and elevation 
    of bridge deck points if any are found, otherwise None.    
    '''
    
    bridge_decks = []
    
    for i, lidar_url in enumerate(lidar_urls):
        
        # if bridge_decks.get(lidar_url,0)!=0:
        #     print('Already exists')
        #     continue

        response = download_with_retry(lidar_url,10,20)
        if response == None:
            continue
        
        las = laspy.read(response.content)

        if (las.classification == 17).any():
            print("Bridge deck found",f"({i+1}/{len(lidar_urls)})")
            # 17 is used for bridge decks https://www.usgs.gov/ngp-standards-and-specifications/lidar-base-specification-tables
            x = np.array(las.x[las.classification == 17])
            y = np.array(las.y[las.classification == 17])
            z = np.array(las.z[las.classification == 17])
            geometry = [Point(x1,y1) for x1, y1 in zip(x,y)]
            geom_value = list(zip(geometry,z))
            bridge_decks = bridge_decks + geom_value
        else:
            print("No bridge deck",f"({i+1}/{len(lidar_urls)})")

    if len(bridge_decks) > 0:
        bridge_decks_gdf = gpd.GeoDataFrame(bridge_decks,columns=['geometry','elevation_m'],geometry='geometry',crs=las.header.parse_crs())
        bridge_decks_gdf.to_crs('epsg:4326',inplace=True)
        return bridge_decks_gdf
    else:
        return None


def interpolate_points(links,interpolate_dist_m):

    interpolated_points_dict = {}

    #takes around 47 seconds for network the size of ITP
    for index, row in links.iterrows():
        
        line = row.geometry

        interpolated_points = []
        interpolated_distances = []

        #start with interpolate_dist_m and add +interpolate_dist_m until current_dist_m is longer than the line
        current_dist_m = interpolate_dist_m

        while current_dist_m < line.length:
            interpolated_point = line.interpolate(current_dist_m)
            interpolated_points += [mapping(interpolated_point)['coordinates']]
            interpolated_distances.append(current_dist_m)
            current_dist_m += interpolate_dist_m

        coords = line.coords
        first_point = coords[0]
        last_point = coords[-1]
        
        # only use first and last point if the line is less than interpolate_dist_m
        if line.length <= interpolate_dist_m:
            interpolated_points = [first_point,last_point]
            distances = [0,line.length]
        else:
            interpolated_points = [first_point] + interpolated_points #+ [last_point]
            distances = [0] + interpolated_distances #+ [line.length]
        
        #TODO maybe just store these all as one array?
        interpolated_points_dict[index] = {
            'geometry': np.array(interpolated_points),
            'distances': np.array(distances),
            # create array of nan values to fill with elevation (m) sampled from the tiff using nanmax
            #UPDATE nanmax doesn't like two np.nans so using -999 instead
            'elevations': np.array([-99999 for x in range(0,len(interpolated_points))])
        }

    return interpolated_points_dict

def get_dem_urls(gdf):
    '''
    Accepts a geodataframe and returns 1-m USGS DEM TIFF download links
    that touch the unary union + envelope of the geodataframe

    Note that you will 
    '''
    gdf_coords = list(gdf.union_all().envelope.exterior.coords)
    gdf_coords = [f"{round(lon,4)}%20{round(lat,4)}" for lon, lat in gdf_coords]
    gdf_coords = ','.join(gdf_coords)
    url = f"https://tnmaccess.nationalmap.gov/api/v1/products?polygon={gdf_coords}&datasets=Digital%20Elevation%20Model%20%28DEM%29%201%20meter&prodExtents=&prodFormats=GeoTIFF&outputFormat=JSON&max=10000"
    response = requests.get(url).json()['items']
    #urls = [item['downloadURL'] for item in response if 'Statewide' in item['downloadURL']]
    urls = [item['downloadURL'] for item in response]
    return urls, response

from bikewaysim import general_utils
def get_lidar_urls(gdf,cell_size_mi=3):
    '''
    Accepts a geodataframe and returns USGS LiDAR LAZ download links
    that touch the unary union + envelope of the geodataframe.

    Uses grids (3 square miles by default to query TNM API to get around the max record limit)
    '''
    # get bounding coordinates of gdf and create a grid
    gdf_coords = list(gdf.union_all().envelope.exterior.coords)
    cells = general_utils.create_grid(gdf,cell_size_mi)
    cells.to_crs('epsg:4326',inplace=True)
    
    # use grid to get the lidar LAZ urls from USGS TNM
    items = []
    for geom in tqdm(cells.geometry):    
        geom_coords = list(geom.exterior.coords)
        geom_coords = [f"{round(lon,4)}%20{round(lat,4)}" for lon, lat in geom_coords]
        geom_coords = ','.join(geom_coords)
        url = f"https://tnmaccess.nationalmap.gov/api/v1/products?polygon={geom_coords}&datasets=Lidar%20Point%20Cloud%20(LPC)&prodFormats=LAS,LAZ&max=10000"
        response = download_with_retry(url,5,2)
        json_response = response.json()
        items.append(json_response)
    
    # process the results
    lidar_urls = []
    for item in items:
        if item['total'] == 0:
            continue
        x = [(y['title'],y['downloadLazURL']) for y in item['items']]
        
        # retrieve only the title and the url
        lidar_urls += x

    return list(set(lidar_urls))

def download_dem(urls,output_fp):

    for i, url in enumerate(urls):
        url = url.strip()
        file_name = url.split('/')[-1]
        file_fp = output_fp / file_name

        if file_fp.exists():
            print(f"File {url.split('/')[-1]} already exists. ({i+1}/{len(urls)})")
        else:
            print(f"Downloading {url.split('/')[-1]}... ({i+1}/{len(urls)})")
            response = requests.get(url)

        if response.status_code == 200:
            with open(file_fp, 'wb') as output_file:
                output_file.write(response.content)
            #print("Download successful.")
        else:
            print(f"Failed to download {url}. Status code: {response.status_code}")


def sample_elevation(tiff_url,links,interpolated_points_dict):
   #open the raster using the link
    src = rasterio.open(tiff_url)

    #find all links that intersect with the current raster
    xmin, ymin, xmax, ymax = src.bounds
    bbox = box(xmin,ymin,xmax,ymax)
    intersection = links.intersects(bbox)

    if (intersection == True).any():
        #print('intersection detected')
        for index in links[intersection].index:
            dict_values = interpolated_points_dict.get(index)

            sampled = np.array([val[0] for val in src.sample(dict_values['geometry'])])
            
            # deal with na values (because the default nan value is -99999)
            #sampled[sampled < 0] = -99999

            #if at least one non-null value
            if np.isnan(sampled).all() == False:
                interpolated_points_dict[index]['elevations'] = np.nanmax([sampled,dict_values['elevations']],axis=0).round(1) 


def point_knockout(item,grade_threshold):
    '''
    When grade exceeding the threshold is detected, remove 
    '''
    
    df_og = pd.DataFrame({'distance':item['distances'],'elevation':item['elevations']})
    
    df = df_og.copy()

    #calculate the percent grade for each section
    df['segment_grade'] = df['elevation'].diff() / df['distance'].diff() * 100

    above_threshold = []

    while (df['segment_grade'].abs() > grade_threshold).any():
        #get index of segments that exceed threshold (selects the second point)
        above_threshold += df[df['segment_grade'].abs() > grade_threshold].index.tolist()
        #subset df
        df = df[(df['segment_grade'].abs() <= grade_threshold) | df['segment_grade'].isna()]
        #repeat grade calc
        df['segment_grade'] = df['elevation'].diff() / df['distance'].diff() * 100

    #change these points to nan
    df_og.loc[df_og.index.isin(above_threshold),'elevation'] = np.nan

    #change elevations in dict
    item['elevations'] = df_og['elevation'].to_numpy()

    return item


def exceeds_threshold(selected_linkids,interpolated_points_dict,grade_threshold):
    '''
    Take in selected links and return which a subsetted version
    where there is at least one 10m segment exceeding the grade
    threshold.
    '''
    exceeds_threshold = []
    for linkid in selected_linkids:
        item = interpolated_points_dict[linkid]
        output = elevation_stats(item['distances'],item['elevations'],grade_threshold)
        if (len(output['bad_ascent_grades']) > 0) | (len(output['bad_descent_grades']) > 0):
            exceeds_threshold.append(linkid)
    #print(len(exceeds_threshold),'/',len(interpolated_points_dict),'links exceed the threshold')
    return exceeds_threshold

def simple_elevation_stats(distances,elevations,unit='m'):
    '''
    Calculates:
    - total ascent (m)
    - total descent (m)
    - ascent grade (%)
    - descent grade (%)
    '''

    #find the total distance to get average grade
    total_distance = distances.max() - distances.min()

    if total_distance < 0:
        print('error')
    
    #caluclate the elevation change between points and add to list
    elevation_deltas = np.diff(elevations)
    # distance_deltas = np.diff(distances)

    #get total ascent and descent
    ascent = elevation_deltas[elevation_deltas > 0].sum()
    descent = elevation_deltas[elevation_deltas < 0].sum()

    #get average percent ascent and descent grade over link distance
    ascent_grade = np.round(ascent / total_distance * 100,2)
    descent_grade = np.round(descent / total_distance * 100,2)

    outputs = {
        f'ascent_{unit}': np.round(ascent,1), # total rise
        f'descent_{unit}': np.round(descent,1), # total descent
        f'ascent_grade_%': ascent_grade, # average ascent grade over link length
        f'descent_grade_%': descent_grade, # average descent grade over link length
    }
    return outputs

def elevation_stats(distances,elevations,grade_threshold=None):
    '''
    Calculates:
    - total ascent (m)
    - total descent (m)
    - ascent grade (%)
    - descent grade (%)
    - index of segments that have an ascent percent grade greater than grade_threshold
    - index of segments that have an descent percent grade greater than -1 * grade_threshold
    - elevation changes for each segment (m)
    - grade for each segment (%) 
    
    '''

    outputs = {}

    total_distance = distances[-1] - distances[0]
    
    #caluclate the elevation change between points and add to list
    elevation_deltas = np.diff(elevations)
    distance_deltas = np.diff(distances)

    outputs['elevation_deltas'] = elevation_deltas
    outputs['distance_deltas'] = distance_deltas
    
    if grade_threshold is not None:
        #calculate grade per section
        segment_grades = elevation_deltas / distance_deltas * 100
        outputs['segment_grades'] = segment_grades
        bad_ascent_grades = np.flatnonzero(segment_grades > grade_threshold)
        bad_descent_grades = np.flatnonzero(segment_grades < -grade_threshold)
        outputs['bad_ascent_grades'] = bad_ascent_grades
        outputs['bad_descent_grades'] = bad_descent_grades

    #get total ascent and descent
    ascent = elevation_deltas[elevation_deltas > 0].sum()
    descent = elevation_deltas[elevation_deltas < 0].sum()
    outputs['ascent'] = ascent
    outputs['descent'] = descent

    #get average percent ascent and descent grade over link's entire distance
    ascent_grade = np.round(ascent / total_distance * 100,2)
    descent_grade = np.round(descent / total_distance * 100,2)
    outputs['ascent_grade'] = ascent_grade
    outputs['descent_grade'] = descent_grade

    # outputs = {
    #     'ascent': ascent, # total rise
    #     'descent':descent, # total descent
    #     'segment_legnth': total_distance, # length used for grade calculations
    #     'ascent_grade': ascent_grade, # average ascent grade over link length
    #     'descent_grade': descent_grade, # average descent grade over link length
    #     'bad_ascent_grades': bad_ascent_grades, # index of segments that exceed the specified grade_threshold
    #     'bad_descent_grades': bad_descent_grades, # index of segments that exceed the specified grade_threshold
    #     'elevation_deltas': elevation_deltas,
    #     'distance_deltas': distance_deltas,
    #     'segment_grades': segment_grades,
    # }

    return outputs

def visualize(links,
              dem_crs,
              interpolated_points_dict,
              list_of_linkids,
              grade_threshold,
              export_filepath,
              one_off=False):
    '''
    # Visualization Function
    This function takes a list of linkids and accesses items from interpolated_points_dict to make a plot showing
    1) the vertical profile
    2) the horizontal alignment with a satellite basemap
    3) the horizontal alignment with a streets basemap
    
    Nodes exceeding the grade threshold (going in the foward direction are highlighted).
    '''
    #TODO:
    # change orinetation of link to be horizontal using azimuth
    # stack maps vertically instead of horizontally
    # have a parameter for controlling vertical exaggeration
    # pull in the raster data for visualization

    fig, (ax1,ax2,ax3) = plt.subplots(ncols=3,figsize=(12,5))

    for linkid in list_of_linkids:

        item = interpolated_points_dict[linkid]

        #Extract values
        x = item['distances']
        y = item['elevations']
        points = np.array([Point(x,y) for x,y in item['geometry']])
        line = LineString(item['geometry'])
        minx, miny, maxx, maxy = line.bounds

        # Plot the original data and the smoothed curve
        ax1.plot(x, y, 'o', label='DEM Data')

        output = elevation_stats(x,y,grade_threshold)

        #Highlight bad grades
        if (len(output['bad_ascent_grades']) > 0) | (len(output['bad_descent_grades']) > 0):

            bad_ascent_grades = output['bad_ascent_grades']
            bad_descent_grades = output['bad_descent_grades']

            #get positions for highlighting grades
            bad_ascent_grades = np.unique(np.hstack([bad_ascent_grades,bad_ascent_grades - 1]))
            bad_descent_grades = np.unique(np.hstack([bad_descent_grades,bad_descent_grades - 1]))
            bad_ascent_x = x[bad_ascent_grades]
            bad_ascent_y = y[bad_ascent_grades]
            bad_descent_x = x[bad_descent_grades]
            bad_descent_y = y[bad_descent_grades]
        
            # plot bad points
            ax1.plot(bad_ascent_x,bad_ascent_y,'o',color='green',label=f'Above {grade_threshold}%')
            ax1.plot(bad_descent_x,bad_descent_y,'o',color='red',label=f'Below -{grade_threshold}%')

            #TODO change to highlight the segment that's bad (both points)
            # use mask to just get bad one
            bad_ascent_grades_points = points[bad_ascent_grades]
            bad_ascent_grades_points = gpd.GeoDataFrame({'geometry':bad_ascent_grades_points},geometry='geometry',crs=dem_crs)
            bad_ascent_grades_points['type'] = f'> {grade_threshold}%'

            bad_descent_grades_points = points[bad_descent_grades]
            bad_descent_grades_points = gpd.GeoDataFrame({'geometry':bad_descent_grades_points},geometry='geometry',crs=dem_crs)
            bad_descent_grades_points['type'] = f'< -{grade_threshold}%'

            bad_grades_points = pd.concat([bad_ascent_grades_points,bad_descent_grades_points])

            color_dict = {
                f'> {grade_threshold}%': 'green',
                f'< -{grade_threshold}%': 'red',
            }
            bad_grades_points['color'] = bad_grades_points['type'].map(color_dict)

            bad_grades_points.plot(ax=ax2,color=bad_grades_points['color'],zorder=4)
            bad_grades_points.plot(ax=ax3,color=bad_grades_points['color'],zorder=4)

        #TODO recalculate bad grades for each to include
        #TODO have it only show new lidar values
        if 'lidar' in item.keys():
            lidar_y = item['lidar']
            mask = y != lidar_y
            ax1.plot(x[mask], lidar_y[mask], 'x',label='Resampled with Lidar')

        if 'smoothed' in item.keys():
            smoothed_y = item['smoothed']
            ax1.plot(x, smoothed_y, '+',label='Smoothed')
        
        ax1.grid(True,linestyle='-.')
        ax1.set_xlabel('Distance (m)')
        ax1.set_ylabel('Elevation (m)')
        ax1.set_ylim(220,360) # TODO make this an input
        ax1.legend()

        #Second and Third figure
        #for drawing the link
        x_coord = np.array([x for x, y in item['geometry']])
        y_coord = np.array([y for x, y in item['geometry']])
        
        # ARROW CODE
        # length of line segment
        ds=10
        # number of line segments per interval
        Ns = np.round(np.sqrt( (x_coord[1:]-x_coord[:-1])**2 + (y_coord[1:]-y_coord[:-1])**2 ) / ds).astype(int)
        # sub-divide intervals w.r.t. Ns
        subdiv = lambda z, Ns=Ns: np.concatenate([ np.linspace(z[ii], z[ii+1], Ns[ii]) for ii, _ in enumerate(z[:-1]) ])
        x_coord, y_coord = subdiv(x_coord), subdiv(y_coord)
        
        ax2.quiver(x_coord[:-1], y_coord[:-1], x_coord[1:]-x_coord[:-1], y_coord[1:]-y_coord[:-1], scale_units='xy', angles='xy', scale=1, width=.004, headlength=4, headwidth=4)
        ax3.quiver(x_coord[:-1], y_coord[:-1], x_coord[1:]-x_coord[:-1], y_coord[1:]-y_coord[:-1], scale_units='xy', angles='xy', scale=1, width=.004, headlength=4, headwidth=4)

        #If you want a legend for map (didnot figure out arrow in the legend)
        # from matplotlib.lines import Line2D
        # from matplotlib.patches import Arrow
        # custom_points = [Line2D([0], [0], marker="o", linestyle="none", markersize=10, color=color) for color in color_dict.values()]
        # #custom_line = Arrow(0,0,dx=0.1,dy=0,width=0.1,linewidth=0.5,color='black')
        # #custom_points.append()
        # leg_points = ax2.legend(custom_points, ['> 15%','< -15%'])#, title = 'Legend', alignment='right')
        # ax2.add_artist(leg_points)

        #make sure fig is square
        padding = 100
        x_diff = np.abs(maxx - minx)
        y_diff = np.abs(maxy - miny)
        diff = (x_diff - y_diff)/2
        if diff > 0:
            ax2.set_xlim(minx-padding,maxx+padding)
            ax2.set_ylim(miny-padding-diff,maxy+padding+diff)

            ax3.set_xlim(minx-padding,maxx+padding)
            ax3.set_ylim(miny-padding-diff,maxy+padding+diff)
        else:
            ax2.set_xlim(minx-padding-np.abs(diff),maxx+padding+np.abs(diff))
            ax2.set_ylim(miny-padding,maxy+padding)

            ax3.set_xlim(minx-padding-np.abs(diff),maxx+padding+np.abs(diff))
            ax3.set_ylim(miny-padding,maxy+padding)

        ax2.set_axis_off()
        ax3.set_axis_off()

        #since we switched data source, let cx figure out the zoom level
        cx.add_basemap(ax2,source=cx.providers.Esri.WorldImagery,crs=dem_crs,alpha=0.5)
        cx.add_basemap(ax3,source=cx.providers.Esri.WorldTopoMap,crs=dem_crs)

        #maybe if we wanted a high res version of this later
        #https://stackoverflow.com/questions/42483449/mapbox-gl-js-export-map-to-png-or-pdf

        name = links.loc[links['osmid']==linkid,'name'].item()
        plt.suptitle(f"{name} ({linkid}) vertical profile (avg ascent grade = {output['ascent_grade']}, descent grade = {output['descent_grade']})")
        
        #just display that one
        if one_off:  
            plt.show()
        else: 
            try: 
                plt.savefig(export_filepath / f"{name}_{linkid}.png",dpi=300)
            except:
                plt.savefig(export_filepath / f"{linkid}.png",dpi=300)
            #clear the axes for next figure
            ax1.cla()
            ax2.cla()
            ax3.cla()


def visualize_one(
        linkid,
        links,
        dem_crs,
        interpolated_points_dict,
        grade_threshold,
        maptilerapikey,
        lidar=False,
        smoothed=False
        ):
    '''
    # Visualization Function
    This function takes a list of linkids and accesses items from interpolated_points_dict to make a plot showing
    1) the vertical profile
    2) the horizontal alignment with a satellite basemap
    3) the horizontal alignment with a streets basemap
    
    Nodes exceeding the grade threshold (going in the foward direction are highlighted).
    '''
    #TODO:
    # change orinetation of link to be horizontal using azimuth
    # stack maps vertically instead of horizontally
    # have a parameter for controlling vertical exaggeration
    # pull in the raster data for visualization

    fig, (ax1,ax2,ax3) = plt.subplots(nrows=3,figsize=(12,30))

    item = interpolated_points_dict[linkid]

    #Extract values
    x = item['distances']
    y = item['elevations']
    points = np.array([Point(x,y) for x,y in item['geometry']])
    line = LineString(item['geometry'])
    minx, miny, maxx, maxy = line.bounds

    # Plot the original data and the smoothed curve
    ax1.plot(x, y, 'o', label='DEM Data')

    output = elevation_stats(x,y,grade_threshold)

    #Highlight bad grades
    if (len(output['bad_ascent_grades']) > 0) | (len(output['bad_descent_grades']) > 0):

        bad_ascent_grades = output['bad_ascent_grades']
        bad_descent_grades = output['bad_descent_grades']

        #get positions for highlighting grades
        bad_ascent_grades = np.unique(np.hstack([bad_ascent_grades,bad_ascent_grades - 1]))
        bad_descent_grades = np.unique(np.hstack([bad_descent_grades,bad_descent_grades - 1]))
        bad_ascent_x = x[bad_ascent_grades]
        bad_ascent_y = y[bad_ascent_grades]
        bad_descent_x = x[bad_descent_grades]
        bad_descent_y = y[bad_descent_grades]
    
        # plot bad points
        ax1.plot(bad_ascent_x,bad_ascent_y,'o',color='green',label=f'Above {grade_threshold}%')
        ax1.plot(bad_descent_x,bad_descent_y,'o',color='red',label=f'Below -{grade_threshold}%')

        #TODO change to highlight the segment that's bad (both points)
        # use mask to just get bad one
        bad_ascent_grades_points = points[bad_ascent_grades]
        bad_ascent_grades_points = gpd.GeoDataFrame({'geometry':bad_ascent_grades_points},geometry='geometry',crs=dem_crs)
        bad_ascent_grades_points['type'] = f'> {grade_threshold}%'

        bad_descent_grades_points = points[bad_descent_grades]
        bad_descent_grades_points = gpd.GeoDataFrame({'geometry':bad_descent_grades_points},geometry='geometry',crs=dem_crs)
        bad_descent_grades_points['type'] = f'< -{grade_threshold}%'

        bad_grades_points = pd.concat([bad_ascent_grades_points,bad_descent_grades_points])

        color_dict = {
            f'> {grade_threshold}%': 'green',
            f'< -{grade_threshold}%': 'red',
        }
        bad_grades_points['color'] = bad_grades_points['type'].map(color_dict)

        bad_grades_points.plot(ax=ax2,color=bad_grades_points['color'],zorder=4)
        bad_grades_points.plot(ax=ax3,color=bad_grades_points['color'],zorder=4)

    #TODO recalculate bad grades for each to include
    #TODO have it only show new lidar values
    if lidar:
        lidar_y = item['lidar']
        mask = y != lidar_y
        ax1.plot(x[mask], lidar_y[mask], 'x',label='Resampled with Lidar')

    if smoothed:
        smoothed_y = item['smoothed']
        ax1.plot(x, smoothed_y, '+',label='Smoothed')
    
    ax1.grid(True,linestyle='-.')
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Elevation (m)')
    ax1.legend()

    #Second and Third figure
    #for drawing the link
    x_coord = np.array([x for x, y in item['geometry']])
    y_coord = np.array([y for x, y in item['geometry']])
    
    # # ARROW CODE
    # length of line segment
    ds=10
    # number of line segments per interval
    Ns = np.round(np.sqrt( (x_coord[1:]-x_coord[:-1])**2 + (y_coord[1:]-y_coord[:-1])**2 ) / ds).astype(int)
    # sub-divide intervals w.r.t. Ns
    subdiv = lambda z, Ns=Ns: np.concatenate([ np.linspace(z[ii], z[ii+1], Ns[ii]) for ii, _ in enumerate(z[:-1]) ])
    x_coord, y_coord = subdiv(x_coord), subdiv(y_coord)
    
    ax2.quiver(x_coord[:-1], y_coord[:-1], x_coord[1:]-x_coord[:-1], y_coord[1:]-y_coord[:-1], scale_units='xy', angles='xy', scale=1, width=.004, headlength=4, headwidth=4)
    ax3.quiver(x_coord[:-1], y_coord[:-1], x_coord[1:]-x_coord[:-1], y_coord[1:]-y_coord[:-1], scale_units='xy', angles='xy', scale=1, width=.004, headlength=4, headwidth=4)

    #If you want a legend for map (didnot figure out arrow in the legend)
    # from matplotlib.lines import Line2D
    # from matplotlib.patches import Arrow
    # custom_points = [Line2D([0], [0], marker="o", linestyle="none", markersize=10, color=color) for color in color_dict.values()]
    # #custom_line = Arrow(0,0,dx=0.1,dy=0,width=0.1,linewidth=0.5,color='black')
    # #custom_points.append()
    # leg_points = ax2.legend(custom_points, ['> 15%','< -15%'])#, title = 'Legend', alignment='right')
    # ax2.add_artist(leg_points)

    # #make sure fig is square
    # padding = 100
    # x_diff = np.abs(maxx - minx)
    # y_diff = np.abs(maxy - miny)
    # diff = (x_diff - y_diff)/2
    # if diff > 0:
    #     ax2.set_xlim(minx-padding,maxx+padding)
    #     ax2.set_ylim(miny-padding-diff,maxy+padding+diff)

    #     ax3.set_xlim(minx-padding,maxx+padding)
    #     ax3.set_ylim(miny-padding-diff,maxy+padding+diff)
    # else:
    #     ax2.set_xlim(minx-padding-np.abs(diff),maxx+padding+np.abs(diff))
    #     ax2.set_ylim(miny-padding,maxy+padding)

    #     ax3.set_xlim(minx-padding-np.abs(diff),maxx+padding+np.abs(diff))
    #     ax3.set_ylim(miny-padding,maxy+padding)

    ax2.set_axis_off()
    ax3.set_axis_off()

    #since we switched data source, let cx figure out the zoom level
    cx.add_basemap(ax2,source=cx.providers.MapTiler.Satellite(key=maptilerapikey),crs=links.crs,alpha=0.5)
    cx.add_basemap(ax3,source=cx.providers.MapTiler.Streets(key=maptilerapikey),crs=links.crs)

    #maybe if we wanted a high res version of this later
    #https://stackoverflow.com/questions/42483449/mapbox-gl-js-export-map-to-png-or-pdf

    name = links.loc[links['osmid']==linkid,'name'].item()
    plt.suptitle(f"{name} ({linkid}) vertical profile (avg ascent grade = {output['ascent_grade']}, descent grade = {output['descent_grade']})")
    
    #just display that one
    plt.show()