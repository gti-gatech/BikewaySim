import numpy as np
from shapely.ops import LineString, Point
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import contextily as cx

#code credit
#https://github.com/geopandas/geopandas/issues/2279
#https://stackoverflow.com/questions/8247973/how-do-i-specify-an-arrow-like-linestyle-in-matplotlib
from IPython.display import display, clear_output

def point_knockout(item,grade_threshold):
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

    # #fit a spline
    # spline = splrep(df['distance'], df['elevation'], s=0.5)

    # #add spline to dict
    # item['spline'] = spline


def exceeds_threshold(selected_linkids,interpolated_points_dict,grade_threshold):
    '''
    Take in selected links and return which a subsetted version
    where there is at least one 10m segment exceeding the grade
    threshold.
    '''
    exceeds_threshold = []
    for linkid in selected_linkids:
        item = interpolated_points_dict.get(linkid,0)
        output = elevation_stats(item['distances'],item['elevations'],grade_threshold)
        if len(output['bad_ascent_grades']) > 0 | len(output['bad_descent_grades']) > 0:
            exceeds_threshold.append(linkid)
    print(len(exceeds_threshold),'/',len(interpolated_points_dict),'links exceed the threshold')
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
    total_distance = distances.max()
    
    #caluclate the elevation change between points and add to list
    elevation_deltas = np.diff(elevations)
    distance_deltas = np.diff(distances)

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

def elevation_stats(distances,elevations,grade_threshold,key_prefix=''):
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

    #find the total distance to get average grade
    total_distance = distances.max()
    
    #caluclate the elevation change between points and add to list
    elevation_deltas = np.diff(elevations)
    distance_deltas = np.diff(distances)
    
    #calculate grade per 10 m section
    segment_grades = elevation_deltas / distance_deltas * 100
    bad_ascent_grades = np.flatnonzero(segment_grades > grade_threshold)
    bad_descent_grades = np.flatnonzero(segment_grades < -grade_threshold)

    #get total ascent and descent
    ascent = elevation_deltas[elevation_deltas > 0].sum()
    descent = elevation_deltas[elevation_deltas < 0].sum()

    #get average percent ascent and descent grade over link distance
    ascent_grade = np.round(ascent / total_distance * 100,2)
    descent_grade = np.round(descent / total_distance * 100,2)

    outputs = {
        f'{key_prefix}ascent': ascent, # total rise
        f'{key_prefix}descent':descent, # total descent
        f'{key_prefix}ascent_grade': ascent_grade, # average ascent grade over link length
        f'{key_prefix}descent_grade': descent_grade, # average descent grade over link length
        f'{key_prefix}bad_ascent_grades': bad_ascent_grades, # index of segments that exceed the specified grade_threshold
        f'{key_prefix}bad_descent_grades': bad_descent_grades, # index of segments that exceed the specified grade_threshold
        f'{key_prefix}elevation_deltas': elevation_deltas,
        f'{key_prefix}distance_deltas': distance_deltas,
        f'{key_prefix}segment_grades': segment_grades,
    }

    return outputs

def visualize(links,dem_crs,interpolated_points_dict,list_of_linkids,grade_threshold,export_filepath,maptilerapikey,one_off=False,lidar=False,smoothed=False):
    '''
    # Visualization Function
    This function takes a list of linkids and accesses items from interpolated_points_dict to make a plot showing
    1) the vertical profile
    2) the horizontal alignment with a satellite basemap
    3) the horizontal alignment with a streets basemap
    
    Nodes exceeding the grade threshold (going in the foward direction are highlighted).
    '''

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

        # key_prefix = ''
        # if lidar:
        #     key_prefix = 'lidar_'
        # if smoothed:
        #     key_prefix = 'smoothed_'
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
        cx.add_basemap(ax2,zoom=17,source=cx.providers.MapTiler.Satellite(key=maptilerapikey),crs=links.crs,alpha=0.5)
        cx.add_basemap(ax3,zoom=16,source=cx.providers.MapTiler.Streets(key=maptilerapikey),crs=links.crs)

        #maybe if we wanted a high res version of this later
        #https://stackoverflow.com/questions/42483449/mapbox-gl-js-export-map-to-png-or-pdf

        name = links.loc[links['osmid']==linkid,'name'].item()
        plt.suptitle(f'{name} ({linkid}) vertical profile (avg ascent grade = {output['ascent_grade']}, descent grade = {output['descent_grade']})')
        
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