'''
Throwing in all my visualization code into this .py

'''


########################################################################################

# Visualization and QAQC Tools

########################################################################################

import folium
import geopandas as gpd
from folium.plugins import MarkerCluster, PolyLineTextPath
from folium.map import FeatureGroup
from shapely.ops import Point, MultiLineString, LineString
import branca

from bikewaysim.paths import config
from bikewaysim.impedance_calibration import stochastic_optimization

def construct_line_dict(keys,result_dict,geo_dict):
    """
    Function for creating line dictionary for visualization functions that assumes
    a dictionary with keys corresponding to a dataframe of the link sequence
    """
    line_dict = {}
    for key in keys:
        if key == 'matched_edges':
            new_key = 'Chosen'
        elif key == 'shortest_edges':
            new_key = 'Shortest'
        elif key == 'modeled_edges':
            new_key = 'Modeled'
        else:
            new_key = key
        
        line_dict[new_key] = {
            'links': result_dict[key].values,
            'coords': stochastic_optimization.get_route_line(result_dict[key].values,geo_dict),
        }
    return line_dict

def add_metrics_to_tooltip(line_dict,length_dict,geo_dict):
    '''
    Function used to add various overlap metrics to the line dictionary
    '''

    chosen = line_dict['Chosen']['links']
    shortest = line_dict['Shortest']['links']

    line_dict['Chosen']['detour_pct'] = stochastic_optimization.detour_factor(chosen,shortest,length_dict)

    for key, item in line_dict.items():
        if key == 'Chosen':
            continue
        line = item['links']
        line_dict[key]['jaccard index'] = round(stochastic_optimization.jaccard_exact(chosen,line,length_dict),3)
        line_dict[key]['frechet_dist'] = round(stochastic_optimization.frechet_distance(chosen,line,geo_dict),3)
        line_dict[key]['buffer_dist'] = round(stochastic_optimization.jaccard_buffer(chosen,line,geo_dict),3)
        line_dict[key]['detour_pct'] = round(stochastic_optimization.detour_factor(line,shortest,length_dict),3)

    return line_dict

def basic_three_viz(tripid,results_dict,crs,length_dict,geo_dict,tile_info_dict):
    line_dict = construct_line_dict(['matched_edges','shortest_edges','modeled_edges'],results_dict[tripid],geo_dict)
    line_dict = add_metrics_to_tooltip(line_dict,length_dict,geo_dict)
    mymap = visualize_three(tripid,line_dict,results_dict[tripid]['coords'],crs,tile_info_dict)
    return mymap

def retrieve_geos(x,y,results_dict,links,latlon=False):
    '''
    Pulls out the chosen, shortest, and modeled geometry from results dict that intersect
    with the inputed coordinate
    '''
    links = links.copy()
    links.set_index(['linkid','reverse_link'],inplace=True)
    feature = Point(x,y).buffer(100)
    trips_intersecting_geo = []
    for tripid, item in results_dict.items():
        #get line geo
        shortest_geo = links.loc[[tuple(x) for x in results_dict[tripid]['matched_edges'].values],'geometry'].tolist()
        #test if intersecting
        if MultiLineString(shortest_geo).intersects(feature):
            trips_intersecting_geo.append(tripid)
    return trips_intersecting_geo

import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Extract colors from a ColorBrewer scheme (e.g., 'Set2')
# # Convert them to HEX format if needed
colorbrewer_hex = [colors.to_hex(c) for c in plt.get_cmap('Set2').colors]

#TODO only take the four that have the best metric
from collections import defaultdict
def visualize_three(tripid,match_dict,modeled_dicts,calibration_dicts,geo_dict,coords_dict,crs,tile_info_dict,custom_route=None):
    '''
    Takes in a tripid, a dictionary of map matched and shortest routes, various modeled results, a dictionary
    with the link geometry, a dictionary with the simplified coordinates, and dict with information to retrieve
    the correct tiles
    '''

    qgis_export = {}

    # handle the chosen and the shortest (these should always by provided)
    chosen = match_dict[tripid]['matched_edges'].values
    shortest = match_dict[tripid]['shortest_edges'].values
    chosen = stochastic_optimization.get_route_line(chosen,geo_dict)
    shortest = stochastic_optimization.get_route_line(shortest,geo_dict)
    
    #get start/end and center from the chosen geometry
    line_geo = LineString(chosen)
    line_geo = gpd.GeoSeries(line_geo,crs=crs)
    line_geo = line_geo.to_crs(epsg='4326')
        
    start_pt = list(line_geo.iloc[0].coords)[0]
    end_pt = list(line_geo.iloc[-1].coords)[-1]
    x_mean = line_geo.unary_union.centroid.x
    y_mean = line_geo.unary_union.centroid.y

    # Create a Folium map centered around the mean of the GPS points
    center = [y_mean,x_mean]
    mymap = folium.Map(location=center,
                       zoom_start=15,
                       control_scale=True,
                       tiles=None)
    # add tiles
    folium.TileLayer(**tile_info_dict).add_to(mymap)
    
    # initialize the legend
    legend_lines = ""
    idx = 0 # for the color selection

    # add chosen to folium
    chosen = gpd.GeoDataFrame(
        {'name':'Chosen',
        'length':match_dict[tripid]['chosen_length'],
        'detour':match_dict[tripid]['chosen_detour'],
        'color': colorbrewer_hex[idx],
        'geometry':LineString(chosen)},
        index = [0],
        crs = crs
    ).to_crs('epsg:4326')
    qgis_export['chosen'] = chosen
    chosen = chosen.to_json()
    tooltip = folium.GeoJsonTooltip(fields= ['length','detour'])
    folium.GeoJson(chosen,name='Chosen',
                   style_function=lambda x,
                   color=colorbrewer_hex[idx]: {'color': color, 'weight': 12, 'opacity':0.5},
                   tooltip=tooltip
                   ).add_to(mymap)
    label = 'Chosen'
    legend_lines += f'''
    <p><span style="display:inline-block; background-color:{colorbrewer_hex[idx]}; width:50px; height:10px; vertical-align:middle;"></span>&emsp;{label}</p>
    '''
    idx += 1

    # add shortest to folium
    shortest = gpd.GeoDataFrame(
        {'name':'Shortest',
        'length':match_dict[tripid]['shortest_length'],
        'jaccard_exact':match_dict[tripid]['shortest_jaccard_exact'],
        'jaccard_buffer':match_dict[tripid]['shortest_jaccard_buffer'],
        'color':colorbrewer_hex[idx],
        'geometry':LineString(shortest)},
        index = [0],
        crs = crs
    ).to_crs('epsg:4326')
    qgis_export['shortest'] = shortest
    shortest = shortest.to_json()
    tooltip = folium.GeoJsonTooltip(fields= ['length','jaccard_exact','jaccard_buffer'])
    folium.GeoJson(shortest,name='Shortest',
                   style_function=lambda x,
                   color=colorbrewer_hex[idx]: {'color': color, 'weight': 12, 'opacity':0.5},
                   tooltip=tooltip
                   ).add_to(mymap)
    label = 'Shortest'
    legend_lines += f'''
    <p><span style="display:inline-block; background-color:{colorbrewer_hex[idx]}; width:50px; height:10px; vertical-align:middle;"></span>&emsp;{label}</p>
    '''
    idx += 1
    
    # add the modeled to folium
    subset = {model_name:modeled_dict[tripid] for model_name, modeled_dict in modeled_dicts.items() if modeled_dict.get(tripid) is not None}

    #find max jaccard exact and buffer (i.e. the best match)
    max_jaccard_exact = {model_name:item['modeled_jaccard_exact'] for model_name, item in subset.items()}
    max_jaccard_exact = max(max_jaccard_exact,key=max_jaccard_exact.get)

    max_jaccard_buffer = {model_name:item['modeled_jaccard_buffer'] for model_name, item in subset.items()}
    max_jaccard_buffer = max(max_jaccard_buffer,key=max_jaccard_buffer.get)

    # if these are the same plot as one
    if max_jaccard_buffer == max_jaccard_exact:
        model_name = "Overall Best: " + max_jaccard_exact
        modeled_dict = subset[max_jaccard_exact]
        betas = {x['col']:x['beta'] for x in calibration_dicts[max_jaccard_exact]['betas_tup']}
        color = colorbrewer_hex[idx]
        modeled = modeled_dict['modeled_edges'].values
        modeled = stochastic_optimization.get_route_line(modeled,geo_dict)
        modeled = gpd.GeoDataFrame(
            {'name':model_name,
                'length':modeled_dict['modeled_length'],
                'detour':modeled_dict['modeled_detour'],
                'jaccard_exact':modeled_dict['modeled_jaccard_exact'],
                'jaccard_buffer':modeled_dict['modeled_jaccard_buffer'],
                **betas,
                'color':colorbrewer_hex[idx],
                # TODO need calibration coefficients and objective function type
                'geometry':LineString(modeled)},
            index = [0],
            crs = crs
        ).to_crs('epsg:4326')
        qgis_export[model_name] = modeled
        modeled = modeled.to_json()
        tooltip = folium.GeoJsonTooltip(fields= ['name','length','detour','jaccard_exact','jaccard_buffer',*list(betas.keys())])
        folium.GeoJson(modeled,name=model_name,
                    style_function=lambda x,
                    color=color: {'color': color, 'weight': 12, 'opacity':0.5},
                    tooltip=tooltip
                    ).add_to(mymap)
        legend_lines += f'''
        <p><span style="display:inline-block; background-color:{color}; width:50px; height:10px; vertical-align:middle;"></span>&emsp;{model_name}</p>
        '''
        idx += 1

    else:

        model_name = "Exact Best: " + max_jaccard_exact
        modeled_dict = subset[max_jaccard_exact]
        betas = {x['col']:x['beta'] for x in calibration_dicts[max_jaccard_exact]['betas_tup']}
        color = colorbrewer_hex[idx]
        modeled = modeled_dict['modeled_edges'].values
        modeled = stochastic_optimization.get_route_line(modeled,geo_dict)
        modeled = gpd.GeoDataFrame(
            {'name':model_name,
                'length':modeled_dict['modeled_length'],
                'detour':modeled_dict['modeled_detour'],
                'jaccard_exact':modeled_dict['modeled_jaccard_exact'],
                'jaccard_buffer':modeled_dict['modeled_jaccard_buffer'],
                **betas,
                'color':colorbrewer_hex[idx],
                # TODO need calibration coefficients and objective function type
                'geometry':LineString(modeled)},
            index = [0],
            crs = crs
        ).to_crs('epsg:4326')
        qgis_export[model_name] = modeled
        modeled = modeled.to_json()
        tooltip = folium.GeoJsonTooltip(fields= ['name','length','detour','jaccard_exact','jaccard_buffer',*list(betas.keys())])
        folium.GeoJson(modeled,name=model_name,
                    style_function=lambda x,
                    color=color: {'color': color, 'weight': 12, 'opacity':0.5},
                    tooltip=tooltip
                    ).add_to(mymap)
        legend_lines += f'''
        <p><span style="display:inline-block; background-color:{color}; width:50px; height:10px; vertical-align:middle;"></span>&emsp;{model_name}</p>
        '''
        idx += 1

        model_name = "Buffer Best" + max_jaccard_buffer
        modeled_dict = subset[max_jaccard_buffer]
        betas = {x['col']:x['beta'] for x in calibration_dicts[max_jaccard_exact]['betas_tup']}
        color = colorbrewer_hex[idx]
        modeled = modeled_dict['modeled_edges'].values
        modeled = stochastic_optimization.get_route_line(modeled,geo_dict)
        modeled = gpd.GeoDataFrame(
            {'name':model_name,
                'length':modeled_dict['modeled_length'],
                'detour':modeled_dict['modeled_detour'],
                'jaccard_exact':modeled_dict['modeled_jaccard_exact'],
                'jaccard_buffer':modeled_dict['modeled_jaccard_buffer'],
                **betas,
                'color':colorbrewer_hex[idx],
                # TODO need calibration coefficients and objective function type
                'geometry':LineString(modeled)},
            index = [0],
            crs = crs
        ).to_crs('epsg:4326')
        qgis_export[model_name] = modeled
        modeled = modeled.to_json()
        tooltip = folium.GeoJsonTooltip(fields= ['name','length','detour','jaccard_exact','jaccard_buffer',*list(betas.keys())])
        folium.GeoJson(modeled,name=model_name,
                    style_function=lambda x,
                    color=color: {'color': color, 'weight': 12, 'opacity':0.5},
                    tooltip=tooltip
                    ).add_to(mymap)
        legend_lines += f'''
        <p><span style="display:inline-block; background-color:{color}; width:50px; height:10px; vertical-align:middle;"></span>&emsp;{model_name}</p>
        '''
        idx += 1

    # add the rest of the modeled results as a feature group
    modeled_fg = folium.FeatureGroup(name=f"All Modeled (N={len(modeled_dict)})",show=False)
    color = colorbrewer_hex[idx]
    for model_name, modeled_dict in subset.items():
  
        modeled = modeled_dict['modeled_edges'].values
        betas = {x['col']:x['beta'] for x in calibration_dicts[max_jaccard_exact]['betas_tup']}
        modeled = stochastic_optimization.get_route_line(modeled,geo_dict)
        modeled = gpd.GeoDataFrame(
            {'name':model_name,
             'length':modeled_dict['modeled_length'],
             'detour':modeled_dict['modeled_detour'],
             'jaccard_exact':modeled_dict['modeled_jaccard_exact'],
             'jaccard_buffer':modeled_dict['modeled_jaccard_buffer'],
             **betas,
             'color':colorbrewer_hex[idx],
             # TODO need calibration coefficients and objective function type
             'geometry':LineString(modeled)},
            index = [0],
            crs = crs
        ).to_crs('epsg:4326')
        qgis_export[model_name] = modeled
        modeled = modeled.to_json()
        tooltip = folium.GeoJsonTooltip(fields= ['name','length','detour','jaccard_exact','jaccard_buffer',*list(betas.keys())])
        folium.GeoJson(modeled,name=model_name,
                    style_function=lambda x,
                    color=color: {'color': color, 'weight': 12, 'opacity':0.5},
                    tooltip=tooltip,
                    highlight_function=lambda x: {'color': 'yellow', 'weight': 20}
                    ).add_to(modeled_fg)
    legend_lines += f'''
    <p><span style="display:inline-block; background-color:{color}; width:50px; height:10px; vertical-align:middle;"></span>&emsp;All Modeled (N={len(modeled_dicts)})</p>
    '''
    modeled_fg.add_to(mymap)
    idx += 1

    # add the trace coordinates so we can see if there is a map matching error
    coords = coords_dict[tripid]
    coords = [Point(x) for x in coords]
    coords = gpd.GeoDataFrame({'color':'#a48f91','geometry':coords},crs=crs)
    coords.to_crs('epsg:4326',inplace=True)
    qgis_export['coords'] = coords
    coords = coords.to_json()
    coords = folium.GeoJson(coords,
                            name='coords',
                            marker=folium.Circle(radius=10, fill_color='#a48f91', fill_opacity=1, color="black", weight=0),
                            show=False).add_to(mymap)

    # Add start and end points with play and stop buttons
    start_icon = folium.Icon(color='green',icon='play',prefix='fa')
    end_icon = folium.Icon(color='red',icon='stop',prefix='fa')
    folium.Marker(location=[start_pt[1], start_pt[0]],icon=start_icon).add_to(mymap)
    folium.Marker(location=[end_pt[1], end_pt[0]],icon=end_icon).add_to(mymap)
    od = gpd.GeoDataFrame({'type': ['start','end'],'color': ['#41d22e','#dc4451'], 'geometry': [Point(start_pt),Point(end_pt)]},crs='epsg:4326')
    qgis_export['od'] = od
    
    if custom_route is not None:
        custom_route.add_to(mymap)
        legend_lines += f'''
        <p><span style="display:inline-block; background-color:yellow; width:50px; height:10px; vertical-align:middle;"></span>&emsp;Custom Route</p>
        '''

    # Add layer control to toggle layers on/off
    folium.LayerControl(collapsed=False).add_to(mymap)
    legend_html = f'''    
    {{% macro html(this, kwargs) %}}               
    <div style="
        position: fixed; 
        bottom: 50px; left: 10px; width: auto; height: auto; 
        z-index:9999; font-size:14px; background-color: white; 
        border:2px solid grey; padding: 10px; opacity: 0.9;">
        <p>Trip ID: {tripid}</p>
        {legend_lines}
    {{% endmacro %}}
    </div>
    '''
    legend = branca.element.MacroElement()
    legend._template = branca.element.Template(legend_html) 
    mymap.get_root().add_child(legend)

    format_qgis(qgis_export)

    return mymap

import pandas as pd
def format_qgis(qgis_export):
    '''
    Exports the layers in the folium map to qgis
    '''

    chosen = qgis_export['chosen'].to_crs(config['projected_crs_epsg'])
    shortest = qgis_export['shortest'].to_crs(config['projected_crs_epsg'])
    best_modeled = pd.concat([item for key, item in qgis_export.items() if 'Best' in key],axis=0,ignore_index=True).to_crs(config['projected_crs_epsg'])
    other_modeled = pd.concat([item for key, item in qgis_export.items() if key not in ('od','coords','chosen','shortest')],axis=0,ignore_index=True).to_crs(config['projected_crs_epsg'])
    ods = qgis_export['od'].to_crs(config['projected_crs_epsg'])
    coords_gdf = qgis_export['coords'].to_crs(config['projected_crs_epsg'])
    
    chosen.to_file(config['calibration_fp']/'viz.gpkg',layer='chosen')
    shortest.to_file(config['calibration_fp']/'viz.gpkg',layer='shortest')
    best_modeled.to_file(config['calibration_fp']/'viz.gpkg',layer='best modeled')
    other_modeled.to_file(config['calibration_fp']/'viz.gpkg',layer='other modeled')
    ods.to_file(config['calibration_fp']/'viz.gpkg',layer='ods')
    coords_gdf.to_file(config['calibration_fp']/'viz.gpkg',layer='coords')

#TODO add turns for these visuals
# folium.GeoJson(
#     modeled_turn_type.to_crs(epsg='4326').to_json(),
#     name="Modeled Turn Types",
#     show=False,
#     tooltip=folium.GeoJsonTooltip(fields=['turn_type','signalized','unsig_major_road_crossing']),
#     marker=folium.Circle(radius=30, fill_color="orange", fill_opacity=0.5, color="black", weight=0),
#     style_function=lambda x: {
#         'fillColor':turn_dict[x['properties']['signalized']]
#     }
# ).add_to(m)
# turn_type_colors = {
#     'left': 'orange',
#     'straight': 'black',
#     'right': 'red',
#     'u-turn': 'black'
# }



# def visualize_route_attributes(
#         tripid,
#         results_dict, # contains the edge list etc
#         matched_gdf,
#         modeled_gdf,
#         links_df,
#         turns_df,
#         nodes_df,
#         route_attribute_cols,
#         ):

#     '''
#     This function displays the matched vs shortest/modeled route for a particular trip
#     It also displays the trip characteristics side be side and plots the any signalized
#     intersections and stressful turns passed through.
#     '''

#     # Create copies to prevent alteration
#     matched_gdf = matched_gdf.copy()
#     modeled_gdf = modeled_gdf.copy()

#     # Subset data to relevant trip
#     matched_gdf = matched_gdf[matched_gdf['tripid']==tripid]
#     modeled_gdf = modeled_gdf[modeled_gdf['tripid']==tripid]
    
#     # Create a Folium map centered around the mean of the matched route
#     minx, miny, maxx, maxy = matched_gdf.to_crs(epsg='4326').total_bounds
#     x_mean = (maxx - minx) / 2 + minx
#     y_mean = (maxy - miny) / 2 + miny
#     center = [y_mean,x_mean]
#     m = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")
    
#     # Add GeoJSON data to FeatureGroups
#     folium.GeoJson(matched_gdf.to_crs(epsg='4326').to_json(),
#                    name='Matched',
#                    tooltip=folium.GeoJsonTooltip(fields=route_attribute_cols),
#                    style_function=lambda x: {'color': 'red'}).add_to(m)
    
#     folium.GeoJson(modeled_gdf.to_crs(epsg='4326').to_json(),
#                    name='Modeled',
#                    tooltip=folium.GeoJsonTooltip(fields=route_attribute_cols),
#                    style_function=lambda x: {'color': 'blue'}).add_to(m)

#     # Get the start and end points
#     start_node = results_dict[tripid]['origin_node']
#     start_node = nodes_df.to_crs('epsg:4326').loc[nodes_df['N']==start_node,'geometry'].item()

#     end_node = results_dict[tripid]['destination_node']
#     end_node = nodes_df.to_crs('epsg:4326').loc[nodes_df['N']==end_node,'geometry'].item()

#     # Add start and end points with play and stop buttons to map
#     #start_icon = folium.Icon(color='green',icon='play',prefix='fa')
#     #end_icon = folium.Icon(color='red',icon='stop',prefix='fa')
#     folium.Marker(location=[start_pt.y, start_pt.x],color='green').add_to(m)
#     folium.Marker(location=[end_pt.y, end_pt.x],color='red').add_to(m)

#     # Add signals and turns for matched route

#     edges = match_dict[tripid]['edges']
#     list_of_edges = list(zip(edges['linkid'],edges['reverse_link']))
#     list_of_turns = [(list_of_edges[i][0],list_of_edges[i][1],list_of_edges[i+1][0],list_of_edges[i+1][1]) for i in range(0,len(list_of_edges)-1)]


#     #from these we want to get the locations and number of singalized intersections and stressful crossing passed through
    
#     df_of_turns = pd.DataFrame(list_of_turns,columns=['source_linkid','source_reverse_link','target_linkid','target_reverse_link'])
#     subset = pseudo_df.merge(df_of_turns,on=['source_linkid','source_reverse_link','target_linkid','target_reverse_link'])

#     # from this subset we can get the right node ids
#     #TODO turns should be by edges probably?
#     #turns = subset[['source_B','turn_type']]
#     signals = subset.loc[subset['signalized']==True,'source_B'].value_counts()
#     two_way_stops = subset.loc[subset['unsignalized']==True,'source_B'].value_counts()

#     #and then get the correct rows of the gdf
#     #turns = nodes.merge(signals,left_on='N',right_on='')
#     signals = nodes.merge(signals,left_on='N',right_index=True)
#     signals.columns = ['N','geometry','num_times']
#     two_way_stops = nodes.merge(two_way_stops,left_on='N',right_index=True)
#     two_way_stops.columns = ['N','geometry','num_times']

#     # get the start and end point for plotting
#     start_N = gdf.loc[gdf['tripid']==tripid,'start'].item()
#     start_pt = nodes.to_crs('epsg:4326').loc[nodes['N']==start_N,'geometry'].item()
#     end_N = gdf.loc[gdf['tripid']==tripid,'end'].item()
#     end_pt = nodes.to_crs('epsg:4326').loc[nodes['N']==end_N,'geometry'].item()





 
#    # Add FeatureGroups to the map


#    if signals.shape[0] > 0:
#       signals_geojson = signals.to_crs(epsg='4326').to_json()
#       signals_fg = FeatureGroup(name='Signals')

#       folium.GeoJson(
#       signals_geojson,
#       name="Traffic Signal Turn Movement",
#       marker=folium.Circle(radius=20, fill_color="red", fill_opacity=.5, color="black", weight=1),
#       tooltip=folium.GeoJsonTooltip(fields=['N','num_times']),
#       popup=folium.GeoJsonPopup(fields=['N','num_times']),
#       #    style_function= lambda feature: {
#       #        'fillColor': colormap(feature['properties']['speed_mph']),
#       #    },
#       highlight_function=lambda feature: {"color":"yellow","weight":3}
#       ).add_to(signals_fg)
#       signals_fg.add_to(mymap)

#    if two_way_stops.shape[0] > 0:
#       two_way_stops_geojson = two_way_stops.to_crs(epsg='4326').to_json()
#       two_way_stops_fg = FeatureGroup(name='Two Way Stop (chosen)')

#       folium.GeoJson(
#       two_way_stops_geojson,
#       name="Two Way Stop with High Stress Cross Street",
#       marker=folium.Circle(radius=20, fill_color="yellow", fill_opacity=.5, color="black", weight=1),
#       tooltip=folium.GeoJsonTooltip(fields=['N','num_times']),
#       popup=folium.GeoJsonPopup(fields=['N','num_times']),
#       #    style_function= lambda feature: {
#       #        'fillColor': colormap(feature['properties']['speed_mph']),
#       #    },
#       highlight_function=lambda feature: {"color":"yellow","weight":3}
#       ).add_to(two_way_stops_fg)

#       two_way_stops_fg.add_to(mymap)




#    #autofit content not in this version?
#    #folium.FitOverlays().add_to(mymap)

#    # Add layer control to toggle layers on/off
#    folium.LayerControl().add_to(mymap)

#    #retrive overlap
#    exact_overlap = gdf.loc[gdf['tripid']==tripid,'shortest_exact_overlap_prop'].item()
#    buffer_overlap = gdf.loc[gdf['tripid']==tripid,'shortest_buffer_overlap'].item()

#    attr = gdf.loc[gdf['tripid']==tripid].squeeze()

#    # Add legend with statistics
#    legend_html = f'''
#    <div style="position: fixed; 
#             bottom: 5px; left: 5px; width: 300px; height: 500px; 
#             border:2px solid grey; z-index:9999; font-size:14px;
#             background-color: white;
#             opacity: 0.9;">
#    &nbsp; <b>Tripid: {tripid}</b> <br>
#    &nbsp; Start Point &nbsp; <i class="fa fa-play" style="color:green"></i><br>
#    &nbsp; End Point &nbsp; <i class="fa fa-stop" style="color:red"></i><br>
#    &nbsp; Exact Overlap: {exact_overlap*100:.2f}% <br>
#    &nbsp; Buffer Overlap: {buffer_overlap*100:.2f}% <br>

#    &nbsp; Trip Type: {attr['trip_type']} <br>
#    &nbsp; Length (mi): {attr['length_ft']/5280:.0f} <br>
#    &nbsp; Age: {attr['age']} <br>
#    &nbsp; Gender: {attr['gender']} <br>
#    &nbsp; Income: {attr['income']} <br>
#    &nbsp; Ethnicity: {attr['ethnicity']} <br>
#    &nbsp; Cycling Frequency: {attr['cyclingfreq']} <br>
#    &nbsp; Rider History: {attr['rider_history']} <br>
#    &nbsp; Rider Type: {attr['rider_type']} <br><br>

#    &nbsp; Residential %: {attr['highway.residential']*100:.2f}% <br>
#    &nbsp; Secondary %: {attr['highway.secondary']*100:.2f}% <br>
#    &nbsp; Tertiary %: {attr['highway.tertiary']*100:.2f}% <br>

#    &nbsp; # of bridges: {int(attr['bridge'])} <br>
#    &nbsp; # of left turns: {int(attr['left'])} <br>
#    &nbsp; # of straight turns: {int(attr['straight'])} <br>
#    &nbsp; # of right turns: {int(attr['right'])} <br>
#    &nbsp; # of stressful turns: {int(attr['unsignalized'])} <br>
#    &nbsp; # of signalized turns: {int(attr['signalized'])} <br>

#    </div>
#    '''

#    mymap.get_root().html.add_child(folium.Element(legend_html))

#    # Save the map to an HTML file or display it in a Jupyter notebook
#    #mymap.save('map.html')
#    # mymap.save('/path/to/save/map.html')  # Use an absolute path if needed
#    return mymap  # Uncomment if you are using Jupyter notebook

   #TODO add in the legend with trip info and then we're golden




# DEPRECATED
# def loss_function(betas,betas_links,betas_turns,links,
#                        pseudo_links,pseudo_G,
#                        matched_traces,link_impedance_function,
#                        turn_impedance_function,exact,follow_up):

#     #use initial/updated betas to calculate link costs
#     print('setting link costs')
#     links = link_impedance_function(betas, betas_links, links)
#     cost_dict = dict(zip(links['linkid'],links['link_cost']))
    
#     #add link costs to pseudo_links
#     pseudo_links['source_link_cost'] = pseudo_links['source_linkid'].map(cost_dict)
#     pseudo_links['target_link_cost'] = pseudo_links['target_linkid'].map(cost_dict)

#     #use initial/updated betas to calculate turn costs
#     print('setting turn costs')
#     pseudo_links = turn_impedance_function(betas, betas_turns, pseudo_links)

#     #add the source edge, target edge, and turn costs
#     #TODO experiment with multiplying the turn cost
#     pseudo_links['total_cost'] = pseudo_links['source_link_cost'] + pseudo_links['target_link_cost'] + pseudo_links['turn_cost']

#     #only keep link with the lowest cost
#     print('finding lowest cost')
#     costs = pseudo_links.set_index(['source','target'])['total_cost']

#     #update edge weights
#     print('updating edge weights')
#     nx.set_edge_attributes(pseudo_G,values=costs,name='weight')

#     #update edge ids (what was this for?)
    
#     #do shortest path routing
#     shortest_paths = {}
#     print(f'Shortest path routing with coefficients: {betas}')    
#     for source, targets in matched_traces.groupby('start')['end'].unique().items():

#         #add virtual links to pseudo_G
#         pseudo_G, virtual_edges = modeling_turns.add_virtual_links(pseudo_links,pseudo_G,source,targets)
        
#         #perform shortest path routing for all target nodes from source node
#         #(from one to all until target node has been visited)
#         for target in targets:  
#             #cant be numpy int64 or throws an error
#             target = int(target)
            
#             try:
#                 #TODO every result is only a start node, middle node, then end node
#                 length, node_list = nx.single_source_dijkstra(pseudo_G,source,target,weight='weight')
#             except:
#                 print(source,target)
#                 length, node_list = nx.single_source_dijkstra(pseudo_G,source,target,weight='weight')

#             #get edge list
#             edge_list = node_list[1:-1]

#             #get geometry from edges
#             modeled_edges = links.set_index(['source','target']).loc[edge_list]

#             # modeled_edges = links.merge(linkids.loc[edge_list],on=['linkid','reverse_link'],how='inner')
#             # modeled_edges = gpd.GeoDataFrame(modeled_edges,geometry='geometry')

#             shortest_paths[(source,target)] = {
#                 'edges': set(modeled_edges['linkid'].tolist()),
#                 'geometry':MultiLineString(modeled_edges['geometry'].tolist()),#modeled_edges.dissolve()['geometry'].item(),
#                 'length':MultiLineString(modeled_edges['geometry'].tolist()).length
#                 }

#         #remove virtual links
#         pseudo_G = modeling_turns.remove_virtual_edges(pseudo_G,virtual_edges)
    
#     print('calculating objective function')

#     #turn shortest paths dict to dataframe
#     shortest_paths = pd.DataFrame.from_dict(shortest_paths,orient='index')
#     shortest_paths.reset_index(inplace=True)
#     shortest_paths.columns = ['start','end','linkids','geometry','length']
#     #shortest_paths[['start','end']] = shortest_paths['index'].apply(lambda x: pd.Series(x))
#     #shortest_paths.drop(columns=['index'],inplace=True)

#     #add modeled paths to matched_traces dataframe
#     merged = matched_traces.merge(shortest_paths,on=['start','end'],suffixes=(None,'_modeled'))

#     if exact:
#         sum_all = merged['length'].sum() * 5280
#         all_overlap = 0

#         for idx, row in merged.iterrows():
#             #find shared edges
#             chosen_and_shortest = row['linkids_modeled'] & row['linkids']
#             #get the lengths of those links
#             overlap_length = links.set_index('linkid').loc[list(chosen_and_shortest)]['length_ft'].sum()
#             #overlap_length = np.sum([link_lengths.get(link_tup,'error') for link_tup in chosen_and_shortest])
#             all_overlap += overlap_length

#         #calculate objective function value
#         val = all_overlap / sum_all
#         print('Exact overlap percent is:',np.round(val*100,1),'%')
    
#     #calculate approximate overlap (new approach)
#     else:
#         #buffer and dissolve generated route and matched route
#         buffer_ft = 500

#         merged.set_geometry('geometry',inplace=True)
#         merged['buffered_geometry'] = merged.buffer(buffer_ft)
#         merged.set_geometry('buffered_geometry',inplace=True)
#         merged['area'] = merged.area

#         merged.set_geometry('geometry_modeled',inplace=True)
#         merged['buffered_geometry_modeled'] = merged.buffer(buffer_ft)
#         merged.set_geometry('buffered_geometry_modeled',inplace=True)
#         merged['area_modeled'] = merged.area

#         #for each row find intersection between buffered features
#         merged['intersection'] = merged.apply(lambda row: row['buffered_geometry'].intersection(row['buffered_geometry_modeled']), axis=1)

#         # merged['intersection'] = merged.apply(
#         #     lambda row: shapely.intersection(row['buffered_geometry'],row['buffered_geometry_modeled']))
#         merged.set_geometry('intersection',inplace=True)
#         merged['intersection_area'] = merged.area

#         #find the overlap with the total area (not including intersections)
#         #if the modeled/chosen links are different, then overlap decreases
#         #punishes cirquitious modeled routes that utilize every link in the chosen one but include extraneous ones
#         merged['overlap'] = merged['intersection_area'] / (merged['area_modeled'] + merged['area'] - merged['intersection_area'])

#         #find average overlap (using median to reduce impact of outliers?)
#         val = merged['overlap'].median()
#         print('Median overlap percent is:',np.round(val*100,1),'%')
    
#     if follow_up:
#         return merged

#     return -val#, merged



# def visualize(tripid,gdf,nodes):

#    '''
#    This function displays the matched vs shortest route for a particular trip
#    It also displays the trip characteristics side be side and plots the any signalized
#    intersections and stressful turns passed through.
#    '''

#    #gdf contains all the trips and the trip gemometries as mutlilinestrings
#    gdf = gdf.copy()

#    # Your GeoDataFrames
#    chosen_path = gdf.loc[gdf['tripid']==tripid,['tripid','geometry']]
#    shortest_path = gdf.loc[gdf['tripid']==tripid,['tripid','shortest_geo']].set_geometry('shortest_geo').set_crs(gdf.crs)
#    intersection = gdf.loc[gdf['tripid']==tripid,['tripid','shortest_intersect_geo']].set_geometry('shortest_intersect_geo').set_crs(gdf.crs)

#    #from these we want to get the locations and number of singalized intersections and stressful crossing passed through
#    edges = match_dict[tripid]['edges']
#    list_of_edges = list(zip(edges['linkid'],edges['reverse_link']))
#    list_of_turns = [(list_of_edges[i][0],list_of_edges[i][1],list_of_edges[i+1][0],list_of_edges[i+1][1]) for i in range(0,len(list_of_edges)-1)]
#    df_of_turns = pd.DataFrame(list_of_turns,columns=['source_linkid','source_reverse_link','target_linkid','target_reverse_link'])
#    subset = pseudo_df.merge(df_of_turns,on=['source_linkid','source_reverse_link','target_linkid','target_reverse_link'])

#    # from this subset we can get the right node ids
#    #TODO turns should be by edges probably?
#    #turns = subset[['source_B','turn_type']]
#    signals = subset.loc[subset['signalized']==True,'source_B'].value_counts()
#    two_way_stops = subset.loc[subset['unsignalized']==True,'source_B'].value_counts()

#    #and then get the correct rows of the gdf
#    #turns = nodes.merge(signals,left_on='N',right_on='')
#    signals = nodes.merge(signals,left_on='N',right_index=True)
#    signals.columns = ['N','geometry','num_times']
#    two_way_stops = nodes.merge(two_way_stops,left_on='N',right_index=True)
#    two_way_stops.columns = ['N','geometry','num_times']

#    # get the start and end point for plotting
#    start_N = gdf.loc[gdf['tripid']==tripid,'start'].item()
#    start_pt = nodes.to_crs('epsg:4326').loc[nodes['N']==start_N,'geometry'].item()
#    end_N = gdf.loc[gdf['tripid']==tripid,'end'].item()
#    end_pt = nodes.to_crs('epsg:4326').loc[nodes['N']==end_N,'geometry'].item()

#    # Create a Folium map centered around the mean of the chosen route
#    x_mean = chosen_path.to_crs(epsg='4326').geometry.item().centroid.x
#    y_mean = chosen_path.to_crs(epsg='4326').geometry.item().centroid.y
#    center = [y_mean,x_mean]
#    mymap = folium.Map(location=center, zoom_start=14)

#    # Convert GeoDataFrames to GeoJSON
#    chosen_path_geojson = chosen_path.to_crs(epsg='4326').to_json()
#    shortest_path_geojson = shortest_path.to_crs(epsg='4326').to_json()
#    intersection_geojson = intersection.to_crs(epsg='4326').to_json()

#    # Create FeatureGroups for each GeoDataFrame
#    chosen_path_fg = FeatureGroup(name='Chosen Path')
#    shortest_path_fg = FeatureGroup(name='Shortest Path',show=False)
#    intersection_fg = FeatureGroup(name='Buffer Intersection',show=False)

#    # Add GeoJSON data to FeatureGroups
#    folium.GeoJson(chosen_path_geojson, name='Chosen Path', style_function=lambda x: {'color': 'red'}).add_to(chosen_path_fg)
#    folium.GeoJson(shortest_path_geojson, name='Shortest Path', style_function=lambda x: {'color': 'blue'}).add_to(shortest_path_fg)
#    folium.GeoJson(intersection_geojson, name='Buffer Intersection', style_function=lambda x: {'color': 'yellow'}).add_to(intersection_fg)

#    # Add FeatureGroups to the map
#    chosen_path_fg.add_to(mymap)
#    shortest_path_fg.add_to(mymap)
#    intersection_fg.add_to(mymap)

#    if signals.shape[0] > 0:
#       signals_geojson = signals.to_crs(epsg='4326').to_json()
#       signals_fg = FeatureGroup(name='Signals')

#       folium.GeoJson(
#       signals_geojson,
#       name="Traffic Signal Turn Movement",
#       marker=folium.Circle(radius=20, fill_color="red", fill_opacity=.5, color="black", weight=1),
#       tooltip=folium.GeoJsonTooltip(fields=['N','num_times']),
#       popup=folium.GeoJsonPopup(fields=['N','num_times']),
#       #    style_function= lambda feature: {
#       #        'fillColor': colormap(feature['properties']['speed_mph']),
#       #    },
#       highlight_function=lambda feature: {"color":"yellow","weight":3}
#       ).add_to(signals_fg)
#       signals_fg.add_to(mymap)

#    if two_way_stops.shape[0] > 0:
#       two_way_stops_geojson = two_way_stops.to_crs(epsg='4326').to_json()
#       two_way_stops_fg = FeatureGroup(name='Two Way Stop (chosen)')

#       folium.GeoJson(
#       two_way_stops_geojson,
#       name="Two Way Stop with High Stress Cross Street",
#       marker=folium.Circle(radius=20, fill_color="yellow", fill_opacity=.5, color="black", weight=1),
#       tooltip=folium.GeoJsonTooltip(fields=['N','num_times']),
#       popup=folium.GeoJsonPopup(fields=['N','num_times']),
#       #    style_function= lambda feature: {
#       #        'fillColor': colormap(feature['properties']['speed_mph']),
#       #    },
#       highlight_function=lambda feature: {"color":"yellow","weight":3}
#       ).add_to(two_way_stops_fg)

#       two_way_stops_fg.add_to(mymap)


#    # Add start and end points with play and stop buttons
#    start_icon = folium.Icon(color='green',icon='play',prefix='fa')
#    end_icon = folium.Icon(color='red',icon='stop',prefix='fa')
#    folium.Marker(location=[start_pt.y, start_pt.x],icon=start_icon).add_to(mymap)
#    folium.Marker(location=[end_pt.y, end_pt.x],icon=end_icon).add_to(mymap)

#    #autofit content not in this version?
#    #folium.FitOverlays().add_to(mymap)

#    # Add layer control to toggle layers on/off
#    folium.LayerControl().add_to(mymap)

#    #retrive overlap
#    exact_overlap = gdf.loc[gdf['tripid']==tripid,'shortest_exact_overlap_prop'].item()
#    buffer_overlap = gdf.loc[gdf['tripid']==tripid,'shortest_buffer_overlap'].item()

#    attr = gdf.loc[gdf['tripid']==tripid].squeeze()

#    # Add legend with statistics
#    legend_html = f'''
#    <div style="position: fixed; 
#             bottom: 5px; left: 5px; width: 300px; height: 500px; 
#             border:2px solid grey; z-index:9999; font-size:14px;
#             background-color: white;
#             opacity: 0.9;">
#    &nbsp; <b>Tripid: {tripid}</b> <br>
#    &nbsp; Start Point &nbsp; <i class="fa fa-play" style="color:green"></i><br>
#    &nbsp; End Point &nbsp; <i class="fa fa-stop" style="color:red"></i><br>
#    &nbsp; Exact Overlap: {exact_overlap*100:.2f}% <br>
#    &nbsp; Buffer Overlap: {buffer_overlap*100:.2f}% <br>

#    &nbsp; Trip Type: {attr['trip_type']} <br>
#    &nbsp; Length (mi): {attr['length_ft']/5280:.0f} <br>
#    &nbsp; Age: {attr['age']} <br>
#    &nbsp; Gender: {attr['gender']} <br>
#    &nbsp; Income: {attr['income']} <br>
#    &nbsp; Ethnicity: {attr['ethnicity']} <br>
#    &nbsp; Cycling Frequency: {attr['cyclingfreq']} <br>
#    &nbsp; Rider History: {attr['rider_history']} <br>
#    &nbsp; Rider Type: {attr['rider_type']} <br><br>

#    &nbsp; Residential %: {attr['highway.residential']*100:.2f}% <br>
#    &nbsp; Secondary %: {attr['highway.secondary']*100:.2f}% <br>
#    &nbsp; Tertiary %: {attr['highway.tertiary']*100:.2f}% <br>

#    &nbsp; # of bridges: {int(attr['bridge'])} <br>
#    &nbsp; # of left turns: {int(attr['left'])} <br>
#    &nbsp; # of straight turns: {int(attr['straight'])} <br>
#    &nbsp; # of right turns: {int(attr['right'])} <br>
#    &nbsp; # of stressful turns: {int(attr['unsignalized'])} <br>
#    &nbsp; # of signalized turns: {int(attr['signalized'])} <br>

#    </div>
#    '''

#    mymap.get_root().html.add_child(folium.Element(legend_html))

#    # Save the map to an HTML file or display it in a Jupyter notebook
#    #mymap.save('map.html')
#    # mymap.save('/path/to/save/map.html')  # Use an absolute path if needed
#    return mymap  # Uncomment if you are using Jupyter notebook