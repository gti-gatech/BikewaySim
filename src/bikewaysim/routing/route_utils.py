'''
Functions for retrieving link geometries from a list of edges
'''
import numpy as np

#retrieve coordinates, reversing coordinate sequence if neccessary
def retrieve_coordinates(link,geo_dict):
    line = np.array(geo_dict[link[0]].coords)
    if link[1] == True:
        line = line[::-1]
    return line

def get_correct_link_direction(link_coord_seq,reverse_link):
    '''
    Flips link coordinates according to specified direction
    '''
    if reverse_link:
        return link_coord_seq[::-1]
    else:
        return link_coord_seq

def get_route_line(route,geo_dict): 
    '''
    Takes in a list of links in (linkid:int,reverse_link:bool) format and returns a
    list with the routes coordinates so it can be turned into a LineString
    '''
    
    #get all the links
    route = [get_correct_link_direction(geo_dict.get(linkid[0],False).coords,linkid[1]) for linkid in route]
    #remove the last point of each link except for the last one
    route = [x[0:-1] if idx != len(route) - 1 else x for idx, x in enumerate(route)]
    #flatten to produce one linestring
    route = [x for xs in route for x in xs]
    return route