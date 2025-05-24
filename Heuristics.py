from math import radians, cos, sin, sqrt, atan2

def haversine_distance(coord1, coord2):
    # Calculate the great-circle distance between two coordinates using the Haversine formula
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371  # Earth radius in kilometers

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def heuristic(current, destinations, coords):
    # Heuristic function: minimum Haversine distance from current node to any destination
    current_coord = coords.get(current)
    if not current_coord:
        return 0
    return min(haversine_distance(current_coord, coords[dest]) for dest in destinations if dest in coords)