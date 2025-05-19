from math import radians, cos, sin, sqrt, atan2

def haversine_distance(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371  # km

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def heuristic(current, destinations, coords):
    current_coord = coords.get(current)
    if not current_coord:
        return 0
    return min(haversine_distance(current_coord, coords[dest]) for dest in destinations if dest in coords)
