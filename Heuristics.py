def heuristic(node, destinations, coords):
    x1, y1 = coords[node]
    return min(
        ((coords[dest][0] - x1) ** 2 + (coords[dest][1] - y1) ** 2) ** 0.5
        for dest in destinations
    )