
import math

def heuristic(node, end_nodes, coords):

    x1, y1 = coords[node]
    end_coords = [(coords[goal][0], coords[goal][1]) for goal in end_nodes]
    return min(
        math.sqrt((x2 - x1)**2 + (y2 - y1)**2) #heuristics distance formula
        for x2, y2 in end_coords
    )
