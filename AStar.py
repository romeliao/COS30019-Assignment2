from Heuristics import heuristic
from queue import PriorityQueue

def a_star_search(origin, destinations, edges, coords):
    priority_queue = PriorityQueue()
    priority_queue.put((0, origin))
    came_from = {origin: None}
    cost = {origin: 0}
    nodes_created = 1

    while not priority_queue.empty():
        current = priority_queue.get()[1]

        if current in destinations:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path, nodes_created

        for neighbor, edge_cost in edges.get(current, []):
            current_cost = cost[current] + edge_cost

            if neighbor not in cost or current_cost < cost[neighbor]:
                cost[neighbor] = current_cost
                priority = current_cost + heuristic(neighbor, destinations, coords)
                priority_queue.put((priority, neighbor))
                came_from[neighbor] = current
                nodes_created += 1

    return None, nodes_created

def nx_to_edge(graph):
    edges = {}
    for u, v, data in graph.edges(data=True):
        cost = data["weight"]
        edges.setdefault(u, []).append((v, cost))
        edges.setdefault(v, []).append((u, cost))
    return edges

def get_coords(scats_data):
    return {scats: (info["Latitude"], info["Longitude"]) for scats, info in scats_data.items()}
