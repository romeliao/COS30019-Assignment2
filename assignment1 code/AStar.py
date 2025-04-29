from Heuristics import heuristic
from queue import PriorityQueue

def a_star_search(origin, destinations, edges, coords):
    priority_queue = PriorityQueue()
    priority_queue.put((0, origin))    # origin always has no cost
    came_from = {origin: None}  # tracks parent node
    cost = {origin: 0}  # lowest calculated cost to reach the end of a node
    nodes_created = 1 
    
    while not priority_queue.empty():
        current = priority_queue.get()[1]  # main loop extracting new nodes to queue with the lowest priority

        if current in destinations:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]  # Backtrack to reconstruct the path
            path.reverse()
            return path, nodes_created
        
        for neighbor, edge_cost in edges.get(current, []):  # Get neighbors
            current_cost = cost[current] + edge_cost
            nodes_created += 1
            
            if neighbor not in cost or current_cost < cost[neighbor]:  # Check if a lower cost is found
                cost[neighbor] = current_cost
                priority = current_cost + heuristic(neighbor, destinations, coords)  # f(n) = g(n) + h(n)
                priority_queue.put((priority, neighbor))
                came_from[neighbor] = current  # Record the path
    
    print("No path found.")  # Debugging
    return None, nodes_created