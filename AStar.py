
from Heuristics import heuristic
from queue import PriorityQueue

def a_star_search(origin, destinations, edges, coords):
    priority_queue = PriorityQueue()
    priority_queue.put((0, origin))    #origin always has no cost
    came_from = {origin: None} #tracks parent node
    cost = {origin: 0} #lowest calculated cost to reach the end of a node
    nodes_created = 1 
    
    while not priority_queue.empty():
        current = priority_queue.get()[1] #main loop extracting new nodes to que with the lowest priority, current represents the current node evaluated
        
        if current in destinations:

            path = []
            while current is not None:
                path.append(current)
                current = came_from[current] #if a set destination is found during a loop, the path will backtrack and returns the path and total nodes
            path.reverse()
            return path, nodes_created
        
        for neighbor, cost in edges.get(current, []): #calculates the cost to reach a new node
            current_cost = cost[current] + cost
            nodes_created += 1
            
            if neighbor not in cost or current_cost < cost[neighbor]: #determines if a neighbor has a lower tentative cost
                cost[neighbor] = current_cost
                priority = current_cost + heuristic(neighbor, destinations, coords) #f(n) = new cost + heuristics
                priority_queue.put((priority, neighbor)) #adds new neighbor to priority
                came_from[neighbor] = current #records path
    
    return None, nodes_created 