from Heuristics import heuristic
def hill_climbing_search(origin, destinations, edges, nodes):
    #inittialization
    current = origin                #starting point
    path = [current]                #tracks coords in an array
    nodes_created = 1               #keeps track of nodes visited
    visited = set([current])        #visited nodes will be appended to an array set
    
    while current not in destinations:
        neighbors = edges.get(current, []) #retrieves other nodes nearby origin point or current pooint
        if not neighbors:
            return None, nodes_created  #return none if no nodes were found and the current amount of nodes created
        
        best_neighbor = None

        best_heuristic = float('inf')
        
        for neighbor, _ in neighbors:
            if neighbor in visited:
                continue
                
            h = heuristic(neighbor, destinations, nodes) #calculates the nearest euclidean distance using heuristics
            if h < best_heuristic:
                best_heuristic = h #if distance is lower than best heuristic, it will become the new best route
                best_neighbor = neighbor
        
        if best_neighbor is None:
            return None, nodes_created #termination if no best neighbor is found
        
        current = best_neighbor #sequence: move to best -> updates searched path -> increments node
        path.append(current)
        visited.add(current)
        nodes_created += 1
    
    return path, nodes_created