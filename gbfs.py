import heapq

def get_neighbors(node, edges):
    neighbors = []
    for edge, cost in edges.items():
        src, dst = edge
        if src == node:
            neighbors.append((dst, cost))
    return neighbors

def gbfs(start, goals, edges, nodes):
    # Greedy Best-First Search (GBFS) algorithm.

    def heuristic(n):
        # Calculate the heuristic (Euclidean distance) from node n to the closest goal.
        return min(
            ((nodes[g][0] - nodes[n][0]) ** 2 + (nodes[g][1] - nodes[n][1]) ** 2) ** 0.5
            for g in goals
        )
    
    # Initialize the priority queue
    pq = []
    heapq.heappush(pq, (heuristic(start), start, [start], 0))  # Use list for path

    visited = set()
    while pq:
        _, node, path, cost = heapq.heappop(pq)
        if node in visited:
            continue

        visited.add(node)
        
        # Return as soon as a goal is found
        if node in goals:
            return node, len(visited), path  # Return goal node, number of visited nodes, and path
        
        for neighbour, edge_cost in get_neighbors(node, edges):
            if neighbour not in visited:  # Avoid adding visited nodes
                heapq.heappush(pq, (heuristic(neighbour), neighbour, path + [neighbour], cost + edge_cost))
    
    # Return None if no solution is found
    return None
