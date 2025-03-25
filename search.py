import sys
import heapq

#Function to read file and parse input like nodes, edges, origin and destination.
def parse_input(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')  # read file and split by new line

    # initialize data structure
    nodes, edges, origin, destination = {}, {}, None, []
    section = None

    # process each line in the file
    for line in lines:
        line = line.strip()
        if line.startswith("Nodes"):
            section = "nodes"
        elif line.startswith("Edges"):
            section = "edges"
        elif line.startswith("Origin"):
            section = "origin"
        elif line.startswith("Destination"):
            section = "destination"

        # checks node coordinates
        elif section == "nodes":
            parts = line.split(": ")
            node_id = int(parts[0])  # convert node_id to Int
            coordinates = tuple(map(int, parts[1][1:-1].split(",")))#converts X and Y coordinates to tuple
            nodes[node_id] = coordinates  # stores in dictionary

        # checks edges and weight
        elif section == "edges":
            parts = line.split(": ")
            if len(parts) != 2 or not parts[0].startswith("(") or not parts[0].endswith(")"):
                print(f"Skipping invalid edge line: {line}")
                continue
            try:
                edge = tuple(map(int, parts[0][1:-1].split(','))) #extract node as tuple
                weight = int(parts[1])
                edges[edge] = weight
            except ValueError:
                print(f"Skipping invalid edge line: {line}")
                continue

        # stores origin node and converts it to an Int
        elif section == "origin":
            origin = int(line)

        # Stores destination node and converts it to a list of Int
        elif section == "destination":
            destination = list(map(int, line.replace(" ", "").split(';')))  # Handle semicolon-separated values

    # returns the processed data
    return nodes, edges, origin, destination

def get_neighbors(node, edges):
    neighbors = {}
    for edge, cost in edges.items():
        src, dst = edge
        if src == node:
            neighbors[dst] = cost
    return neighbors  # Move the return statement outside the loop
    
def gbfs(start, goals, edges, nodes):
    def heuristic(n):
        return min(
            ((nodes[g][0] - nodes[n][0]) ** 2 + (nodes[g][1] - nodes[n][1]) ** 2) ** 0.5
            for g in goals)  # calculate the distance between the current node and the goal node.
    
    # Initialize the priority queue
    pq = []

    # Push the start node to the priority queue
    heapq.heappush(pq, (heuristic(start), start, [start], 0))

    visited = set()
    while pq:
        _, node, path, cost = heapq.heappop(pq)
        if node in visited:
            continue

        visited.add(node)
        if node in goals:
            return node, len(visited), path
        
        for neighbour, edge_cost in get_neighbors(node, edges).items():
            heapq.heappush(pq, (heuristic(neighbour), neighbour, path + [neighbour], cost + edge_cost))
    
    # Return None if no solution is found
    return None, len(visited), None

def main():
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        return
    
    filename, method = sys.argv[1], sys.argv[2]
    nodes, edges, origin, destinations = parse_input(filename)
    
    result = gbfs(origin, destinations, edges, nodes)
    if result:
        goal, num_nodes, path = result
        if path is not None:
            print(f"{filename} {method}\n{goal} {num_nodes}\n{' -> '.join(map(str, path))}")
        else:
            print(f"{filename} {method}\n{goal} {num_nodes}\nNo path found.")
    else:
        print(f"{filename} {method}\nNo solution found.")
        
if __name__ == "__main__":
    main()
