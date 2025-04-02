import sys
from gbfs import gbfs
from bfs import bfs, trace_path as trace_bfs_path
from dfs import dfs, trace_path as trace_dfs_path

def parse_input(file_path):
    nodes = {}
    edges = {}
    origin = None
    destination = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    section = None
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace
        if not line:  # Skip empty lines
            continue

        if line.startswith("Nodes"):
            section = "nodes"
        elif line.startswith("Edges"):
            section = "edges"
        elif line.startswith("Origin"):
            section = "origin"
        elif line.startswith("Destinations"):
            section = "destination"
        elif section == "nodes":
            try:
                node_id, coords = line.split(": ")
                x, y = map(int, coords.strip("()").split(","))
                nodes[int(node_id)] = (x, y)
            except ValueError:
                print(f"Skipping invalid node line: {line}")
        elif section == "edges":
            try:
                edge, cost = line.split(": ")
                src, dst = map(int, edge.strip("()").split(","))
                edges[(src, dst)] = int(cost)
            except ValueError:
                print(f"Skipping invalid edge line: {line}")
        elif section == "origin":
            try:
                origin = int(line)
            except ValueError:
                print(f"Skipping invalid origin line: {line}")
        elif section == "destination":
            try:
                destination = list(map(int, line.replace(" ", "").split(";")))
            except ValueError:
                print(f"Skipping invalid destination line: {line}")

    return nodes, edges, origin, destination

def main():
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        return
    
    filename, method = sys.argv[1], sys.argv[2].upper()
    nodes, edges, origin, destinations = parse_input(filename)
    
    goal, expanded, path = None, 0, None  # Initialize variables with default values

    if method == "GBFS":
        result = gbfs(origin, destinations, edges, nodes)
        if result:
            goal, expanded, path = result
    elif method == "BFS":
        result = bfs(origin, destinations, edges)
        if result:
            goal, expanded = result
            path = trace_bfs_path(goal)
    elif method == "DFS":
        result = dfs(origin, destinations, edges)
        if result:
            goal, expanded = result
            path = trace_dfs_path(goal)
    else:
        print(f"Unsupported method: {method}")
        return

    print(f"File: {filename}, Method: {method}")
    if goal:
        print(f"Goal Node: {goal}, Nodes Explored: {expanded}")
        print(f"Path: {' -> '.join(map(str, path)) if path else 'No path found.'}")
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()