import sys
from gbfs import gbfs  # Import gbfs and get_neighbors from gbfs.py

def parse_input(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')

    nodes, edges, origin, destination = {}, {}, None, []
    section = None

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

        elif section == "nodes":
            if not line or ": " not in line:
                print(f"Skipping invalid node line (missing ': '): {line}")
                continue
            parts = line.split(": ")
            try:
                node_id = int(parts[0])
                coordinates = tuple(map(int, parts[1][1:-1].split(",")))
                nodes[node_id] = coordinates
            except ValueError:
                print(f"Skipping invalid node line (invalid node ID or coordinates): {line}")
            except IndexError:
                print(f"Skipping invalid node line (missing coordinates): {line}")

        elif section == "edges":
            parts = line.split(": ")
            if len(parts) != 2 or not parts[0].startswith("(") or not parts[0].endswith(")"):
                print(f"Skipping invalid edge line (missing or malformed edge format): {line}")
                continue
            try:
                edge = tuple(map(int, parts[0][1:-1].split(',')))
                weight = int(parts[1])
                edges[edge] = weight
                edges[(edge[1], edge[0])] = weight
            except ValueError:
                print(f"Skipping invalid edge line (invalid node IDs or weight): {line}")

        elif section == "origin":
            origin = int(line)

        elif section == "destination":
            destination = list(map(int, line.replace(" ", "").split(';')))

    return nodes, edges, origin, destination

def main():
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        return
    
    filename, method = sys.argv[1], sys.argv[2]
    nodes, edges, origin, destinations = parse_input(filename)
    
    result = gbfs(origin, destinations, edges, nodes)
    # Inside the main function
    if result:
        goal, num_nodes, path = result
        print(f"File: {filename}, Method: {method.upper()}")
        print(f"Goal Node: {goal}, Nodes Explored: {num_nodes}")
        print(f"Path: {' -> '.join(map(str, path)) if path else 'No path found.'}")
    else:
        print(f"File: {filename}, Method: {method.upper()}")
        print("No solution found.")

        
if __name__ == "__main__":
    main()