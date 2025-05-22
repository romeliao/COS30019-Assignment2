# Step class to track parent nodes for path reconstruction
class Step:
    def __init__(self, id, parent=None):
        self.id = id
        self.parent = parent

# Depth-First Search algorithm
def dfs(start, goal_list, edges):
    stack = [Step(start)]
    visited = set()
    expanded = 0

    while stack:
        current = stack.pop()
        expanded += 1

        if current.id in visited:
            continue

        visited.add(current.id)

        if current.id in goal_list:
            return current, expanded

        # Add neighbors in reverse order for consistent DFS behavior
        for neighbor, _ in reversed(edges.get(current.id, [])):
            if neighbor not in visited:
                stack.append(Step(neighbor, current))

    return None, expanded

# Reconstruct path from end node to start node
def trace_path(end_step):
    path = []
    while end_step:
        path.append(end_step.id)
        end_step = end_step.parent
    return path[::-1]

# Convert a NetworkX graph to an adjacency list with edge weights
def nx_to_edge(graph):
    """
    Converts a NetworkX graph into a dictionary of adjacency lists with edge weights.

    Args:
        graph (networkx.Graph): The NetworkX graph.

    Returns:
        dict: Edge dictionary {node: [(neighbor, weight), ...]}
    """
    edges = {}
    for u, v, data in graph.edges(data=True):
        cost = data["weight"]
        edges.setdefault(u, []).append((v, cost))
        edges.setdefault(v, []).append((u, cost))  # Assume undirected graph
    return edges

# Extract coordinates from SCATS metadata
def get_coords(scats_data):
    """
    Extracts coordinates from SCATS metadata.

    Args:
        scats_data (dict): Metadata for SCATS nodes with Latitude and Longitude.

    Returns:
        dict: Dictionary mapping SCATS ID to (lat, lon) tuple.
    """
    return {
        scats: (info["Latitude"], info["Longitude"])
        for scats, info in scats_data.items()
    }