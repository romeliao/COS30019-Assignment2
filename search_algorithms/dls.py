def dls(start, goal_list, edges, depth_limit):
    def recursive_dls(node, depth, visited, path):
        nonlocal expanded
        expanded += 1

        path.append(node)  # Add current node to path

        if node in goal_list:  # Goal check
            return node

        if depth == 0:  # Depth limit reached
            path.pop()  # Backtrack
            return None

        visited.add(node)  # Mark as visited to avoid cycles

        for neighbor, _ in edges.get(node, []):
            if neighbor not in visited:
                result = recursive_dls(neighbor, depth - 1, visited, path)
                if result is not None:
                    return result

        path.pop()  # Backtrack
        return None

    expanded = 0
    visited = set()
    path = []

    result = recursive_dls(start, depth_limit, visited, path)

    if result is not None:
        return result, expanded, path
    else:
        return None, expanded, None


def nx_to_edge(graph):
    edges = {}
    for u, v, data in graph.edges(data=True):
        cost = data["weight"]
        edges.setdefault(u, []).append((v, cost))
        edges.setdefault(v, []).append((u, cost))
    return edges


def get_coords(scats_data):
    return {scats: (info["Latitude"], info["Longitude"]) for scats, info in scats_data.items()}
