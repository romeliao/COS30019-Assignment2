def dls(start, goal_list, edges, depth_limit):
    """
    Perform Depth-Limited Search (DLS) to find a path to one of the goal nodes.

    Args:
        start (int): The starting node.
        goal_list (list): A list of goal nodes.
        edges (dict): A dictionary where keys are nodes and values are lists of tuples (neighbor, cost).
        depth_limit (int): The maximum depth to explore.

    Returns:
        tuple: A tuple containing the goal Step object, the number of nodes expanded, and the path.
               Returns (None, expanded, None) if no solution is found within the depth limit.
    """
    def recursive_dls(node, depth, visited, path):
        nonlocal expanded
        expanded += 1

        # Add the current node to the path
        path.append(node)

        # Check if the current node is a goal
        if node in goal_list:
            return node

        # Stop if the depth limit is reached
        if depth == 0:
            path.pop()  # Backtrack
            return None

        # Mark the node as visited
        visited.add(node)

        # Explore neighbors
        for neighbor, _ in edges.get(node, []):
            if neighbor not in visited:
                result = recursive_dls(neighbor, depth - 1, visited, path)
                if result:
                    return result

        # Backtrack
        path.pop()
        return None

    # Initialize variables
    expanded = 0
    visited = set()
    path = []

    # Start the recursive DLS
    result = recursive_dls(start, depth_limit, visited, path)

    # Return the result
    if result:
        return result, expanded, path
    else:
        return None, expanded, None