def dls(start, goal_list, edges, depth_limit):

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