class Step:
    def __init__(self, id, parent=None):
        self.id = id
        self.parent = parent

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

        # Check if the current node is a goal
        if current.id in goal_list:
            return current, expanded

        # Iterate over neighbors of the current node in reverse order
        for neighbor, _ in reversed(edges.get(current.id, [])):  # Get neighbors from edges
            if neighbor not in visited:
                stack.append(Step(neighbor, current))

    # Return None if no solution is found
    return None, expanded

def trace_path(end_step):
    path = []
    while end_step:
        path.append(end_step.id)
        end_step = end_step.parent
    return path[::-1]