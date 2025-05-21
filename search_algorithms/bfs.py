from collections import deque

class Step:
    def __init__(self, id, parent=None, total_cost=0):
        self.id = id
        self.parent = parent
        self.total_cost = total_cost

def bfs(start, goal_list, edges):
    queue = deque()
    visited = set()
    tracked = set()

    queue.append(Step(start))
    tracked.add(start)

    expanded = 0

    while queue:
        current = queue.popleft()
        expanded += 1
        visited.add(current.id)

        # Check if the current node is a goal
        if current.id in goal_list:
            return current, expanded

        # Iterate over neighbors of the current node
        for neighbor, edge_cost in edges.get(current.id, []):  # Get neighbors from edges
            if neighbor not in visited and neighbor not in tracked:
                cost = current.total_cost + edge_cost
                queue.append(Step(neighbor, current, cost))
                tracked.add(neighbor)

    # Return None if no solution is found
    return None, expanded

def trace_path(end_step):
    path = []
    while end_step:
        path.append(end_step.id)
        end_step = end_step.parent
    return path[::-1]