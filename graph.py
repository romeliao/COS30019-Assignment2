class Graph:
    def __init__(self):
        self.nodes = {}  
        self.edges = {}  
        self.origin = None
        self.destinations = set()

    def add_node(self, node_id, coords):
        self.nodes[node_id] = coords
        self.edges[node_id] = []

    def add_edge(self, start, end, cost):
        if start not in self.edges:
            self.edges[start] = []
        self.edges[start].append((end, cost))

    def search(self, depth_limit):
        if depth_limit <= 0:
            print("Invalid depth limit. Must be a positive integer.")
            return

        visited = set()
        path = []
        found_destinations = []

        def dfs(node, depth):
            if depth > depth_limit:
                return
            visited.add(node)
            path.append(node)

            if node in self.destinations and node not in found_destinations:
                found_destinations.append(node)
                print(f"Destination {node} reached with path: {path}")

            for neighbor, _ in self.edges.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, depth + 1)

            path.pop()

        dfs(self.origin, 0)
        if found_destinations:
            print(f"Found destinations: {found_destinations}")
        else:
            print("Unfortunately, there are no destinations within the depth limit.")
        print(f"Visited nodes: {sorted(visited)}")