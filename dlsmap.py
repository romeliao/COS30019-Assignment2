def load_graph_from_file(filename):
    from graph import Graph

    graph = Graph()

    try:
        with open(filename, 'r') as file:
            section = None
            for line_no, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                # Detect section headers
                if line.startswith("Nodes:"):
                    section = "nodes"
                    continue
                elif line.startswith("Edges:"):
                    section = "edges"
                    continue
                elif line.startswith("Origin:"):
                    section = "origin"
                    continue
                elif line.startswith("Destinations:"):
                    section = "destinations"
                    continue

                # Parse data under each section
                try:
                    if section == "nodes":
                        node_id, coords = line.split(":")
                        node_id = int(node_id.strip())
                        if coords.count("(") != coords.count(")") or coords.count("(") != 1:
                            raise ValueError(f"Unbalanced parentheses in coordinates: {coords}")
                        x, y = map(int, coords.strip(" ()").split(","))
                        graph.add_node(node_id, (x, y))
                        print(f"Added node {node_id} with coordinates ({x}, {y})")

                    elif section == "edges":
                        edge, cost = line.split(":")
                        if edge.count("(") != edge.count(")") or edge.count("(") != 1:
                            raise ValueError(f"Unbalanced parentheses in edge: {edge}")
                        start, end = map(int, edge.strip(" ()").split(","))
                        cost = int(cost.strip())
                        graph.add_edge(start, end, cost)
                        print(f"Added edge from {start} to {end} with cost {cost}")

                    elif section == "origin":
                        graph.origin = int(line.strip())
                        print(f"Set origin to {graph.origin}")

                    elif section == "destinations":
                        destinations = set(map(int, line.split(";")))
                        graph.destinations.update(destinations)
                        print(f"Set destinations to {graph.destinations}")

                except ValueError as ve:
                    print(f"ValueError in section '{section}', line {line_no}: {line} — {ve}")
                except Exception as e:
                    print(f"Error in section '{section}', line {line_no}: {line} — {e}")

            # Check for missing sections
            if not graph.nodes:
                print("Error: No nodes defined in the input file.")
            if not graph.edges:
                print("Warning: No edges defined in the input file.")
            if graph.origin is None:
                print("Error: Origin is not defined in the input file.")
            if not graph.destinations:
                print("Warning: No destinations defined in the input file.")

    except FileNotFoundError:
        print(f"The file '{filename}' was not found.")
    except Exception as e:
        print(f"An error occurred while loading the graph: {e}")

    return graph

if __name__ == "__main__":
    filename = "input_graph.txt"
    depth_limit = 3  # Set a valid depth limit

    graph = load_graph_from_file(filename)
    if graph.origin is not None and graph.destinations:
        print("Starting depth-limited search...")
        graph.search(depth_limit)
    else:
        print("Graph is missing an origin or destinations.")