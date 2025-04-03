
from AStar import a_star_search
from HillClimbing import hill_climbing_search
from Parsing import parse_input_file

def main():
    import sys
    
    filename = sys.argv[1]
    search_method = sys.argv[2].lower() #stripping functions so it is simpler to process
    
    nodes, edges, origin, destinations = parse_input_file(filename)
    
    if search_method == "a_star": #py main.py PathFinder-test.txt a_star
        path, nodes_created = a_star_search(origin, destinations, edges, nodes)
    elif search_method == "hill_climb": #py main.py PathFinder-test.txt hill_climb
        path, nodes_created = hill_climbing_search(origin, destinations, edges, nodes)
    else:
        print("Unknown method")
        return
    
    if path:
        print(f"{path[-1]} {nodes_created}")
        print(" ".join(map(str, path)))
    else:
        print("Unknown path")

if __name__ == "__main__":
    main()