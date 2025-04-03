
import re
from collections import defaultdict

def parse_input_file(filename):
 #initialization
    nodes = {}  #store node id 1, 2, 3 etc
    edges = defaultdict(list)  #stores edges
    origin = None              #starting node    
    destinations = set()        #target nodes
    
    current_section = None

    mapping = { #case sensitive mapping 
        "nodes:": "nodes",
        "edges:": "edges",
        "origin:": "origin",
        "destinations:": "destinations"
    }
    
    with open(filename, 'r') as file: #file process loop
        for line in file:
            line = line.strip() #skipping line breaks
            if not line:
                continue
            
            if line.lower() in mapping:
                current_section = mapping[line.lower()]
                continue
            
            if current_section == "nodes":
                match = re.match(r'(\d+):\s*\((\d+),\s*(\d+)\)', line) #format matching PathFinder file
                if match:
                    node_num = int(match.group(1)) #nodes above are regexed so node_num, x, y can extract the integers
                    x = int(match.group(2))
                    y = int(match.group(3))
                    nodes[node_num] = (x, y)
            
            elif current_section == "edges":
                match = re.match(r'\((\d+),\s*(\d+)\):\s*(\d+)', line) #format matching PathFinder file
                if match:
                    from_node = int(match.group(1))
                    to_node = int(match.group(2))
                    cost = int(match.group(3))
                    edges[from_node].append((to_node, cost))
            
            elif current_section == "origin":
                origin = int(line.strip())
            
            elif current_section == "destinations":
                destination_list = [d.strip() for d in line.split(';')] #splits the list of qued destinations to be split after each ;
                destinations = set(int(d) for d in destination_list if d.isdigit()) #turns the split integers to a set and makes sure there is no string
    
    return nodes, edges, origin, destinations
