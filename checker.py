########################
##### DATA READING #####
########################

import pandas as pd
import random
import numpy as np
import networkx as nx
### CONSTANTS ###

#max time for the line (hours)
max_time = 6
#minimum amount of cities in a line
min_cities = 3
#train speed (km/h)
speed = 300
#minutes spent at each stop (minutes converted to hours)
stop_time = 5/60.0
#maximum amount of lines
max_lines = 20

# Distances in between cities
adj_matrix = pd.read_excel("city_distance_matrix.xlsx", index_col=0)
# Demand between cities
dem_matrix = pd.read_excel("city_demand_matrix.xlsx", index_col = 0)

edges = []

## Create a list of all rail tracks

for i, c1 in enumerate(adj_matrix.index):
    for j, c2 in enumerate(adj_matrix.columns):
        if j <= i:
            continue
        dist = adj_matrix.iloc[i, j]
        if dist > 0:
            edges.append((c1, c2, dist))

######################
##### EVALUATION #####
######################

def evaluate_lines(lines):
    
    G = nx.Graph()
    for line in lines:
        for i in range(len(line)-1):
            c1, c2 = line[i], line[i+1]
            dist = adj_matrix.loc[c1, c2]
            if dist > 0:
                G.add_edge(c1, c2, weight=dist)
    
    total_weighted_time = 0.0
    total_penalty = 0.0
    covered_cities = set()

    all_cities = set(adj_matrix.index)
    
    for c1 in all_cities:
        for c2 in all_cities:
            if c1 == c2:
                continue
            demand = demand_matrix.loc[c1, c2]
            
            try:
                # Compute shortest path distance using the graph
                shortest_dist = nx.shortest_path_length(G, source=c1, target=c2, weight='weight')
                travel_time = shortest_dist / speed
                total_weighted_time += travel_time * demand

            except nx.NetworkXNoPath:
                # No path exists between c1 and c2 -> heavy penalty
                total_penalty += 24.0  # 24-hour penalty per disconnected pair
    
    for line in lines:
        line_time = 0.0
        for i in range(len(line)-1):
            dist = adj_matrix.loc[line[i], line[i+1]]
            travel_time = dist / speed
            if i > 0:
                travel_time += stop_time
            line_time += travel_time
        if line_time > max_time:
            total_penalty += 24.0  # 24-hour penalty
            
    # 5️⃣ Penalize duplicate lines
    unique_lines = {tuple(line) for line in lines}
    duplicates = len(lines) - len(unique_lines)
    total_penalty += duplicates * 24.0

    # 6️⃣ Fitness = weighted travel time minus total penalties
    fitness = total_weighted_time - total_penalty
    return fitness

##############################
##### GENETIC ALGORITHM ######
##############################



### GENERATION ALGORITHM ###
def generate_random_line():

    cities = list(adj_matrix.index)
    start_city = random.choice(cities)
    line = [start_city]
    total_time = 0.0
    
    while True:
        current_city = line[-1]
        # Find neighbors with valid tracks
        neighbors = [
            (city, adj_matrix.loc[current_city, city])
            for city in cities if city not in line
        ]
        neighbors = [(city, dist) for city, dist in neighbors if dist > 0]
        
        if not neighbors:
            break  # dead end
        
        # Shuffle neighbors for randomness
        random.shuffle(neighbors)
        extended = False
        
        for city, dist in neighbors:
            travel_time = dist / speed
            stops = (len(line) - 1) * (stop_time)  # dwell per stop (excluding start and end)
            new_total_time = total_time + travel_time + stops
            if new_total_time <= max_time:
                line.append(city)
                total_time = new_total_time
                extended = True
                break
        
        if not extended:
            break
        
        # Random chance to stop early
        if len(line) >= min_cities and random.random() < 0.3:
            break

    if(len(line) < min_cities):
        return generate_random_line()
    else: 
        return line
    
def generate_lines():
    lines = []
    for _ in range(20):
        line = generate_random_line()
        lines.append(line)
    return evaluate_lines(lines)


