import pandas as pd
import random
import numpy as np
import networkx as nx
import heapq
from collections import defaultdict
from joblib import Parallel, delayed

# ACO PARAMETERS
num_ants = 50             # number of ants per generation
num_generations = 100      # number of generations
evap_rate = 0.1           # evaporation rate
pher_dep = 1e9            # pheromone deposit multiplier
PHEROMONE_MAX = 5.0
PHEROMONE_MIN = 0.01
### CONSTANTS ###

max_time = 6                #max time for the line (hours)
min_cities = 3              #minimum amount of cities in a line
speed = 300                 #train speed (km/h)
stop_time = 5.0/60.0        #minutes spent at each stop (minutes converted to hours)
transfer_time = 30.0/60.0   #minutes spent at each transfer (minutes converted to hours)
max_lines = 12              #maximum amount of lines
unconnected_penalty = 72   #penalty for two cities being not connected

### SPREADSHEETS ###
adj_matrix = pd.read_excel("city_distance_matrix.xlsx", index_col = 0)  # Distances in between cities
dem_matrix = pd.read_excel("city_demand_matrix.xlsx", index_col = 0)    # Demand between cities

cities = list(adj_matrix.index)
num_cities = len(cities)
c_to_i = {city: idx for idx, city in enumerate(cities)}

edges = []                                                      # Graph of connecting cities
pheromone = np.ones((num_cities, num_cities), dtype=float)      # Pheromone matrix

## Create a list of all rail tracks
for i, c1 in enumerate(adj_matrix.index):
    for j, c2 in enumerate(adj_matrix.columns):
        if j <= i:
            continue
        dist = adj_matrix.loc[c1, c2]
        if dist > 0:
            edges.append((c1, c2, dist))
            



######################
##### EVALUATION #####
######################
def dijkstra(origin, graph):
    dist = {city: [] for city in graph}

    pq = [(0, origin, [origin], None)] # Priority queue with time, current city, the path to get there &  what line it's on
    while pq:
        time, city, path, line_id = heapq.heappop(pq)
        
        if not dist[city] or abs(time - dist[city][0][0]) <= transfer_time - stop_time:
            dist[city].append((time, path, line_id))

            #Add all neighbors
            for neighbor, travel_time, track_line in graph[city]:
                new_time = time + travel_time
                new_path = path + [neighbor]
                
                if line_id != None and line_id != track_line:
                    new_time += transfer_time
                elif line_id == track_line:
                    new_time += stop_time
                
                heapq.heappush(pq, (new_time, neighbor, new_path, track_line))

    return dist
                


def evaluate_lines(lines):
    
    cities = adj_matrix.index
    total = 0
    #Graph of all the routes
    G = {}  #city, travel_time, line_id
    for city in cities:
        G[city] = []
    for line_id, line in enumerate(lines):
        for i in range(len(line)-1):
            c1, c2 = line[i], line[i+1]
            dist = adj_matrix.loc[c1, c2]
            time = dist / speed

            if c1 not in G:
                G[c1] = []
            if c2 not in G:
                G[c2] = []

            G[c1].append((c2, time, line_id)) 
            G[c2].append((c1, time, line_id))
    
    for origin_city in cities:
        best_times = dijkstra(origin_city, G)
        for destination in best_times:
            demand_to_dest = dem_matrix.loc[origin_city, destination]
            possible_routes = best_times[destination]
            if not possible_routes:
                total += unconnected_penalty * demand_to_dest
                continue
            
            min_time = min(possible_routes)[0]
            total += min_time * demand_to_dest

    return total

# Initialize pheromones on all city pairs

cities = list(adj_matrix.index)

def construct_ant():
    lines = []
    for _ in range(max_lines):
        line = []
        start_city = random.choice(cities)
        line.append(start_city)
        total_time = 0

        while True:
            c1 = line[-1]
            # Find neighbors with valid tracks
            neighbors = []
            for next_city in cities:
                if next_city not in line:
                    dist = adj_matrix.loc[c1, next_city]
                    #Check if there is a track between the cities
                    if dist > 0 and total_time + dist/speed <= max_time:
                        neighbors.append((next_city, dist))

            if not neighbors:
                break  # dead end
        
            probs = []
            candidate_cities = []
            for next_city, _ in neighbors:
                pher = pheromone[c_to_i[c1]][c_to_i[next_city]]
                probs.append(pher)
                candidate_cities.append(next_city)
            probs = np.array(probs) / sum(probs)
            c2 = np.random.choice(candidate_cities, p=probs)
            line.append(c2)
            travel_time = adj_matrix.loc[c1, c2] / speed
            total_time += travel_time
            c1 = c2

        if len(line) >= min_cities:
            lines.append(line)
    return lines

########################
##### MAIN ACO ########
########################

def aco():
    best_network = None
    best_fitness = float('inf')
    fitness_history = []
    for iteration in range(num_generations):
        ant_networks = [construct_ant() for _ in range(num_ants)]
        fitnesses = Parallel(n_jobs=-1)(
            delayed(evaluate_lines)(network) for network in ant_networks
        )

        iteration_best_idx = np.argmin(fitnesses)
        iteration_best_net = ant_networks[iteration_best_idx]
        iteration_best_fit = fitnesses[iteration_best_idx]

        if iteration_best_fit < best_fitness:
            best_fitness = iteration_best_fit
            best_network = iteration_best_net
            stagnation = 0
        else:
            stagnation += 1

        print(f"Iteration {iteration+1}, best fitness: {iteration_best_fit}")

        # Evaporation
        pheromone[:] = pheromone * (1 - evap_rate)

        # Reinforce pheromones for best ant
        for line in iteration_best_net:
            for i in range(len(line)-1):
                c1 = line[i]
                c2 = line[i+1]
                deposit = pher_dep / iteration_best_fit
                pheromone[c_to_i[c1]][c_to_i[c2]] += deposit
                pheromone[c_to_i[c2]][c_to_i[c1]] += deposit

        if stagnation < 10:  # Only reinforce if not stagnating
            for line in best_network:
                for i in range(len(line)-1):
                    c1, c2 = line[i], line[i+1]
                    deposit = (2 * pher_dep) / best_fitness
                    pheromone[c_to_i[c1]][c_to_i[c2]] += deposit
                    pheromone[c_to_i[c2]][c_to_i[c1]] += deposit

        pheromone[:] = np.clip(pheromone, PHEROMONE_MIN, PHEROMONE_MAX) #MIN-MAX ACO
        
        if stagnation > 20:
            pheromone[:] = np.ones((num_cities, num_cities))
            stagnation = 0
        fitness_history.append(iteration_best_fit)
    return fitness_history, best_network, best_fitness

def run_aco():
    runs = {}
    for run in range (5):
        print(f"RUN {run+1} ########################")
        fitness_history, best_network, best_fitness = aco()
        runs[f"Run {run}"] = (fitness_history)
    
    df = pd.DataFrame(runs)
    df.index.name = "Generation"

    df.to_excel("aco_results.xlsx", index=True)
    print("Results saved to 'aco_results.xlsx'.")

run_aco()