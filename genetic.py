
import pandas as pd
import random
import numpy as np
import networkx as nx
import heapq
from collections import defaultdict
from joblib import Parallel, delayed

########################
##### DATA READING #####
########################


### CONSTANTS ###

max_time = 6 #max time for the line (hours)
min_cities = 3 #minimum amount of cities in a line
speed = 300 #train speed (km/h)
stop_time = 5.0/60.0 #minutes spent at each stop (minutes converted to hours)
transfer_time = 30.0/60.0 #minutes spent at each transfer (minutes converted to hours)
max_lines = 12 #maximum amount of lines
unconnected_penalty = 72 #penalty for two cities being not connected

### GENETIC PARAMETERS ###

POP_SIZE = 100        # number of candidate solutions
N_GENERATIONS = 200   # iterations
MUTATION_RATE = 0.3  
TOURNAMENT_SIZE = 4
ELITE_POPULATION = 3  # number of elite candidates to carry over unchanged

### SPREADSHEETS ###
adj_matrix = pd.read_excel("city_distance_matrix.xlsx", index_col = 0) # Distances in between cities
dem_matrix = pd.read_excel("city_demand_matrix.xlsx", index_col = 0) # Demand between cities


edges = [] # Graph of connecting cities

## Create a list of all rail tracks

for i, c1 in enumerate(adj_matrix.index):
    for j, c2 in enumerate(adj_matrix.columns):
        if j <= i:
            continue
        dist = adj_matrix.iloc[i, j]
        if dist > 0:
            edges.append((c1, c2, dist))

############################
### GENERATION ALGORITHM ###
############################

def generate_random_line():

    cities = list(adj_matrix.index)
    start_city = random.choice(cities)
    line = [start_city]
    total_time = 0.0
    
    while True:
        current_city = line[-1]

        # Find neighbors with valid tracks
        neighbors = []
        for city in cities:
            if city not in line:
                dist = adj_matrix.loc[current_city, city]
                #Check if there is a track between the cities
                if dist > 0:
                    neighbors.append((city, dist))
        
        if not neighbors:
            break  # dead end
        
        # Shuffle neighbors for randomness
        random.shuffle(neighbors)
        extended = False
        
        #Check if the city can be added to the route (Doesn't exceed the time limit)
        for city, dist in neighbors:
            travel_time = dist / speed
            stops = (len(line) - 1) * (stop_time) 
            new_total_time = total_time + travel_time + stops
            if new_total_time <= max_time:
                line.append(city)
                total_time = new_total_time
                extended = True
                break
        
        if not extended:
            break # line has reached maximum size
        
        # Random chance to stop early
        if len(line) >= min_cities and random.random() < 0.3:
            break

    #Make sure line is not a connection of only 2 cities
    if(len(line) < min_cities):
        return generate_random_line()
    else: 
        return line
    
def generate_lines():
    lines = []
    for _ in range(max_lines):
        line = generate_random_line()
        lines.append(line)
    return lines

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


##############################
##### GENETIC ALGORITHM ######
##############################

def generate_candidate():
    return [generate_random_line() for _ in range(max_lines)]

def crossover(parent1, parent2):
    cut = random.randint(1, max_lines-1)
    child = parent1[:cut] + parent2[cut:]
    return child

def mutate(candidate):
    if random.random() < MUTATION_RATE:
        idx = random.randint(0, len(candidate)-1)
        candidate[idx] = generate_random_line()
    return candidate

def tournament_selection(population, fitnesses):
    competitors = random.sample(list(zip(population, fitnesses)), TOURNAMENT_SIZE)
    return min(competitors, key=lambda x: x[1])[0]



def ga():
    # Create base population
    population = [generate_lines() for _ in range(POP_SIZE)]
    best_solution = None
    best_fitness = float('inf')

    
    for gen in range(N_GENERATIONS):
        # Evaluation
        fitnesses = Parallel(n_jobs=-1)(
            delayed(evaluate_lines)(cand) for cand in population
        )
        fitness_history = []
        # Tracking
        current_best_idx = np.argmin(fitnesses)
        if fitnesses[current_best_idx] < best_fitness:
            best_fitness = fitnesses[current_best_idx]
            best_solution = population[current_best_idx]
        print(f"Gen {gen}: best fitness = {fitnesses[current_best_idx]}")

        # Elitism
        elites = np.argsort(fitnesses)[:ELITE_POPULATION]
        # New population generation with elites included
        new_population = [population[i] for i in elites]
        for _ in range(POP_SIZE - ELITE_POPULATION):
            # Selection
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            # Crossover + Mutation
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        
        population = new_population
        fitness_history.append(fitnesses[current_best_idx])
    
    # Final best solution
    fitnesses = Parallel(n_jobs=-1)(
        delayed(evaluate_lines)(cand) for cand in population
    )   
    best_idx = np.argmin(fitnesses)
    print(f"BEST FITNESS TOTAL: {fitnesses[best_idx]}")
    for line in population[best_idx]:
        print(line)
        print("\n")
    return fitness_history, population[best_idx], fitnesses[best_idx]


def run_ga():
    runs = {}
    networks = []
    for run in range (5):
        print(f"RUN {run+1} ########################")
        fitness_history, best_network, best_fitness = ga()
        runs[f"Run {run}"] = (fitness_history)
        networks.append(best_network)
    
    df = pd.DataFrame(runs)
    df.index.name = "Generation"

    df.to_excel("genetic_algorithm_results.xlsx", index=True)
    print("Results saved to 'genetic_algorithm_results.xlsx'.")

run_ga()
