
import pandas as pd
import random
import numpy as np
import networkx as nx

########################
##### DATA READING #####
########################


### CONSTANTS ###

max_time = 6 #max time for the line (hours)
min_cities = 3 #minimum amount of cities in a line
speed = 300 #train speed (km/h)
stop_time = 5.0/60.0 #minutes spent at each stop (minutes converted to hours)
transfer_time = 15.0/60.0 #minutes spent at each transfer (minutes converted to hours)
max_lines = 20 #maximum amount of lines

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

######################
##### EVALUATION #####
######################

def evaluate_lines(lines):
    #Graph of all the routes
    G = nx.Graph()
    for line_id, line in enumerate(lines):
        for i in range(len(line)-1):
            c1, c2 = line[i], line[i+1]
            dist = adj_matrix.loc[c1, c2]
            G.add_edge(c1, c2, weight=dist, line_id = line_id)

    


POP_SIZE = 30        # number of candidate solutions
N_GENERATIONS = 50   # iterations
MUTATION_RATE = 0.2  
TOURNAMENT_SIZE = 3

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
    return max(competitors, key=lambda x: x[1])[0]


### GENERATION ALGORITHM ###
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
    for _ in range(20):
        line = generate_random_line()
        lines.append(line)
    return evaluate_lines(lines)



##############################
##### GENETIC ALGORITHM ######
##############################

def ga():
    # 1. Initialize
    population = [generate_candidate() for _ in range(POP_SIZE)]
    
    for gen in range(N_GENERATIONS):
        # 2. Evaluate
        fitnesses = [evaluate_lines(cand) for cand in population]
        
        # Track progress
        best_idx = np.argmax(fitnesses)
        print(f"Gen {gen}: best fitness = {fitnesses[best_idx]}")
        
        # 3. Next generation
        new_population = []
        for _ in range(POP_SIZE):
            # Selection
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            # Crossover + Mutation
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        
        population = new_population
    
    # Final best solution
    fitnesses = [evaluate_lines(cand) for cand in population]
    best_idx = np.argmax(fitnesses)
    return population[best_idx], fitnesses[best_idx]

best_plan, best_score = ga()
print("Best solution found:", best_plan)
print("Fitness:", best_score)
