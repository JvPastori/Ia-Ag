import tsplib95
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from time import time

def geo_to_radians(coord):
    deg = int(coord)
    min = coord - deg
    return math.pi * (deg + 5.0 * min / 3.0) / 180.0

def geo_distance(coord1, coord2):
    RRR = 6378.388
    lat1 = geo_to_radians(coord1[0])
    lon1 = geo_to_radians(coord1[1])
    lat2 = geo_to_radians(coord2[0])
    lon2 = geo_to_radians(coord2[1])
    q1 = math.cos(lon1 - lon2)
    q2 = math.cos(lat1 - lat2)
    q3 = math.cos(lat1 + lat2)
    return int(RRR * math.acos(0.5 * ((1 + q1) * q2 - (1 - q1) * q3)) + 1)

def load_problem(file_path):
    problem = tsplib95.load(file_path)
    nodes = list(problem.get_nodes())
    n = len(nodes)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if problem.edge_weight_type == 'EUC_2D':
                dx = problem.node_coords[i + 1][0] - problem.node_coords[j + 1][0]
                dy = problem.node_coords[i + 1][1] - problem.node_coords[j + 1][1]
                distance_matrix[i][j] = math.hypot(dx, dy)
            elif problem.edge_weight_type == 'GEO':
                distance_matrix[i][j] = geo_distance(
                    problem.node_coords[i + 1], problem.node_coords[j + 1])
            elif problem.edge_weight_type == 'ATT':
                dx = problem.node_coords[i + 1][0] - problem.node_coords[j + 1][0]
                dy = problem.node_coords[i + 1][1] - problem.node_coords[j + 1][1]
                rij = math.sqrt((dx * dx + dy * dy) / 10.0)
                tij = round(rij)
                distance_matrix[i][j] = tij if tij >= rij else tij + 1
            else:
                distance_matrix[i][j] = problem.get_weight(i + 1, j + 1)
    return nodes, distance_matrix

def initial_population(size, num_cities):
    return [random.sample(range(num_cities), num_cities) for _ in range(size)]

def fitness(route, distance_matrix):
    return sum(distance_matrix[route[i]][route[(i + 1) % len(route)]] for i in range(len(route)))

def tournament_selection(population, fitnesses, k=3):
    selected = random.sample(list(zip(population, fitnesses)), k)
    return min(selected, key=lambda x: x[1])[0]

def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [-1]*len(parent1)
    child[start:end+1] = parent1[start:end+1]
    p2 = [gene for gene in parent2 if gene not in child]
    j = 0
    for i in range(len(child)):
        if child[i] == -1:
            child[i] = p2[j]
            j += 1
    return child

def mutate(route, rate=0.02):
    for i in range(len(route)):
        if random.random() < rate:
            j = random.randint(0, len(route) - 1)
            route[i], route[j] = route[j], route[i]
    return route

def genetic_algorithm(distance_matrix, generations=1000, population_size=100):
    num_cities = len(distance_matrix)
    population = initial_population(population_size, num_cities)
    best_distance = float('inf')
    best_generation = 0
    convergence = []

    for gen in range(generations):
        fitnesses = [fitness(ind, distance_matrix) for ind in population]
        gen_best = min(fitnesses)
        convergence.append(gen_best)

        if gen_best < best_distance:
            best_distance = gen_best
            best_generation = gen

        new_population = []
        for _ in range(population_size):
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            child = mutate(crossover(p1, p2))
            new_population.append(child)

        population = new_population

    return best_distance, best_generation, convergence

def run_instance(file_name):
    print(f'\n🗂️  Executando para: {file_name}')
    nodes, dist_matrix = load_problem(file_name)
    start = time()
    best_dist, gen, convergence = genetic_algorithm(dist_matrix, generations=1000, population_size=150)
    tempo = time() - start
    print(f'✅ Melhor distância: {int(best_dist)} encontrada na geração {gen}')
    print(f'⏱️ Tempo: {tempo:.2f} segundos')
    return file_name, int(best_dist), gen, round(tempo, 2), convergence

if __name__ == "__main__":
    files = ['burma14.tsp', 'att48.tsp', 'gr202.tsp']
    known_optima = {
        'burma14.tsp': 3323,
        'att48.tsp': 10628,
        'gr202.tsp': 40160
    }

    results = []

    for file in files:
        result = run_instance(file)
        results.append(result)

    for file_name, _, _, _, convergence in results:
        plt.plot(convergence, label=file_name)

    plt.title("Convergência do Algoritmo Genético")
    plt.xlabel("Geração")
    plt.ylabel("Melhor Distância")
    plt.legend()
    plt.grid(True)
    plt.savefig("grafico_convergencia.png")
    plt.clf()

    nomes = [r[0] for r in results]
    tempos = [r[3] for r in results]

    plt.bar(nomes, tempos, color='teal')
    plt.title("Tempo de Execução por Instância")
    plt.ylabel("Tempo (s)")
    plt.savefig("grafico_tempos.png")
    plt.clf()
