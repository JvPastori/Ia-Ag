import math
import json
from random import random, randint


class City:
    def __init__(self, name, x, y):
        self.name = name
        self.x = float(x)
        self.y = float(y)
        self.distances = []

    def calc_distance(self, other):
        return round(math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2))


def read_tsplib(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    cities = []
    node_section = False
    for line in lines:
        if "NODE_COORD_SECTION" in line:
            node_section = True
            continue
        if "EOF" in line or not node_section:
            continue

        parts = line.strip().split()
        if len(parts) >= 3:
            city_id, x, y = parts[0], parts[1], parts[2]
            cities.append(City(city_id, x, y))

    for city in cities:
        city.distances = [city.calc_distance(other) for other in cities]

    return cities


class Distance:
    def __init__(self, cities):
        self.cities = cities

    def get_distance(self, i, j):
        return self.cities[i].distances[j]


class Individuals():
    def __init__(self, time_distances, cities, generation=0):
        self.time_distances = time_distances
        self.cities = cities
        self.generation = generation
        self.note_review = 0
        self.chromosome = []
        self.visited_cities = []
        self.travelled_distance = 0
        self.probability = 0

        indices = list(range(len(cities)))
        while indices:
            self.chromosome.append(indices.pop(randint(0, len(indices) - 1)))

    def fitness(self):
        sum_distance = 0
        current_city = self.chromosome[0]

        for i in range(len(self.chromosome)):
            d = Distance(self.cities)
            dest_city = self.chromosome[i]
            distance = d.get_distance(current_city, dest_city)
            sum_distance += distance
            self.visited_cities.append(dest_city)
            current_city = dest_city

            if i == len(self.chromosome) - 1:
                sum_distance += d.get_distance(self.chromosome[-1], self.chromosome[0])

        self.travelled_distance = sum_distance

    def crossover(self, otherIndividual):
        genes_1 = self.chromosome[:]
        genes_2 = otherIndividual.chromosome[:]
        selected_gene = randint(0, len(genes_1) - 1)
        self.exchange_gene(selected_gene, genes_1, genes_2)
        exchanged_genes = [selected_gene]

        while True:
            duplicated_gene = self.get_duplicated_gene(genes_1, exchanged_genes)
            if duplicated_gene == -1:
                break
            self.exchange_gene(duplicated_gene, genes_1, genes_2)
            exchanged_genes.append(duplicated_gene)

        childs = [
            Individuals(self.time_distances, self.cities, self.generation + 1),
            Individuals(self.time_distances, self.cities, self.generation + 1)
        ]
        childs[0].chromosome = genes_1
        childs[1].chromosome = genes_2

        return childs

    def exchange_gene(self, gene, genes_1, genes_2):
        genes_1[gene], genes_2[gene] = genes_2[gene], genes_1[gene]

    def get_duplicated_gene(self, genes, exchanged_genes):
        for gene in range(len(genes)):
            if gene in exchanged_genes:
                continue
            if genes.count(genes[gene]) > 1:
                return gene
        return -1

    def mutate(self, mutationRate):
        if randint(1, 100) <= mutationRate:
            print("Realizando mutação no cromossomo %s" % self.chromosome)
            genes = self.chromosome
            gene_1 = randint(0, len(genes) - 1)
            gene_2 = randint(0, len(genes) - 1)
            genes[gene_1], genes[gene_2] = genes[gene_2], genes[gene_1]
            print("Valor após mutação: %s" % self.chromosome)
        return self


class GeneticAlgorithm():
    def __init__(self, population_size=20, cities=[]):
        self.populationSize = population_size
        self.population = []
        self.generation = 0
        self.best_solution = None
        self.cities = cities

    def init_population(self, time_distances, cities):
        for _ in range(self.populationSize):
            self.population.append(Individuals(time_distances, cities))
        self.best_solution = self.population[0]

    def sort_population(self):
        self.population.sort(key=lambda ind: ind.travelled_distance)

    def best_individual(self, individual):
        if individual.travelled_distance < self.best_solution.travelled_distance:
            self.best_solution = individual

    def sum_travelled_distance(self):
        return sum(ind.travelled_distance for ind in self.population)

    def select_parents(self, sum_travelled_distances):
        total_coefficient = sum(1 / ind.travelled_distance for ind in self.population)
        for ind in self.population:
            ind.probability = (1 / ind.travelled_distance) / total_coefficient

        sorted_value = random()
        i = 0
        accumulated = 0
        while i < len(self.population):
            accumulated += self.population[i].probability
            if accumulated >= sorted_value:
                return i
            i += 1
        return i - 1

    def view_generation(self):
        best = self.population[0]
        print(f"G: {best.generation} -> Valor: {best.travelled_distance} Cromossomo: {best.chromosome}")

    def resolve(self, mutationRate, generations, time_distances, cities):
        self.init_population(time_distances, cities)
        for ind in self.population:
            ind.fitness()

        self.sort_population()
        self.view_generation()

        for _ in range(generations):
            sum_dist = self.sum_travelled_distance()
            new_population = []

            for _ in range(0, self.populationSize, 2):
                p1 = self.select_parents(sum_dist)
                p2 = self.select_parents(sum_dist)
                childs = self.population[p1].crossover(self.population[p2])
                new_population.extend([childs[0].mutate(mutationRate), childs[1].mutate(mutationRate)])

            self.population = new_population

            for ind in self.population:
                ind.fitness()

            self.sort_population()
            self.view_generation()
            self.best_individual(self.population[0])

        print("\nMelhor solução -> G: %s - Distância percorrida: %s - Cromossomo: %s" % (
            self.best_solution.generation,
            self.best_solution.travelled_distance,
            self.best_solution.visited_cities
        ))

        return [
            self.best_solution.generation,
            self.best_solution.travelled_distance,
            self.best_solution.visited_cities
        ]


if __name__ == "__main__":
    filepath = "burma14.tsp"  # Caminho para o arquivo TSPLIB
    cities = read_tsplib(filepath)
    time_distances = [city.distances for city in cities]

    ga = GeneticAlgorithm(population_size=20, cities=cities)
    result = ga.resolve(mutationRate=1, generations=500, time_distances=time_distances, cities=cities)

    print("Resultado final:")
    print({
        'generation': result[0],
        'travelled_distance': result[1],
        'chromosome': result[2],
        'cities': [cities[i].name for i in result[2]]
    })
