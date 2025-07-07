import numpy as np
import matplotlib.pyplot as plt
POP_SIZE = 100
MAX_GENERATIONS = 100
GENE_RANGE = (-5.12, 5.12)
NUM_DIMENSIONS = 2
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.05
ELITE_COUNT = 1

def de_jongs_f1(x):
    return -np.sum(x**2)

class GeneticAlgorithm:
    def __init__(self, fitness_function, pop_size, crossover_rate, mutation_rate):

        self.fitness_function = fitness_function
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        self.history = []
        self.generations = 0
        self.population = self.init_population()
    def init_population(self):
        return np.random.uniform(GENE_RANGE[0], GENE_RANGE[1], (self.pop_size, NUM_DIMENSIONS))
    def calc_fitnesses(self):
        return np.array([self.fitness_function(genes) for genes in self.population])
    def select_parents(self, fitnesses):
        parents = []
        for _ in range(2):
            tournament_indices = np.random.choice(len(self.population), size=3, replace=False)
            tournament_fitnesses = fitnesses[tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
            parents.append(self.population[winner_index])
        return np.array(parents)
    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            alpha = np.random.rand() # Blending factor
            offspring1 = alpha * parent1 + (1 - alpha) * parent2
            offspring2 = alpha * parent2 + (1 - alpha) * parent1
            offspring1 = np.clip(offspring1, GENE_RANGE[0], GENE_RANGE[1])
            offspring2 = np.clip(offspring2, GENE_RANGE[0], GENE_RANGE[1])
            return offspring1, offspring1
        else:
            return parent1.copy(), parent2.copy()
    def mutate(self, individual):
        mutated_individual = individual.copy()
        for i in range(len(mutated_individual)):
            if np.random.rand() < self.mutation_rate:
                mutation_strength = (GENE_RANGE[1] - GENE_RANGE[0]) * 0.05
                mutated_individual[i] += np.random.normal(0, mutation_strength)
                mutated_individual[i] = np.clip(mutated_individual[i], GENE_RANGE[0], GENE_RANGE[1])
        return mutated_individual
    def train(self):
        while self.generations < MAX_GENERATIONS:
            self.generations += 1
            fitnesses = self.calc_fitnesses()
            #Elitism
            best_individual_idx = np.argmax(fitnesses)
            best_individual_this_gen = self.population[best_individual_idx].copy()
            best_fitness_this_gen = fitnesses[best_individual_idx]
            self.history.append(best_fitness_this_gen)

            if best_fitness_this_gen > -0.000001:
                break
            new_population = [best_individual_this_gen]

            while len(new_population) < self.pop_size:
                parents = self.select_parents(fitnesses)
                offspring1, offspring2 = self.crossover(parents[0], parents[1])
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)
                new_population.append(offspring1)
                if len(new_population) < self.pop_size:
                    new_population.append(offspring2)
            self.population = np.array(new_population)

        final_fitnesses = self.calc_fitnesses()
        overall_best_idx = np.argmax(final_fitnesses)
        overall_best_solution = self.population[overall_best_idx]
        overall_best_objective = -final_fitnesses[overall_best_idx]

        print(f"\n--- GA Finished in {self.generations} ---")
        print(f"Best solution found: x = {overall_best_solution}")
        print(f"Minimum F1 value found: {overall_best_objective:.6f}")
        print(f"Target minimum: 0.0 at (0, 0)")
        return overall_best_solution

