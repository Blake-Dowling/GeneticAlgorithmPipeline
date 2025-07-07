
import numpy as np


#################### Initialization Functions ####################
def initialize_random(rng, gene_range, population_size, num_dimensions):
    return rng.uniform(gene_range[0], gene_range[1], (population_size, num_dimensions))

#################### Fitness Functions ####################
def de_jongs_f1(x):
    y= -np.sum(x**2, axis=1)
    return y

#################### Selection Functions ####################
def tournament_selection(rng, population, fitnesses, num_parents):
    parents = []
    for _ in range(num_parents):
        random_indices = rng.choice(len(population), size=3, replace=False) #Select 3 random indices from population
        fittest_selection_index = np.argmax(fitnesses[random_indices]) #Index with greatest fitness of selected 3
        fittest_population_index = random_indices[fittest_selection_index] #Index in population of fittest of selection
        parents.append(population[fittest_population_index])
    return np.array(parents)

def linear_crossover(rng, parent1, parent2, crossover_rate, gene_range):
    if rng.random() < crossover_rate:
        alpha = rng.random() #Blending factor
        offspring1 = alpha * parent1 + (1 - alpha) * parent2
        offspring2 = alpha * parent2 + (1 - alpha) * parent1
        offspring1 = np.clip(offspring1, gene_range[0], gene_range[1])
        offspring2 = np.clip(offspring2, gene_range[0], gene_range[1])
        return np.array([offspring1, offspring1])
    else:
        return np.array([parent1.copy(), parent2.copy()])
    
def uniform_mutate(rng, individual, mutation_rate, gene_range):
    mutated_individual = individual.copy()
    for i, _ in enumerate(mutated_individual):
        if rng.random() < mutation_rate:
            mutation_strength = 0.05 * (gene_range[1] - gene_range[0])
            mutated_individual[i] += rng.uniform(-mutation_strength, mutation_strength)
            mutated_individual[i] = np.clip(mutated_individual[i], gene_range[0], gene_range[1])
    return mutated_individual
