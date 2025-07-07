
import numpy as np

def run(
        initialization_function,
        fitness_function,
        selection_function,
        crossover_function,
        mutation_function,
        max_generations = 100,
        gene_range=[-5.12, 5.12],
        population_size=100,
        num_dimensions=2,
        elite_count=1,
        num_parents=2,
        crossover_rate=0.8,
        mutation_rate=0.05
    ):
    rng = np.random.default_rng(seed=42)
    #Initialize 
    population = initialization_function(rng, gene_range, population_size, num_dimensions)
    num_generations = 0
    while True:
        num_generations += 1
        #Calc fitness
        fitnesses = fitness_function(population)
        sorted_indices = np.argsort(fitnesses)[::-1]
        population = population[sorted_indices]
        fitnesses = fitnesses[sorted_indices]
        best_fitness = fitnesses[0]
        
        if num_generations >= max_generations:
            break

        #Elitism
        new_population = list(population[:elite_count].copy())

        #Create new population
        while len(new_population) < population_size:
            #Selection
            parents = selection_function(rng, population, fitnesses, num_parents) #Select num_parents parents
            #Crossover
            offspring = crossover_function(rng, parents[0], parents[1], crossover_rate, gene_range) #Crossover function
            #Mutation
            for i, _ in enumerate(offspring):
                offspring[i] = mutation_function(rng, offspring[i], mutation_rate, gene_range) #Mutation function
                if len(new_population) >= population_size:
                    break
                new_population.append(offspring[i])
        population = np.array(new_population)
    return best_fitness
    
