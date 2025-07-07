import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


import warnings
from runga import run

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cluster._kmeans")

class GAOptimizer(BaseEstimator, RegressorMixin):
    def __init__(self,
                 initialization_function,
                 fitness_function,
                 selection_function,
                 crossover_function,
                 mutation_function,
                 gene_range=[-5.12, 5.12],
                 population_size=100,
                 max_generations=100,
                 num_dimensions=2,
                 elite_count=1,
                 num_parents=2,
                 crossover_rate=0.8,
                 mutation_rate=0.05,
                 n_ga_runs_per_eval=5,
                 verbose_ga=False
                 ):
        self.initialization_function = initialization_function
        self.fitness_function = fitness_function
        self.selection_function = selection_function
        self.crossover_function = crossover_function
        self.mutation_function = mutation_function
        self.max_generations = max_generations
        self.gene_range = gene_range
        self.population_size = population_size
        self.num_dimensions = num_dimensions
        self.elite_count = elite_count
        self.num_parents = num_parents
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n_ga_runs_per_eval = n_ga_runs_per_eval
        self.verbose_ga = verbose_ga

        self.avg_best_objective_ = None
        self.best_solution_found_ = None
    
    def fit(self, X, y=None):
        best_objectives_across_runs = []
        for i in range(self.n_ga_runs_per_eval):
            best_obj_this_run = run(
                initialization_function=self.initialization_function,
                fitness_function=self.fitness_function,
                selection_function=self.selection_function,
                crossover_function=self.crossover_function,
                mutation_function=self.mutation_function,
                max_generations=self.max_generations,
                gene_range=self.gene_range,
                population_size = self.population_size,
                num_dimensions=self.num_dimensions,
                elite_count=self.elite_count,
                num_parents=self.num_parents,
                crossover_rate = self.crossover_rate,
                mutation_rate = self.mutation_rate
            )
        best_objectives_across_runs.append(best_obj_this_run)
        if self.verbose_ga:
            print(f"    GA Run {i+1}/{self.n_ga_runs_per_eval}: Best Objective = {best_obj_this_run:.6f}")

        self.avg_best_objective_ = np.mean(best_objectives_across_runs)
        print(f"  Avg Best Objective over {self.n_ga_runs_per_eval} runs: {self.avg_best_objective_:.6f}")
        
        return self
    
    def score(self, X, y=None):
        if self.avg_best_objective_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.avg_best_objective_ 