from constants import (ENEMY_GROUP_1, ENEMY_GROUP_2,
                       NUM_RUNS, NUM_GENERATIONS,
                       OUTPUT_FOLDER_TRAINING, OUTPUT_FOLDER_TESTING)
from neat_evolution import NeatRunner

# TODO: these are the old ones from task 1 and DEAP
# This file is adjusted to run the default implementation of NEAT for Task 2
best_params = {'pop_size': 95, 'bias_mutate_rate': 0.4856922882200584,
    'response_mutate_rate': 0.15015508877056483, 'weight_mutate_rate': 0.5142192411719388,
    'elitism': 0, 'initial_mutation_rate': 0.5292155144805903,
    'final_mutation_rate': 0.3618820359917754, 'initial_crossover_rate': 0.3276435798729214,
    'final_crossover_rate': 0.06957466791681326,
    'num_hidden': 10} #obtained from group 1 with NEAT


def start_run(run_idx: int, enemies: list, num_generations: int, 
                pop_size: int, elitism: int,
                bias_mutate_rate : float,
                response_mutate_rate : float,
                weight_mutate_rate : float,
                num_hidden : int,
                initial_mutation_rate : float,
                initial_crossover_rate : float,
                final_mutation_rate : float,
                final_crossover_rate : float,
                model_folder: str = OUTPUT_FOLDER_TRAINING,
                results_folder: str = OUTPUT_FOLDER_TESTING
                ):
    neatRunner = NeatRunner(train_enemies=enemies,
                            num_generations=num_generations,
                            model_folder=model_folder,
                            results_folder=results_folder,
                            use_adjusted_mutation_rate=False,
                            initial_mutation_rate=initial_mutation_rate,
                            initial_crossover_rate=initial_crossover_rate,
                            final_mutation_rate=final_mutation_rate,
                            final_crossover_rate=final_crossover_rate)

    neatRunner.set_params(bias_mutate_rate=bias_mutate_rate,
                          response_mutate_rate=response_mutate_rate,
                          weight_mutate_rate=weight_mutate_rate,
                          pop_size=pop_size,
                          elitism=elitism,
                          n_hidden_neurons=num_hidden
                          )
    neatRunner.run_evolutionary_algorithm(run_idx)
    return neatRunner


def start_runs(enemies: list, n_runs: int, num_generations: int,
               pop_size: int, elitism: int, bias_mutate_rate: float,
               response_mutate_rate: float, weight_mutate_rate: float,
               initial_mutation_rate: float, initial_crossover_rate: float,
               final_mutation_rate: float, final_crossover_rate: float,
               num_hidden: int):
    for run_idx in range(n_runs):
        start_run(run_idx=run_idx, enemies=enemies,
                num_generations=num_generations,
                pop_size=pop_size, elitism=elitism,
                bias_mutate_rate=bias_mutate_rate,
                response_mutate_rate=response_mutate_rate,
                weight_mutate_rate=weight_mutate_rate,
                initial_mutation_rate=initial_mutation_rate,
                initial_crossover_rate=initial_crossover_rate,
                final_mutation_rate=final_mutation_rate,
                final_crossover_rate=final_crossover_rate,
                num_hidden=num_hidden)


if __name__ == "__main__":
    for group in [ENEMY_GROUP_1, ENEMY_GROUP_2]:
        start_runs(enemies=group, num_generations=NUM_GENERATIONS,
                   n_runs=NUM_RUNS, **best_params)
