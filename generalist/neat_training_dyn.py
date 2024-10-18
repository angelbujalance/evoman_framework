from constants import (ENEMY_GROUP_1, ENEMY_GROUP_2,
                       NUM_RUNS, NUM_GENERATIONS,
                       OUTPUT_FOLDER_TRAINING, OUTPUT_FOLDER_TESTING)
from neat_evolution import NeatRunner

OUTPUT_FOLDER_TRAINING += '_dynamic_full'
OUTPUT_FOLDER_TESTING += '_dynamic_full'

print(OUTPUT_FOLDER_TRAINING)

# TODO: these are the old ones from task 1 and DEAP
# This file is adjusted to run the default implementation of NEAT for Task 2
best_params = {'pop_size': 93,
               'elitism': 0,
               'num_hidden': 28}



def start_run(run_idx: int, enemies: list, num_generations: int, 
                pop_size: int, elitism: int,
                num_hidden : int,
                model_folder: str = OUTPUT_FOLDER_TRAINING,
                results_folder: str = OUTPUT_FOLDER_TESTING
                ):
    neatRunner = NeatRunner(train_enemies=enemies,
                            num_generations=num_generations,
                            model_folder=model_folder,
                            results_folder=results_folder,
                            use_adjusted_mutation_rate=True)

    neatRunner.set_params(pop_size=pop_size,
                          elitism=elitism,
                          n_hidden_neurons=num_hidden
                          )
    neatRunner.run_evolutionary_algorithm(run_idx)
    return neatRunner


def start_runs(enemies: list, n_runs: int, num_generations: int,
               pop_size: int, elitism: int,
               num_hidden: int):
    for run_idx in range(n_runs):
        start_run(run_idx=run_idx, enemies=enemies,
                num_generations=num_generations,
                pop_size=pop_size, elitism=elitism,
                num_hidden=num_hidden)


if __name__ == "__main__":
    for group in [ENEMY_GROUP_1, ENEMY_GROUP_2]:
        start_runs(enemies=group, num_generations=NUM_GENERATIONS,
                   n_runs=NUM_RUNS, **best_params)