import numpy as np
import os
import pandas as pd

from constants import (ENEMY_GROUP_1, ENEMY_GROUP_2, ALL_ENEMIES,
                       PATH_DEAP)
from deap_evolution import DeapRunner
from general_testing_create_tables_plots import save_table_for_enemy_group
# from best_individual_runs_DEAP import read_results


def eval_enemies(train_enemies: list, test_enemies: list,
                 run_idx: int):
    deapRunner = DeapRunner(
        train_enemies=train_enemies, test_enemies=test_enemies,
        num_generations=0)
    deapRunner.run_idx = run_idx
    folder = deapRunner.get_input_folder()
    best_solution = np.loadtxt(os.path.join(folder, 'best.txt'))
    print(f'\nRUNNING SAVED BEST SOLUTION OF RUN {run_idx}\n')
    fitness, player_energy, enemy_energy, time = deapRunner.run_game(
        best_solution)

    return fitness, player_energy, enemy_energy, time


def get_best_run_idx(folder, num_runs):
    all_best_fitness = []

    for i in range(num_runs):
        results_path = os.path.join(
            folder, 'logbook.csv')
        try:
            data = pd.read_csv(results_path)
            best_fitness = data['max'].tolist()

            # Collect data for all generations
            all_best_fitness.append(best_fitness)
        except Exception as e:
            print(f"Error reading {results_path}: {e}")

    # Find the best run by max fitness across all generations and runs
    all_best_fitness = np.array(all_best_fitness)
    max_fitness_value = np.max(all_best_fitness)
    best_run_idx, best_generation_idx = np.unravel_index(
        np.argmax(all_best_fitness), all_best_fitness.shape)

    print(
        f"Highest Best Fitness: {max_fitness_value}, found in run " +
        f"{best_run_idx}, generation {best_generation_idx}")

    return best_run_idx, best_generation_idx, max_fitness_value


if __name__ == "__main__":
    num_runs = 10

    groups = [ENEMY_GROUP_1, ENEMY_GROUP_2]
    enemies = ALL_ENEMIES

    best_run_idx, _, _ = get_best_run_idx(PATH_DEAP, num_runs)

    for group in groups:
        all_results = {}

        for enemy in enemies:
            results = eval_enemies(train_enemies=group, test_enemies=[enemy],
                                   run_idx=best_run_idx)

            all_results[enemy] = results

        save_table_for_enemy_group(all_results, "DEAP", group)
