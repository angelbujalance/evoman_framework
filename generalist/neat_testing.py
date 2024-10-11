import numpy as np
import os
import pickle

from neat_evolution import NeatRunner
from enemy_groups import ENEMY_GROUP_1, ENEMY_GROUP_2, ALL_ENEMIES, enemy_group_to_str
from general_testing_create_tables_plots import save_table_for_enemy_group


def eval_enemies(train_enemies: list, test_enemies: list,
                 run_idx: int):
    neatRunner = NeatRunner(
        train_enemies=train_enemies, test_enemies=test_enemies,
        run_idx=run_idx, num_generations=0)
    folder = neatRunner.get_input_folder()

    print(f'\nRUNNING SAVED BEST SOLUTION OF RUN {run_idx}\n')
    best_file = os.path.join(folder, f'best_individual_run{run_idx}')
    fitness, player_energy, enemy_energy, time = neatRunner.evaluate_from_genome_file(
        best_file)

    return fitness, player_energy, enemy_energy, time


def get_best_run_idx(enemy_group):
    str_enemy_group = enemy_group_to_str(enemy_group)
    relpath = os.path.join("results NEAT", "trained",
                           f"enemies_{str_enemy_group}")

    best_run_idx = 0
    all_best_fitness = 0

    for dir in os.listdir(relpath):
        results_file = os.path.join(relpath, dir, "results.txt")

        lines = np.genfromtxt(results_file, skip_header=True, delimiter=",")
        print(lines)
        line = lines[-1] if lines[0] is list else lines  # last generation
        best_fitness, mean_fitness, std_fitness, gain = line

        if best_fitness > all_best_fitness:
            run_idx = dir.removeprefix("run_")
            best_run_idx = run_idx

    return best_run_idx


if __name__ == "__main__":
    groups = [ENEMY_GROUP_1, ENEMY_GROUP_2]
    enemies = ALL_ENEMIES

    for group in groups:
        all_results = {}
        best_run_idx = get_best_run_idx(group)

        for enemy in enemies:
            results = eval_enemies(train_enemies=group, test_enemies=[enemy],
                                   run_idx=best_run_idx)

            all_results[enemy] = results

        save_table_for_enemy_group(all_results, "NEAT", group)
