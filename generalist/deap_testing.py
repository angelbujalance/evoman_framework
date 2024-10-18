import numpy as np
import os
import pandas as pd

from constants import (ENEMY_GROUP_1, ENEMY_GROUP_2, ALL_ENEMIES, enemy_folder,
                       PATH_DEAP, NUM_RUNS, USE_CMA,
                       OUTPUT_FOLDER_TRAINING, OUTPUT_FOLDER_TESTING)
from deap_evolution import DeapRunner
from general_testing_create_tables_plots import save_table_for_enemy_group
# from best_individual_runs_DEAP import read_results

OUTPUT_FOLDER_TRAINING += '_100gens'
OUTPUT_FOLDER_TESTING += '_100gens'

print(OUTPUT_FOLDER_TRAINING)

def eval_enemies(train_enemies: list, test_enemies: list,
                 run_idx: int, use_cma: bool = False):
    deapRunner = DeapRunner(
        train_enemies=train_enemies, test_enemies=test_enemies,
        num_generations=0, use_cma=use_cma)
    deapRunner.run_idx = run_idx
    folder = deapRunner.get_input_folder()
    best_solution = np.loadtxt(os.path.join(folder, 'best.txt'))
    fitness, player_energy, enemy_energy, time = deapRunner.run_game(
        best_solution)

    gain = player_energy - enemy_energy
    return fitness, player_energy, enemy_energy, gain


def get_best_run_idx(enemy_group):
    relpath = os.path.join(PATH_DEAP, OUTPUT_FOLDER_TRAINING,
                           enemy_folder(enemy_group))

    all_best_fitness = []
    best_run_idx = 0

    for run_idx in range(NUM_RUNS):
        results_file = os.path.join(relpath, f"run_{run_idx}", "logbook.csv")

        try:
            data = pd.read_csv(results_file)
            best_fitness = data['max'].tolist()

            # Collect data for all generations
            all_best_fitness.append(best_fitness)
        except Exception as e:
            print(f"Error reading {results_file}: {e}")

    #     if best_fitness > all_best_fitness:
    #         run_idx = dir.removeprefix("run_")
    #         best_run_idx = run_idx

    # return int(best_run_idx)

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
    groups = [ENEMY_GROUP_1, ENEMY_GROUP_2]
    enemies = ALL_ENEMIES

    for group in groups:
        all_results = {}
        best_run_idx, best_generation_idx, max_fitness_value = \
            get_best_run_idx(group)

        for run_idx in range(NUM_RUNS):
            file = os.path.join(PATH_DEAP, OUTPUT_FOLDER_TESTING,
                                enemy_folder(group), f"run_{run_idx}",
                                "results.csv")

            os.makedirs(os.path.dirname(file), exist_ok=True)
            with open(file, "w") as f:
                f.write("enemy,fitness,player_energy,enemy_energy,gain\n")

            for enemy in enemies:
                fitness, player_energy, enemy_energy, gain = \
                    eval_enemies(train_enemies=group, test_enemies=[enemy],
                                 run_idx=run_idx, use_cma=USE_CMA)

                with open(file, "a") as f:
                    f.write(
                        ",".join(map(str, [enemy, fitness, player_energy,
                                           enemy_energy, gain]))
                        + "\n")

                all_results[enemy] = (
                    fitness, player_energy, enemy_energy, gain)

            if run_idx == best_run_idx:
                # Only save the best run as a table.
                save_table_for_enemy_group(
                    all_results, "DEAP", group, use_cma=USE_CMA)
