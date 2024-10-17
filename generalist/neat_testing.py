import numpy as np
import os

from constants import (ENEMY_GROUP_1, ENEMY_GROUP_2, ALL_ENEMIES,
                       enemy_folder, PATH_NEAT, NUM_RUNS,
                       OUTPUT_FOLDER_TRAINING, OUTPUT_FOLDER_TESTING)
from neat_evolution import NeatRunner
from general_testing_create_tables_plots import save_table_for_enemy_group


def eval_enemies(train_enemies: list, test_enemies: list,
                 run_idx: int):
    neatRunner = NeatRunner(train_enemies=train_enemies,
                            test_enemies=test_enemies,
                            num_generations=0)
    neatRunner.run_idx = run_idx
    folder = neatRunner.get_input_folder()

    print(f'\nRUNNING SAVED BEST SOLUTION OF RUN {run_idx}\n')
    best_file = os.path.join(folder, f'best_individual_run{run_idx}')
    fitness, player_energy, enemy_energy, gain = \
        neatRunner.evaluate_from_genome_file(best_file)

    return fitness, player_energy, enemy_energy, gain


def get_best_run_idx(enemy_group):
    relpath = os.path.join(PATH_NEAT, OUTPUT_FOLDER_TRAINING,
                           enemy_folder(enemy_group))

    best_run_idx = 0
    all_best_fitness = 0

    for dir in os.listdir(relpath):
        results_file = os.path.join(relpath, dir, "results.csv")

        lines = np.genfromtxt(results_file, skip_header=True, delimiter=",")
        # last generation
        line = lines[-1] if hasattr(lines[0], "__len__") else lines
        generation, best_fitness, mean_fitness, std_fitness = line

        if best_fitness > all_best_fitness:
            run_idx = dir.removeprefix("run_")
            best_run_idx = run_idx

    return int(best_run_idx)


if __name__ == "__main__":
    groups = [ENEMY_GROUP_1, ENEMY_GROUP_2]
    enemies = ALL_ENEMIES

    for group in groups:
        all_results = {}
        best_run_idx = get_best_run_idx(group)

        for run_idx in range(NUM_RUNS):
            file = os.path.join(PATH_NEAT, OUTPUT_FOLDER_TESTING,
                                enemy_folder(group), f"run_{run_idx}",
                                "results.csv")

            os.makedirs(os.path.dirname(file), exist_ok=True)
            with open(file, "w") as f:
                f.write("enemy,fitness,player_energy,enemy_energy,gain\n")

            for enemy in enemies:
                fitness, player_energy, enemy_energy, gain = \
                    eval_enemies(train_enemies=group, test_enemies=[enemy],
                                 run_idx=run_idx)

                with open(file, "a") as f:
                    f.write(
                        ",".join(map(str, [enemy, fitness, player_energy,
                                           enemy_energy, gain]))
                        + "\n")

                all_results[enemy] = (
                    fitness, player_energy, enemy_energy, gain)

            if run_idx == best_run_idx:
                # Only save the best run as a table.
                save_table_for_enemy_group(all_results, "NEAT", group)
