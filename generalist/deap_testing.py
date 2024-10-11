import numpy as np
import os

from deap_evolution import DeapRunner
from enemy_groups import ENEMY_GROUP_1, ENEMY_GROUP_2, ALL_ENEMIES
from general_testing_create_tables_plots import save_table_for_enemy_group


def eval_enemies(train_enemies: list, test_enemies: list, num_generations: int):
    deapRunner = DeapRunner(
        train_enemies=train_enemies, test_enemies=test_enemies,
        run_idx=0,
        num_generations=num_generations)
    folder = deapRunner.get_input_folder()
    bsol = np.loadtxt(os.path.join(folder, 'best.txt'))
    print('\nRUNNING SAVED BEST SOLUTION\n')
    fitness, player_energy, enemy_energy, time = deapRunner.run_game(bsol)

    return fitness, player_energy, enemy_energy, time


if __name__ == "__main__":
    num_generations = 1

    groups = [ENEMY_GROUP_1, ENEMY_GROUP_2]
    enemies = ALL_ENEMIES

    for group in groups:
        all_results = {}

        for enemy in enemies:
            results = eval_enemies(train_enemies=group, test_enemies=[enemy],
                                   num_generations=num_generations)

            all_results[enemy] = results

        save_table_for_enemy_group(all_results, "DEAP", group)
