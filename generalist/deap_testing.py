import numpy as np
import os

from deap_evolution import DeapRunner
from enemy_groups import ENEMY_GROUP_1, ENEMY_GROUP_2, ALL_ENEMIES, enemy_group_to_str


def eval_enemies(train_enemies: list, test_enemies: list, num_generations: int,
                 num_runs: int = 10):
    gains = []

    for run_idx in range(num_runs):
        deapRunner = DeapRunner(
            train_enemies=train_enemies, test_enemies=test_enemies,
            run_idx=run_idx,
            num_generations=num_generations)
        folder = deapRunner.get_input_folder()
        bsol = np.loadtxt(os.path.join(folder, 'best.txt'))
        print('\nRUNNING SAVED BEST SOLUTION\n')
        fitness, player_energy, enemy_energy, time = deapRunner.run_game(bsol)
        gain = player_energy - enemy_energy
        gains.append(gain)

    return gains


def plot_gains(all_gains: dict):
    ...


if __name__ == "__main__":
    num_runs = 10
    num_generations = 1

    groups = [ENEMY_GROUP_1]
    enemies = [1]

    # groups=[ENEMY_GROUP_1, ENEMY_GROUP_2]
    # enemies = ALL_ENEMIES
    all_gains = {}

    for group in groups:
        for enemy in enemies:
            gains = eval_enemies(train_enemies=group, test_enemies=[enemy],
                                 num_generations=num_generations,
                                 num_runs=num_runs)

            all_gains.setdefault(enemy_group_to_str(group), {})[enemy] = gains

    plot_gains(all_gains)
