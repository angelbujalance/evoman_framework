import numpy as np
import os

from deap_evolution import DeapRunner
from enemy_groups import ENEMY_GROUP_1, ENEMY_GROUP_2, ALL_ENEMIES


def test_enemies(train_enemies: list, test_enemies: list, num_generations: int, experiment_name: str = "DEAP_testing"):
    deapRunner = DeapRunner(
        enemies=train_enemies, num_generations=num_generations, output_base_folder=experiment_name)
    folder = deapRunner.get_run_folder()
    bsol = np.loadtxt(os.path.join(folder, 'best.txt'))
    print('\nRUNNING SAVED BEST SOLUTION\n')
    output = deapRunner.evaluate([bsol])

    if output is None:
        return

    fitness = output[0]

    return fitness


if __name__ == "__main__":
    for group in [ENEMY_GROUP_1, ENEMY_GROUP_2]:
        for enemy in ALL_ENEMIES:
            test_enemies(train_enemies=group, test_enemies=[
                         enemy], num_generations=100)
