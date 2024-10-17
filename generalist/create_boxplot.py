import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from constants import (PATH_DEAP, PATH_NEAT, OUTPUT_FOLDER_TESTING, NUM_RUNS,
                       enemy_folder, enemy_group_to_str,
                       ENEMY_GROUP_1, ENEMY_GROUP_2)


def collect_gains():
    name_EAs = ["DEAP", "NEAT"]
    enemy_groups = [ENEMY_GROUP_1, ENEMY_GROUP_2]

    all_gains = {(name_EA, enemy_group_to_str(enemy_group)): []
                 for name_EA in name_EAs
                 for enemy_group in enemy_groups}

    for name_EA, folder_EA in [("DEAP", PATH_DEAP), ("NEAT", PATH_NEAT)]:
        for enemy_group in [ENEMY_GROUP_1, ENEMY_GROUP_2]:
            str_enemy_group = enemy_group_to_str(enemy_group)
            gains_enemy_group = []

            for run_idx in range(NUM_RUNS):
                results_file = os.path.join(folder_EA,
                                            OUTPUT_FOLDER_TESTING,
                                            enemy_folder(enemy_group),
                                            f"run_{run_idx}",
                                            "results.csv")
                gain = pd.read_csv(results_file)["gain"].apply(np.mean)
                gains_enemy_group.append(gain)

            all_gains[name_EA, str_enemy_group].append(gains_enemy_group)

    return all_gains


def create_boxplot(gains_per_EA_and_enemy_group: dict, output_path: str):
    # print(dfs_per_EA_per_enemy_group)
    combis = list(gains_per_EA_and_enemy_group.keys())

    # print(all_gains)

    labels = combis

    with open("testing.txt", "w")as f:
        f.write(str(gains_per_EA_and_enemy_group))

    # for name_EA in name_EAs:
    #     for enemy_group in enemy_groups:
    #         gains = dfs_per_EA_per_enemy_group[name_EA][enemy_group]
    plt.boxplot(gains_per_EA_and_enemy_group, patch_artist=True, labels=labels)

    plt.xlabel('Evolutionary Algorithms')
    plt.ylabel('Gain')
    plt.title('Average gain of best individual for the trained models')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, 'gains.png'))
    plt.show()


if __name__ == "__main__":
    dfs = collect_gains()

    output_path = os.path.join("results", "boxplots")
    create_boxplot(dfs, output_path)
