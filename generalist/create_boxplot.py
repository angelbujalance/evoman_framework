import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from constants import (PATH_DEAP, PATH_NEAT, OUTPUT_FOLDER_TESTING, NUM_RUNS,
                       enemy_folder, enemy_group_to_str,
                       ENEMY_GROUP_1, ENEMY_GROUP_2)

EA_NAMES = ["DEAP", "NEAT"]
ENEMY_GROUPS = [ENEMY_GROUP_1, ENEMY_GROUP_2]


def collect_gains():
    dfs = []

    for name_EA, folder_EA in [("DEAP", PATH_DEAP), ("NEAT", PATH_NEAT)]:
        for enemy_group in ENEMY_GROUPS:
            str_enemy_group = str(enemy_group)

            for run_idx in range(NUM_RUNS):
                results_file = os.path.join(folder_EA,
                                            OUTPUT_FOLDER_TESTING,
                                            enemy_folder(enemy_group),
                                            f"run_{run_idx}",
                                            "results.csv")
                df = pd.read_csv(results_file)
                df["EA"] = name_EA
                df["enemy_group"] = str_enemy_group
                dfs.append(df)

    combined_df = pd.concat(dfs)
    return combined_df


def create_boxplot(data: pd.DataFrame, output_path: str):
    with open("testing.txt", "w")as f:
        f.write(str(data))

    sns.boxplot(x="EA", y="gain", hue="enemy_group", data=data)

    plt.xlabel("Evolutionary Algorithms")
    plt.ylabel("Gain")
    plt.title(
        "Gain of best individuals, trained models on two enemy groups")
    plt.legend()
    plt.ylim((-100, 100))
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, "gains.png"))
    plt.show()


if __name__ == "__main__":
    dfs = collect_gains()

    output_path = os.path.join("results", "boxplots")
    create_boxplot(dfs, output_path)
