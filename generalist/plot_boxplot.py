import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

from constants import (PATH_DEAP, PATH_NEAT, OUTPUT_FOLDER_TESTING, NUM_RUNS,
                       enemy_folder, ENEMY_GROUP_1, ENEMY_GROUP_2)

from general_testing_create_tables_plots import save_table_ttest

EA_NAMES = ["DEAP", "NEAT"]
ENEMY_GROUPS = [ENEMY_GROUP_1, ENEMY_GROUP_2]


def enemy_group_to_pd_name(enemy_group: list):
    return str(enemy_group)


def collect_gains():
    dfs = []

    for name_EA, folder_EA in [("DEAP", PATH_DEAP), ("NEAT", PATH_NEAT)]:
        for enemy_group in ENEMY_GROUPS:
            str_enemy_group = enemy_group_to_pd_name(enemy_group)

            for run_idx in range(NUM_RUNS):
                results_file = os.path.join(folder_EA,
                                            OUTPUT_FOLDER_TESTING,
                                            enemy_folder(enemy_group),
                                            f"run_{run_idx}",
                                            "results.csv")

                if not os.path.exists(results_file):
                    continue

                df = pd.read_csv(results_file)
                df["EA"] = name_EA
                df["enemy_group"] = str_enemy_group
                dfs.append(df)

    combined_df = pd.concat(dfs)
    return combined_df


def perform_t_test(data: pd.DataFrame):
    means_NEAT = []
    means_DEAP = []
    p_values = []

    for enemy_group in ENEMY_GROUPS:
        str_enemy_group = enemy_group_to_pd_name(enemy_group)
        gains_NEAT = data[(data.EA == "NEAT") &
                          (data.enemy_group == str_enemy_group)]["gain"]

        gains_DEAP = data[(data.EA == "DEAP") &
                          (data.enemy_group == str_enemy_group)]["gain"]

        statistics, p = scipy.stats.ttest_ind(gains_NEAT, gains_DEAP)
        print(f"Group {str_enemy_group}: {p}")
        means_NEAT.append(round(gains_NEAT.mean(), 2))
        means_DEAP.append(round(gains_DEAP.mean(), 2))
        p_values.append(round(p, 5))

    save_table_ttest(ENEMY_GROUPS, means_NEAT, means_DEAP, p_values)


def create_boxplot(data: pd.DataFrame, output_path: str):
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

    output_path = os.path.join("results", "boxplots_100gen")
    perform_t_test(dfs)
    create_boxplot(dfs, output_path)
