import os

from constants import (enemy_group_to_str, PATH_DEAP, PATH_NEAT)


def create_table_for_enemy_group(stats_per_enemy_all_runs: dict):
    """
    stats_per_enemy_all_runs: key is enemy number,
        value is tuple (fitness, player_energy, enemy_energy, gain)
    """
    latex_lines = []
    test_enemies = list(stats_per_enemy_all_runs.keys())
    amount_columns = 1 + len(test_enemies)
    columns_code = "|".join(amount_columns * "c")

    latex_lines.append(r"\begin{tabular}{|"+columns_code+"|}")
    latex_lines.append(r"\hline")
    enemies_code = " & ".join(map(str, test_enemies))
    latex_lines.append(fr"Enemy & {enemies_code}\\")

    player_healths = []
    enemy_healths = []

    for enemy in test_enemies:
        fitness, player_energy, enemy_energy, gain = \
            stats_per_enemy_all_runs[enemy]
        player_healths.append(round(player_energy))
        enemy_healths.append(round(enemy_energy))

    latex_player_healths = " & ".join(map(str, player_healths))
    latex_enemy_healths = " & ".join(map(str, enemy_healths))
    latex_lines.append("\hline")
    latex_lines.append(fr"Player health & {latex_player_healths}\\")
    latex_lines.append("\hline")
    latex_lines.append(fr"Enemy health & {latex_enemy_healths}\\")
    latex_lines.append(r"\hline")
    latex_lines.append(r"\end{tabular}")
    return "\n".join(latex_lines)


def save_table_for_enemy_group(stats_per_enemy_all_runs: dict, name_EA: str,
                               train_enemy_group: list, use_cma: bool = False):
    latex_lines = []

    tabular_line = create_table_for_enemy_group(stats_per_enemy_all_runs)

    latex_lines.append(r"\begin{table}[ht]")
    latex_lines.append(r"\centering")
    latex_lines += "\n" + tabular_line + "\n"
    cma_text = " with CMA-ES " if use_cma else ""
    caption = f"Final healths of player and enemy. With {name_EA}" \
        f"{cma_text} model trained on group {train_enemy_group}."
    latex_lines.append(r"\caption{" + caption + "}\n")

    str_enemy_group = enemy_group_to_str(train_enemy_group)
    latex_lines.append(
        "\\label{tab:test_" + name_EA + "_" + str_enemy_group + "}\n")
    latex_lines.append(r"\end{table}")

    results_folder = PATH_NEAT if name_EA == "NEAT" else PATH_DEAP
    relpath = os.path.join(results_folder, "tables")
    os.makedirs(relpath, exist_ok=True)
    file = os.path.join(relpath, str_enemy_group + ".tex")

    with open(file, "w") as f:
        f.writelines(latex_lines)


def save_table_ttest(enemy_groups, means_NEAT, means_DEAP, p_values):
    latex_lines = ""
    latex_lines = \
        r"""
    \begin{table}[ht]
        \centering
        \begin{tabular}{c|c|c|c}
            & mean NEAT & mean DEAP & p-value \\
            \hline
    """

    for enemy_group, mean_NEAT, mean_DEAP, p in zip(enemy_groups, means_NEAT, means_DEAP, p_values):

        latex_lines += \
            fr"""
                Enemies {enemy_group} & {mean_NEAT} & {mean_DEAP} & {p} \\
        """

    latex_lines += \
        r"""
        \end{tabular}
        \caption{T-test NEAT vs. MuCommaLambda (DEAP) with their mean individual gain and the p-values between them.}
        \label{tab:ttest_results}
    \end{table}
    """

    relpath = os.path.join("results", "general")
    os.makedirs(relpath, exist_ok=True)
    file = os.path.join(relpath, "t_test.tex")

    with open(file, "w") as f:
        f.writelines(latex_lines)
