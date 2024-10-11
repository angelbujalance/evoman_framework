import os

from enemy_groups import enemy_group_to_str


def create_table_for_enemy_group(stats_per_enemy_all_runs: dict):
    """
    stats_per_enemy_all_runs: key is enemy number,
        value is tuple (fitness, player_energy, enemy_energy, time)
    """
    latex_lines = []
    test_enemies = list(stats_per_enemy_all_runs.keys())
    amount_columns = 1 + len(test_enemies)
    columns_code = "|".join(amount_columns * "c")

    latex_lines.append(r"\begin{tabular}{|"+columns_code+"|}\n\hline")
    enemies_code = " & ".join(map(str, test_enemies))
    latex_lines.append(fr"Enemy & {enemies_code}\\")

    player_healths = []
    enemy_healths = []

    for enemy in test_enemies:
        fitness, player_energy, enemy_energy, time = stats_per_enemy_all_runs[enemy]
        player_healths.append(round(player_energy))
        enemy_healths.append(round(enemy_energy))

    latex_player_healths = " & ".join(map(str, player_healths))
    latex_enemy_healths = " & ".join(map(str, enemy_healths))
    latex_lines.append("\hline")
    latex_lines.append(fr"Player health & {latex_player_healths}\\")
    latex_lines.append("\hline")
    latex_lines.append(fr"Enemy health & {latex_enemy_healths}\\")
    latex_lines.append("\hline")
    latex_lines.append(r"\end{tabular}")
    return "\n".join(latex_lines)


def save_table_for_enemy_group(stats_per_enemy_all_runs: dict, name_EA: str, train_enemy_group: list):
    latex_lines = []

    tabular_line = create_table_for_enemy_group(stats_per_enemy_all_runs)

    latex_lines.append(r"\begin{table}[ht]")
    latex_lines.append("\n\centering")
    latex_lines += "\n" + tabular_line + "\n"
    caption = f"Final healths of player and enemy. With {name_EA} model trained on group {train_enemy_group}."
    latex_lines.append(r"\caption{" + caption + "}\n")

    str_enemy_group = enemy_group_to_str(train_enemy_group)
    latex_lines.append(
        "\\label{tab:test_" + name_EA + "_" + str_enemy_group + "}\n")
    latex_lines.append(r"\end{table}")

    print()
    relpath = os.path.join(f"results {name_EA}", "tables")
    os.makedirs(relpath, exist_ok=True)
    file = os.path.join(relpath, str_enemy_group) + ".tex"
    print(latex_lines)
    with open(file, "w") as f:
        f.writelines(latex_lines)
