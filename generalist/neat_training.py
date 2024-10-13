from enemy_groups import ENEMY_GROUP_1, ENEMY_GROUP_2
from neat_evolution import NeatRunner

NUM_RUNS = 10

# TODO: these are the old ones from task 1 and DEAP
best_params = {
    # 'cxpb': 0.58,
    # 'mutpb': 0.26,
    # 'mu': 71,
    # 'lambda_': 172
}


def start_run(run_idx: int, enemies: list, num_generations: int,
              output_base_folder: str = "trained"):
    neatRunner = NeatRunner(train_enemies=enemies,
                            num_generations=num_generations,
                            training_base_folder=output_base_folder)
    neatRunner.run_evolutionary_algorithm(run_idx)
    return neatRunner


def start_runs(enemies: list, n_runs: int, num_generations: int):
    for run_idx in range(n_runs):
        start_run(run_idx=run_idx, enemies=enemies,
                  num_generations=num_generations)


if __name__ == "__main__":
    for group in [ENEMY_GROUP_1, ENEMY_GROUP_2]:
        start_runs(enemies=group, num_generations=2,  # TODO: change back to 30
                   n_runs=NUM_RUNS, **best_params)
