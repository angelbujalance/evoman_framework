from constants import (ENEMY_GROUP_1, ENEMY_GROUP_2,
                       NUM_RUNS, NUM_GENERATIONS,
                       OUTPUT_FOLDER_TRAINING, OUTPUT_FOLDER_TESTING)
from neat_evolution import NeatRunner

# TODO: these are the old ones from task 1 and DEAP
best_params = {
    # 'cxpb': 0.58,
    # 'mutpb': 0.26,
    # 'mu': 71,
    # 'lambda_': 172
}


def start_run(run_idx: int, enemies: list, num_generations: int,
              model_folder: str = OUTPUT_FOLDER_TRAINING,
              results_folder: str = OUTPUT_FOLDER_TESTING):
    neatRunner = NeatRunner(train_enemies=enemies,
                            num_generations=num_generations,
                            model_folder=model_folder,
                            results_folder=results_folder)
    neatRunner.run_evolutionary_algorithm(run_idx)
    return neatRunner


def start_runs(enemies: list, n_runs: int, num_generations: int):
    for run_idx in range(n_runs):
        start_run(run_idx=run_idx, enemies=enemies,
                  num_generations=num_generations)


if __name__ == "__main__":
    for group in [ENEMY_GROUP_1, ENEMY_GROUP_2]:
        start_runs(enemies=group, num_generations=NUM_GENERATIONS,
                   n_runs=NUM_RUNS, **best_params)
