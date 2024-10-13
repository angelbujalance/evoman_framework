from constants import (ENEMY_GROUP_1, ENEMY_GROUP_2,
                       NUM_GENERATIONS, NUM_RUNS,
                       OUTPUT_FOLDER_TRAINING, OUTPUT_FOLDER_TESTING)
from deap_evolution import DeapRunner


# TODO: these are the old ones from task 1
best_params = {
    'cxpb': 0.58,
    'mutpb': 0.26,
    'mu': 71,
    'lambda_': 172
}


def start_run(run_idx: int, enemies: list,
              cxpb: float, mutpb: float, mu: float, lambda_: float,
              num_generations: int,
              model_folder: str = OUTPUT_FOLDER_TRAINING,
              results_folder: str = OUTPUT_FOLDER_TESTING):
    deapRunner = DeapRunner(train_enemies=enemies,
                            num_generations=num_generations,
                            model_folder=model_folder,
                            results_folder=results_folder)
    deapRunner.set_params(cxpb=cxpb, mutpb=mutpb, mu=mu, lambda_=lambda_)
    final_pop, hof, logbook = deapRunner.run_evolutionary_algorithm(run_idx)
    deapRunner.save_logbook()
    return deapRunner


def start_runs(enemies: list, n_runs: int, num_generations: int,
               cxpb: float, mutpb: float, mu: float, lambda_: float):
    for run_idx in range(n_runs):
        start_run(run_idx=run_idx, enemies=enemies,
                  num_generations=num_generations,
                  cxpb=cxpb, mutpb=mutpb, mu=mu, lambda_=lambda_)


if __name__ == "__main__":
    for group in [ENEMY_GROUP_1, ENEMY_GROUP_2]:
        start_runs(enemies=group, num_generations=NUM_GENERATIONS,
                   n_runs=NUM_RUNS, **best_params)
