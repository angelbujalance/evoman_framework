from constants import (ENEMY_GROUP_1, ENEMY_GROUP_2,
                       NUM_GENERATIONS, NUM_RUNS, USE_CMA,
                       OUTPUT_FOLDER_TRAINING, OUTPUT_FOLDER_TESTING)
from deap_evolution import DeapRunner


print(OUTPUT_FOLDER_TRAINING)

def start_run(run_idx: int, enemies: list,
              cxpb: float = None, mutpb: float = None, mu: float = None, lambda_: float = None,
              sigma: float = None,  # Add sigma for CMA-ES
              num_generations: int = NUM_GENERATIONS,
              model_folder: str = OUTPUT_FOLDER_TRAINING,
              results_folder: str = OUTPUT_FOLDER_TESTING,
              use_cma: bool = False): 
    deapRunner = DeapRunner(train_enemies=enemies,
                            num_generations=num_generations,
                            model_folder=model_folder,
                            results_folder=results_folder,
                            use_cma=use_cma)
    
    # Conditionally set parameters based on use_cma flag
    if use_cma:
        deapRunner.set_params(mu=mu, lambda_=lambda_, sigma=sigma, use_cma=True)
    else:
        deapRunner.set_params(cxpb=cxpb, mutpb=mutpb, mu=mu, lambda_=lambda_, use_cma=False)
    
    final_pop, hof, logbook = deapRunner.run_evolutionary_algorithm(run_idx)
    deapRunner.save_logbook()
    return deapRunner


def start_runs(enemies: list, n_runs: int, num_generations: int,
               cxpb: float = None, mutpb: float = None,
               mu: float = None, lambda_: float = None,
               sigma: float = 1.0,  # Add sigma for CMA-ES
               use_cma: bool = False):
    for run_idx in range(n_runs):
        start_run(run_idx=run_idx, enemies=enemies,
                  num_generations=num_generations,
                  cxpb=cxpb, mutpb=mutpb, mu=mu, lambda_=lambda_, 
                  sigma=sigma,  # Pass sigma to start_run for CMA-ES
                  use_cma=use_cma)


# if __name__ == "__main__":
#     for group in [ENEMY_GROUP_1, ENEMY_GROUP_2]:
#         start_runs(enemies=group, num_generations=NUM_GENERATIONS,
#                    use_cma=USE_CMA, n_runs=NUM_RUNS,
#                    **best_params)

if __name__ == "__main__":
    # Set parameters based on whether we are using CMA-ES or MuCommaLambda
    if USE_CMA:
        # For CMA-ES, no need for cxpb and mutpb
        best_params = {
            'mu': 57,
            'lambda_': 117,
            'sigma': 1.36 
        }
        print("Using Optuna's CMA-ES params: ", best_params)
    else:
        # For MuCommaLambda, we use cxpb and mutpb
        best_params = {
            'cxpb': 0.58,
            'mutpb': 0.26,
            'mu': 71,
            'lambda_': 172
        }

    # Run for both enemy groups
    for group in [ENEMY_GROUP_1, ENEMY_GROUP_2]:
        start_runs(enemies=group, num_generations=NUM_GENERATIONS,
                   use_cma=USE_CMA, n_runs=NUM_RUNS,
                   **best_params)