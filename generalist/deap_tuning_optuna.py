import numpy as np
import optuna
import time
import os

from constants import (ENEMY_GROUP_1, ENEMY_GROUP_2, USE_CMA,
                       NUM_GENERATIONS,  NUM_TRIALS_DEAP,
                       OUTPUT_FOLDER_TUNING, OUTPUT_FOLDER_TUNING_BEST,
                       TUNING_POP_SIZE_MIN, TUNING_POP_SIZE_MAX)
from deap_evolution import DeapRunner
from deap_training import start_run

NUM_RUNS = 10


def run_optuna(enemies: list, n_trials: int, num_generations: int):
    # Record start time before Optuna study begins
    start_time = time.time()

    # Start the Optuna study for hyperparameter tuning
    study = optuna.create_study(direction='maximize')

    deapRunner = DeapRunner(enemies, num_generations=num_generations,
                            model_folder=OUTPUT_FOLDER_TUNING,
                            results_folder=OUTPUT_FOLDER_TUNING,
                            use_cma=USE_CMA)

    # Run trials of hyperparameter optimization
    study.optimize(lambda trial: objective(deapRunner, trial),
                   n_trials=n_trials)

    # Output the best parameters
    best_params = study.best_params
    print(f"Best Parameters for enemies {enemies}: {best_params}")

    # Handle best_params based on whether CMA-ES is used or not
    if USE_CMA:
        # Ensure best_params contain the right values for CMA-ES
        assert 'mu' in best_params and 'lambda_' in best_params and 'sigma' in best_params, "Missing CMA-ES parameters."
        # Pass parameters relevant for CMA-ES
        deapRunner_best = start_run(enemies=enemies,
                                    run_idx=0,
                                    num_generations=num_generations,
                                    mu=best_params['mu'],
                                    lambda_=best_params['lambda_'],
                                    sigma=best_params['sigma'],  # Add sigma for CMA-ES
                                    model_folder=OUTPUT_FOLDER_TUNING_BEST,
                                    results_folder=OUTPUT_FOLDER_TUNING_BEST,
                                    use_cma=USE_CMA)
    else:
        # Ensure best_params contain the right values for MuCommaLambda
        assert 'cxpb' in best_params and 'mutpb' in best_params and 'mu' in best_params and 'lambda_' in best_params, "Missing MuCommaLambda parameters."
        # Pass parameters relevant for MuCommaLambda
        deapRunner_best = start_run(enemies=enemies,
                                    run_idx=0,
                                    num_generations=num_generations,
                                    cxpb=best_params['cxpb'],
                                    mutpb=best_params['mutpb'],
                                    mu=best_params['mu'],
                                    lambda_=best_params['lambda_'],
                                    model_folder=OUTPUT_FOLDER_TUNING_BEST,
                                    results_folder=OUTPUT_FOLDER_TUNING_BEST,
                                    use_cma=USE_CMA)

    final_pop, hof, logbook = deapRunner_best.get_results()

    path = deapRunner_best.get_input_folder()

    best_trial = study.best_trial
    print(f"Best trial: {best_trial.number}")
    print(f"Best fitness: {best_trial.value}")
    print(f"Best hyperparameters: {best_trial.params}")

    # Save the best individual
    np.savetxt(os.path.join(path, 'best.txt'), hof[0])

    # Print execution time
    end_time = time.time()
    print('\nExecution time: ' +
          str(round((end_time - start_time) / 60)) + ' minutes \n')
    print('\nExecution time: ' +
          str(round((end_time - start_time))) + ' seconds \n')

    # Save control (simulation has ended) file for bash loop
    with open(os.path.join(path, 'neuroended'), 'w') as file:
        file.write('')



def objective(deapRunner: DeapRunner, trial: optuna.Trial):
    """
    Optuna Objective Function
    """
    if deapRunner.use_cma:
        print("\n\n\nCAME IN HERE FOR THE OPTUNA PARAM SETUP\n\n\n")
        # For CMA-ES optimization, suggest sigma and lambda
        sigma = trial.suggest_float('sigma', 0.1, 2.0)
        mu = trial.suggest_int('mu', 10, 200)
        lambda_ = trial.suggest_int('lambda_', 50, 200)
        deapRunner.set_params(mu=mu, lambda_=lambda_, sigma=sigma, use_cma=True)
    else:
        # Suggest hyperparameters
        cxpb = trial.suggest_float('cxpb', 0.0, 0.9)
        mutpb = trial.suggest_float('mutpb', 0.0, 1.0 - cxpb)
        mu = trial.suggest_int('mu', TUNING_POP_SIZE_MIN, TUNING_POP_SIZE_MAX)
        lambda_ = trial.suggest_int('lambda_', 100, 200)
        deapRunner.set_params(cxpb=cxpb, mutpb=mutpb, mu=mu, lambda_=lambda_)

    # Run the DEAP evolutionary algorithm
    
    _, hof, _ = deapRunner.run_evolutionary_algorithm(trial.number)
    # Ensure mu and lambda_ are compatible with the problem size
    if mu > deapRunner.n_vars:
        raise ValueError(f"mu ({mu}) cannot be greater than the number of variables ({deapRunner.n_vars})")
    if lambda_ <= mu:
        raise ValueError(f"lambda_ ({lambda_}) must be greater than mu ({mu})")
    deapRunner.save_logbook()

    _, hof, _ = deapRunner.get_results()

    # Return the fitness of the best individual
    return hof[0].fitness.values[0]


if __name__ == "__main__":
    for group in [ENEMY_GROUP_1]:
        run_optuna(enemies=group, n_trials=NUM_TRIALS_DEAP,
                   num_generations=NUM_GENERATIONS)
