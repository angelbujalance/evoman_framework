import numpy as np
import optuna
import time
import os

from constants import (ENEMY_GROUP_1, ENEMY_GROUP_2,
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
                            results_folder=OUTPUT_FOLDER_TUNING)

    # Run trials of hyperparameter optimization
    study.optimize(lambda trial: objective(deapRunner, trial),
                   n_trials=n_trials)

    # Output the best parameters
    best_params = study.best_params
    print(f"Best Parameters for enemies {enemies}: {best_params}")

    # Once best parameters are found, you can run the evolutionary algorithm
    # again with the best parameters:
    deapRunner_best = start_run(enemies=enemies,
                                run_idx=0,
                                num_generations=num_generations,
                                cxpb=best_params['cxpb'],
                                mutpb=best_params['mutpb'],
                                mu=best_params['mu'],
                                lambda_=best_params['lambda_'],
                                model_folder=OUTPUT_FOLDER_TUNING_BEST,
                                results_folder=OUTPUT_FOLDER_TUNING_BEST)

    final_pop, hof, logbook = deapRunner_best.get_results()

    path = deapRunner_best.get_input_folder()

    best_trial = study.best_trial
    print(f"Best trial: {best_trial.number}")
    print(f"Best fitness: {best_trial.value}")
    print(f"Best hyperparameters: {best_trial.params}")

    path = deapRunner_best.get_input_folder()
    file = os.path.join(path, "optuna_study_results.csv")

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
        # For CMA-ES optimization, suggest sigma and lambda
        sigma = trial.suggest_float('sigma', 0.1, 2.0)
        mu = trial.suggest_int('mu', 10, 200)
        lambda_ = trial.suggest_int('lambda_', 50, 200)
        deapRunner.set_params(mu=mu, lambda_=lambda_, sigma=sigma)
    else:
        # Suggest hyperparameters
        cxpb = trial.suggest_float('cxpb', 0.0, 0.9)
        mutpb = trial.suggest_float('mutpb', 0.0, 1.0 - cxpb)
        mu = trial.suggest_int('mu', TUNING_POP_SIZE_MIN, TUNING_POP_SIZE_MAX)
        lambda_ = trial.suggest_int('lambda_', 100, 200)
        deapRunner.set_params(cxpb=cxpb, mutpb=mutpb, mu=mu, lambda_=lambda_)

    # Run the DEAP evolutionary algorithm
    
    final_pop, hof, logbook = deapRunner.run_evolutionary_algorithm(trial.number)
    deapRunner.save_logbook()

    _, hof, _ = deapRunner.get_results()

    # Return the fitness of the best individual
    return hof[0].fitness.values[0]


if __name__ == "__main__":
    for group in [ENEMY_GROUP_1, ENEMY_GROUP_2]:
        run_optuna(enemies=group, n_trials=NUM_TRIALS_DEAP,
                   num_generations=NUM_GENERATIONS)
