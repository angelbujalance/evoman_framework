import numpy as np
import optuna
import time
import sys
import os

from enemy_groups import ENEMY_GROUP_1, ENEMY_GROUP_2
from neat_evolution import NeatRunner
from neat_training import start_run

NUM_RUNS = 10


def run_optuna(enemies: list, n_trials: int, num_generations: int):
    # Record start time before Optuna study begins
    start_time = time.time()

    # Start the Optuna study for hyperparameter tuning
    study = optuna.create_study(direction='maximize')

    neatRunner = NeatRunner(enemies, num_generations=num_generations,
                            run_idx=0, training_base_folder="tuning")

    # Run trials of hyperparameter optimization
    study.optimize(lambda trial: objective(neatRunner, trial),
                   n_trials=n_trials)

    # Output the best parameters
    best_params = study.best_params
    print(f"Best Parameters for enemies {enemies}: {best_params}")

    # Once best parameters are found, you can run the evolutionary algorithm again with the best parameters:
    neatRunner_best = start_run(enemies=enemies,
                                run_idx=0,
                                num_generations=num_generations,
                                cxpb=best_params['cxpb'],
                                mutpb=best_params['mutpb'],
                                mu=best_params['mu'],
                                lambda_=best_params['lambda_'],
                                output_base_folder="tuned")

    final_pop, hof, logbook = neatRunner_best.get_results()

    path = neatRunner_best.get_input_folder()

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


def objective(neatRunner: NeatRunner, trial: optuna.Trial):
    """
    Optuna Objective Function
    """
    # Suggest hyperparameters
    n_hidden_neurons = trial.suggest_int(
        'num_hidden', 5, 20)  # Number of hidden neurons
    mutation_rate = trial.suggest_float(
        'mutation_rate', 0.01, 0.5)  # Mutation rate
    pop_size = trial.suggest_int('pop_size', 50, 200)  # Population size
    elitism = trial.suggest_int('elitism', 1, 30)  # Number of elitism
    num_generations = trial.suggest_int(
        'num_generations', 5, 30)  # Number of generations

    # Run the NEAT evolutionary algorithm
    neatRunner.set_params(n_hidden_neurons=n_hidden_neurons, mutation_rate=mutation_rate,
                          pop_size=pop_size, elitism=elitism, num_generations=num_generations)
    final_pop, hof, logbook = neatRunner.run_evolutionary_algorithm()
    neatRunner.save_logbook()

    _, hof, _ = neatRunner.get_results()

    # Return the fitness of the best individual
    return hof[0].fitness.values[0]


if __name__ == "__main__":
    for group in [ENEMY_GROUP_1, ENEMY_GROUP_2]:
        run_optuna(enemies=group, n_trials=50, num_generations=30)
