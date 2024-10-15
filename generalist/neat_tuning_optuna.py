import argparse
import ast

import optuna
# Tree-structured Parzen Estimator (TPE) sampler for better search efficiency
from optuna.samplers import TPESampler
import time
import os

from constants import (ENEMY_GROUP_1, ENEMY_GROUP_2,
                       NUM_GENERATIONS, NUM_TRIALS_NEAT,
                       OUTPUT_FOLDER_TUNING, OUTPUT_FOLDER_TUNING_BEST,
                       TUNING_POP_SIZE_MIN, TUNING_POP_SIZE_MAX)
from neat_evolution import NeatRunner
from neat_training import start_run

# values for testing. Uncomment just to test the file
#NUM_TRIALS_NEAT = 1
#NUM_GENERATIONS = 1


def run_optuna(enemies: list, n_trials: int, num_generations: int):
    # Record start time before Optuna study begins
    start_time = time.time()

    # Start the Optuna study for hyperparameter tuning
    study = optuna.create_study(direction='maximize', sampler=TPESampler())

    neatRunner = NeatRunner(enemies,
                            num_generations=num_generations,
                            model_folder=OUTPUT_FOLDER_TUNING,
                            results_folder=OUTPUT_FOLDER_TUNING)

    # Run trials of hyperparameter optimization
    study.optimize(lambda trial: objective(neatRunner, trial),
                   n_trials=n_trials)

    # Output the best parameters
    best_params = study.best_params
    print(f"Best Parameters for enemies {enemies}: {best_params}")

    # Once best parameters are found, you can run the evolutionary algorithm
    # again with the best parameters:
    neatRunner_best = start_run(enemies=enemies,
                                run_idx=0,
                                num_generations=num_generations,
                                model_folder=OUTPUT_FOLDER_TUNING_BEST,
                                results_folder=OUTPUT_FOLDER_TUNING_BEST)

    best_trial = study.best_trial
    print(f"Best trial: {best_trial.number}")
    print(f"Best fitness: {best_trial.value}")
    print(f"Best hyperparameters: {best_trial.params}")

    path = neatRunner_best.get_input_folder()
    file = os.path.join(path, "optuna_study_results.csv")

    # Save the Optuna study results for further analysis
    study.trials_dataframe().to_csv(file)

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
    pop_size = trial.suggest_int(
        'pop_size', TUNING_POP_SIZE_MIN, TUNING_POP_SIZE_MAX)
    bias_mutate_rate = trial.suggest_float(
        'bias_mutate_rate', 0.01, 0.6)  # Mutation rate
    response_mutate_rate = trial.suggest_float(
        'response_mutate_rate', 0.01, 0.6)  # Mutation rate
    weight_mutate_rate = trial.suggest_float(
        'weight_mutate_rate', 0.01, 0.6)
    elitism = trial.suggest_int('elitism', 0, 6)  # Number of elitism

    initial_mutation_rate = trial.suggest_float(
        'initial_mutation_rate', 0.01, 0.6)
    final_mutation_rate = trial.suggest_float(
        'final_mutation_rate', 0.01, 0.6)
    initial_crossover_rate = trial.suggest_float(
        'initial_crossover_rate', 0.01, 0.6)
    final_crossover_rate = trial.suggest_float(
        'final_crossover_rate', 0.01, 0.6)

    neatRunner.initial_mutation_rate = initial_mutation_rate
    neatRunner.final_mutation_rate = final_mutation_rate
    neatRunner.initial_crossover_rate = initial_crossover_rate
    neatRunner.final_crossover_rate = final_crossover_rate

    # Run the NEAT evolutionary algorithm
    neatRunner.set_params(bias_mutate_rate=bias_mutate_rate,
                          response_mutate_rate=response_mutate_rate,
                          weight_mutate_rate=weight_mutate_rate,
                          pop_size=pop_size,
                          elitism=elitism)

    winner = neatRunner.run_evolutionary_algorithm(trial.number)
    return winner.fitness


if __name__ == "__main__":
    #for group in [ENEMY_GROUP_1, ENEMY_GROUP_2]:
    parser = argparse.ArgumentParser(description='Parser to choose the group')

    parser.add_argument('group', type=str,
                        help='The group used to train the model')

    args = parser.parse_args()
    group = ast.literal_eval(args.group)

    print('-----------')
    print(f'START THE OPTUNA TUNNING OF GROUP 0F ENEMIES: {group}')
    run_optuna(enemies=group, n_trials=NUM_TRIALS_NEAT,
               num_generations=NUM_GENERATIONS)
