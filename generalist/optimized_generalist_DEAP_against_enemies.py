import sys
from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
import os
import random
import csv
import json

from deap import base, creator, tools, algorithms
from optimization_generalist_DEAP_plus_optuna import run_evolutionary_algorithm, save_logbook

from enemy_groups import ENEMY_GROUP_1, ENEMY_GROUP_2, enemy_group_to_str

CURRENT_ENEMY_GROUP = ENEMY_GROUP_1

# Load the best parameters from a JSON file


def load_optimized_parameters(filename):
    with open(filename, 'r') as f:
        return json.load(f)


best_params = {
    'cxpb': 0.58,
    'mutpb': 0.26,
    'mu': 71,
    'lambda_': 172
}

if __name__ == '__main__':
    # Load the best parameters from the previous Optuna optimization

    num_runs = 10

    for i_run in range(num_runs):
        print("----------------------")
        print(
            f"Start running against enemy group {CURRENT_ENEMY_GROUP}, run {i_run}")
        print("----------------------")

        start_time = time.time()

        str_enemy_group = enemy_group_to_str(CURRENT_ENEMY_GROUP)
        experiment_name = f'DEAPexperimentE{str_enemy_group}/DEAP_runE{str_enemy_group}{i_run}'

        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)

        n_hidden_neurons = 10

        # Initialize environment for the specific enemy
        env = Environment(experiment_name=experiment_name,
                          enemies=CURRENT_ENEMY_GROUP,  # Set the current enemy
                          playermode="ai",
                          player_controller=player_controller(
                              n_hidden_neurons),
                          enemymode="static",
                          level=2,
                          speed="fastest",
                          visuals=False,
                          multiplemode="yes")

        env.state_to_log()  # checks environment state

        # Run the evolutionary algorithm with the best parameters for this enemy
        final_pop, hof, logbook = run_evolutionary_algorithm(
            best_params['cxpb'], best_params['mutpb'], best_params['mu'], best_params['lambda_']
        )  # the best params could be a dictionary with the best parameters

        # Save the best individual for this run
        np.savetxt(experiment_name + '/best.txt', hof[0])

        # Save logbook results
        save_logbook(logbook, experiment_name + '/logbook.csv')

        # Print execution time for each run
        end_time = time.time()
        print('\nExecution time for run: ' +
              str(round((end_time - start_time) / 60)) + ' minutes \n')
        print('\nExecution time: ' +
              str(round((end_time - start_time))) + ' seconds \n')

        # Save control (simulation has ended) file for bash loop
        with open(experiment_name + '/neuroended', 'w') as file:
            file.write('')

        env.state_to_log()  # checks environment state
