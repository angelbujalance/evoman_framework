###############################################################################
# EvoMan FrameWork - V1.0 2016                                                #
# DEMO : Neuroevolution - Genetic Algorithm neural network.                   #
# Author: Karine Miras                                                        #
# karine.smiras@gmail.com                                                     #
###############################################################################

# imports framework
import sys
from evoman.environment import Environment
from demo_controller import player_controller
from enemy_groups import ENEMY_GROUP_1, ENEMY_GROUP_2, enemy_group_to_str

# imports other libs
import time
import numpy as np
import os
from deap import base, creator, tools, algorithms
import random
import csv
import optuna  # Import Optuna

CURRENT_ENEMY_GROUP = ENEMY_GROUP_1

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

for i_run in range(10):
    print("----------------------")
    print(f"Start running {i_run}")
    print("----------------------")

    str_enemy_group = enemy_group_to_str(CURRENT_ENEMY_GROUP)
    experiment_name = f'DEAPAexperimentE{str_enemy_group}/DEAP_runE{str_enemy_group}{i_run}'

    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                      enemies=CURRENT_ENEMY_GROUP,
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False,
                      multiplemode="yes")

    # same as the demo file
    env.state_to_log()  # checks environment state

    # same as the demo file
    n_vars = (env.get_num_sensors() + 1) * \
        n_hidden_neurons + (n_hidden_neurons + 1) * 5

    # DEAP setup for evolutionary algorithm
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("attr_float", random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_float, n=n_vars)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Genetic operators
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Evaluation function for DEAP
    def evaluate(individual):
        return (simulation(env, individual),)

    toolbox.register("evaluate", evaluate)

    gens = 30  # Number of generations

    # same as the demo file
    def simulation(env, x):
        f, p, e, t = env.play(pcont=np.array(x))
        return f

    # Initializes the population
    def run_evolutionary_algorithm(cxpb, mutpb, mu, lambda_):
        npop = mu  # Use mu as the population size

        pop = toolbox.population(n=npop)

        # Configure statistics and logbook
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Hall of Fame to keep track of the best individual
        hof = tools.HallOfFame(1, similar=np.array_equal)

        # Run the DEAP evolutionary algorithm (Mu, Lambda)
        final_pop, logbook = algorithms.eaMuCommaLambda(
            pop, toolbox, mu, lambda_, cxpb, mutpb, gens, stats=stats, halloffame=hof, verbose=True)

        return final_pop, hof, logbook

    # Save logbook results to a CSV file
    def save_logbook(logbook, filename):
        # Extract the relevant keys (such as 'gen', 'nevals', 'avg', 'std', 'min', 'max') from the logbook
        # Use the keys from the first logbook entry (assuming all entries have the same keys)
        keys = logbook[0].keys()

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            for record in logbook:
                writer.writerow(record)

    # Optuna Objective Function

    def objective(trial):
        # Suggest hyperparameters
        cxpb = trial.suggest_float('cxpb', 0.0, 0.9)
        mutpb = trial.suggest_float('mutpb', 0.0, 1.0 - cxpb)
        mu = trial.suggest_int('mu', 50, 100)
        lambda_ = trial.suggest_int('lambda_', 100, 200)

        # Run the DEAP evolutionary algorithm
        final_pop, hof, _ = run_evolutionary_algorithm(
            cxpb, mutpb, mu, lambda_)

        # Return the fitness of the best individual
        return hof[0].fitness.values[0]

    if __name__ == '__main__':
        run_mode = 'train'  # or 'test'

        if run_mode == 'test':
            bsol = np.loadtxt(experiment_name + '/best.txt')
            print('\nRUNNING SAVED BEST SOLUTION\n')
            env.update_parameter('speed', 'normal')
            evaluate([bsol])
            sys.exit(0)

        # Record start time before Optuna study begins
        start_time = time.time()

        # Start the Optuna study for hyperparameter tuning
        study = optuna.create_study(direction='maximize')
        # Run 26 trials of hyperparameter optimization
        study.optimize(objective, n_trials=26)

        # Output the best parameters
        best_params = study.best_params
        print(f"Best Parameters: {best_params}")

        # Once best parameters are found, you can run the evolutionary algorithm again with the best parameters:
        final_pop, hof, logbook = run_evolutionary_algorithm(
            best_params['cxpb'], best_params['mutpb'], best_params['mu'], best_params['lambda_'])

        # Save the best individual
        np.savetxt(experiment_name + '/best.txt', hof[0])

        # Save logbook results
        save_logbook(logbook, experiment_name + '/logbook.csv')

        # Print execution time
        end_time = time.time()
        print('\nExecution time: ' +
              str(round((end_time - start_time) / 60)) + ' minutes \n')
        print('\nExecution time: ' +
              str(round((end_time - start_time))) + ' seconds \n')

        # Save control (simulation has ended) file for bash loop
        with open(experiment_name + '/neuroended', 'w') as file:
            file.write('')

        env.state_to_log()  # checks environment state
