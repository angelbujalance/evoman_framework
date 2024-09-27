###############################################################################
# EvoMan FrameWork - V1.0 2016                                                #
# DEMO : Neuroevolution - Genetic Algorithm neural network.                   #
# Author: Karine Miras                                                        #
# karine.smiras@gmail.com                                                     #
###############################################################################

# imports framework
import sys
from evoman.environment import Environment
from deap_controller import player_controller

# imports other libs
import time
import numpy as np
import os
from deap import base, creator, tools, algorithms
import random
import csv

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

for i_run in range(10):
    print("----------------------")
    print(f"Start running {i_run}")
    print("----------------------")

    experiment_name = f'DEAPAexperimentE2/DEAP_runE2{i_run}'

    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)


    n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=[2],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons),
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)

    # same as the demo file
    env.state_to_log()  # checks environment state

    # same as the demo file
    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5


    # DEAP setup for evolutionary algorithm
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("attr_float", random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_vars)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Genetic operators
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Evaluation function for DEAP
    def evaluate(individual):
        return (simulation(env, individual),)
    #should this be:
        return np.array(list(map(lambda y: simulation(env,y), x)))

    toolbox.register("evaluate", evaluate)

    # genetic algorithm params, these same as the demo file
    dom_u = 1
    dom_l = -1
    npop = 100  # Population size
    gens = 30  # Number of generations
    mutpb = 0.2  # Mutation probability

    cxpb = 0.5  # Crossover probability
    mu = npop  # Number of individuals to select
    lambda_ = npop * 2  # Number of offspring to generate

    # same as the demo file
    def simulation(env, x):
        f, p, e, t = env.play(pcont=x)
        return f


    # Initializes the population
    def run_evolutionary_algorithm():
        pop = toolbox.population(n=npop)

        # Configure statistics and logbook
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Hall of Fame to keep track of the best individual
        hof = tools.HallOfFame(1)

        # Run the DEAP evolutionary algorithm (Mu, Lambda)
        final_pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu, lambda_, cxpb, mutpb, gens, stats=stats, halloffame=hof, verbose=True)

        return final_pop, hof, logbook

    # Save logbook results to a CSV file
    def save_logbook(logbook, filename):
        # Extract the relevant keys (such as 'gen', 'nevals', 'avg', 'std', 'min', 'max') from the logbook
        keys = logbook[0].keys()  # Use the keys from the first logbook entry (assuming all entries have the same keys)
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            for record in logbook:
                writer.writerow(record)


    # loads file with the best solution for testing
    if __name__ == '__main__':
        run_mode = 'train'  # or 'test'

        if run_mode == 'test':
            bsol = np.loadtxt(experiment_name + '/best.txt')
            print('\nRUNNING SAVED BEST SOLUTION\n')
            env.update_parameter('speed', 'normal')
            evaluate([bsol])
            sys.exit(0)

        # Start the evolutionary process
        start_time = time.time()

        # Run the DEAP evolutionary algorithm
        final_pop, hof, logbook = run_evolutionary_algorithm()

        # Save the best individual
        np.savetxt(experiment_name + '/best.txt', hof[0])

        # Save logbook results
        # np.savetxt(experiment_name + '/logbook.txt', np.array(logbook))
        save_logbook(logbook, experiment_name + '/logbook.csv')

        # Print execution time
        end_time = time.time()
        print('\nExecution time: ' + str(round((end_time - start_time) / 60)) + ' minutes \n')
        print('\nExecution time: ' + str(round((end_time - start_time))) + ' seconds \n')

        # Save control (simulation has ended) file for bash loop
        with open(experiment_name + '/neuroended', 'w') as file:
            file.write('')

        env.state_to_log()  # checks environment state
