# TODO: Came up with an alternative way to configure baseline NEAT

# imports framework
from evoman.environment import Environment
from comma_controller import PlayerControllerCOMMA

# imports other libs
import time
import numpy as np
import glob
import os
import neat
import visualize
from neat_population import Population
from deap import base, creator, tools, algorithms
import array
import random

# Parameters for neat.Checkpointer
GENERATION_INTERVAL = 5
CHECKPOINT_PREFIX = 'comma-checkpoint-'

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

for i_run in range(10):
    print("----------------------")
    print(f"Start running {i_run}")
    print("----------------------")

    experiment_name = f'COMMAexperiment/COMMA_run{i_run}'

    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10

    player_controller = PlayerControllerCOMMA(n_hidden_neurons)
    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                      enemies=[8],
                      playermode="ai",
                      player_controller=player_controller,
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)

    # default environment fitness is assumed for experiment

    env.state_to_log()  # checks environment state

    # Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorithm

    ini = time.time()  # sets time marker

    # genetic algorithm params

    run_mode = 'train'  # train or NEAT_run0

    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors() + 1) * \
        n_hidden_neurons + (n_hidden_neurons + 1) * 5

    dom_u = 1
    dom_l = -1
    npop = 100
    gens = 30
    mutation = 0.2
    last_best = 0
    # parameters for comma 
    # I took whatever we used for NEAT or took from example code https://github.com/DEAP/deap/blob/60913c5543abf8318ddce0492e8ffcdf37974d86/examples/es/fctmin.py
    mu = 100  # Number of parents
    lambda_ = 100  # Number of offspring
    cxpb = 0.6  # Crossover
    mutpb = 0.2  # Mutation 
    ngen = 30  # Number of generations

    # Create results file to store fitness metrics
    fitness_log_file = open(f'{experiment_name}/results.txt', 'w')
    fitness_log_file.write("best_fitness,mean_fitness,std_fitness\n")


    # runs simulation
    def simulation(env, weights):
        # Use PlayerControllerCOMMA to get actions
        player_controller.set_weights(weights, env.get_num_sensors())
        # Play the game using the DEAP individual's weights
        f, p, e, t = env.play(pcont=None)  # pcont=None because control happens inside the environment
        return f, p, e, p-e

    

    # DEAP comma implementation based on https://github.com/DEAP/deap/blob/60913c5543abf8318ddce0492e8ffcdf37974d86/examples/es/fctmin.py
    MIN_VALUE = -1
    MAX_VALUE = 1
    MIN_STRATEGY = 0.5
    MAX_STRATEGY = 3

    # we maximimze the fitnessfunction so max instead of min
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMax, strategy=None)
    creator.create("Strategy", array.array, typecode="d")

    # Individual generator
    def generateES(icls, scls, size, imin, imax, smin, smax):
        ind = icls(random.uniform(imin, imax) for _ in range(size))
        ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
        return ind

    def checkStrategy(minstrategy):
        def decorator(func):
            def wrapper(*args, **kwargs):
                children = func(*args, **kwargs)
                for child in children:
                    for i, s in enumerate(child.strategy):
                        if s < minstrategy:
                            child.strategy[i] = minstrategy
                return children
            return wrapper
        return decorator


    toolbox = base.Toolbox()
    toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
                     n_vars, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxESBlend, alpha=0.1)
    toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
    toolbox.register("select", tools.selTournament, tournsize=3)

    toolbox.decorate("mate", checkStrategy(MIN_STRATEGY))
    toolbox.decorate("mutate", checkStrategy(MIN_STRATEGY))



    # eval function for deap, 
    # notice: we don't save the best, mean std here (as in neat but do that in the run function instead
    def eval_individual(individual):
        weights = np.array(individual)
        fitness, player_energy, enemy_energy, gain = simulation(env, weights)
        return fitness,  # Return a tuple (DEAP expects a tuple)

    toolbox.register("evaluate", eval_individual)

    def run_comma():
        # Initialize the population
        pop = toolbox.population(n=mu)

        # save mean and std etc,
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("max", np.max)

        # Hall of fame -> keep track of the current best
        hof = tools.HallOfFame(1)
        for gen in range(ngen):
            pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=mu, lambda_=lambda_,
                                                  cxpb=cxpb, mutpb=mutpb, ngen=1,
                                                  stats=stats, halloffame=hof, verbose=False)
            best_fitness = logbook[-1]['max']
            mean_fitness = logbook[-1]['avg']
            std_fitness = logbook[-1]['std']

            # Write statistics to the log file for each generation
            fitness_log_file.write(f"{best_fitness},{mean_fitness},{std_fitness}\n")


        # Save best result
        best_individual = hof[0]
        best_fitness = best_individual.fitness.values[0]

        print(f"Best fitness: {best_fitness}")

        return best_individual, pop, logbook



    if __name__ == '__main__':
        # Determine path to configuration file. This path manipulation is
        # here so that the script will run successfully regardless of the
        # current working directory.
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config_specialist_COMMA')
        checkpoint_path = os.path.join(local_dir, experiment_name, 'checkpoints')

        # Run the DEAP EA algorithm
        best_individual, pop, logbook = run_comma()

        # Log the results to the file
        # fitness_log_file.write(f"{best_individual.fitness.values[0]},{np.mean([ind.fitness.values[0] for ind in pop])},{np.std([ind.fitness.values[0] for ind in pop])}\n")
        fitness_log_file.close()

        fim = time.time() # prints total execution time for experiment
        print('\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
        print('\nExecution time: '+str(round((fim-ini)))+' seconds \n')

        file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
        file.close()

        env.state_to_log()  # checks environment state