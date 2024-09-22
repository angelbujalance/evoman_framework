

# imports framework
from evoman.environment import Environment
from neat_controller import PlayerControllerNEAT

# imports other libs
import time
import numpy as np
import glob
import os
import neat
from neat_population import Population

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'optimization_test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[8],
                  playermode="ai",
                  player_controller=PlayerControllerNEAT(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

# default environment fitness is assumed for experiment

env.state_to_log()  # checks environment state

# Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorithm

ini = time.time()  # sets time marker

# genetic algorithm params

run_mode = 'train'  # train or test

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors() + 1) * \
    n_hidden_neurons + (n_hidden_neurons + 1) * 5

dom_u = 1
dom_l = -1
npop = 100
gens = 30
mutation = 0.2
last_best = 0


# runs simulation
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f, p, e, p-e


# code adapted from https://neat-python.readthedocs.io/en/latest/xor_example.html
def eval_genomes(genomes, config) -> None:
    """
    Fitness function which sets
    """

    for genome_id, genome in genomes:
        # genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        [genome.fitness, genome.player_energy, genome.enemy_energy,
            genome.individual_gain] = simulation(env, net)

        # NOTE: Return values are ignored by neat.Population.run(eval_genomes)
        # return [genome.fitness, genome.player_energy,
        #         genome.enemy_energy, genome.individual_gain]


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    [winner.fitness, winner.player_energy, winner.enemy_energy,
     winner.individual_gain] = simulation(env, winner_net)

    print("Winner fitness: {:.3f}, player_energy: {:.3f}, enemy_energy: {:.3f}, individual_gain: {:.3f}".format(winner.fitness, winner.player_energy, winner.enemy_energy,
          winner.individual_gain))
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_specialist_NEAT')
    run(config_path)
