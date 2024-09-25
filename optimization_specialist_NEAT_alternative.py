# TODO: Came up with an alternative way to configure baseline NEAT

# imports framework
from evoman.environment import Environment
from neat_controller import PlayerControllerNEAT

# imports other libs
import time
import numpy as np
import glob
import os
import neat
import visualize
from neat_population import Population

# Parameters for neat.Checkpointer
GENERATION_INTERVAL = 5
CHECKPOINT_PREFIX = 'neat-checkpoint-'

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

for i_run in range(10):
    print("----------------------")
    print(f"Start running {i_run}")
    print("----------------------")

    experiment_name = f'NEAT_experiment/NEAT_run{i_run}'

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

    # Create results file to store fitness metrics
    fitness_log_file = open(f'{experiment_name}/results.txt', 'w')
    fitness_log_file.write("best_fitness,mean_fitness,std_fitness\n")


    # runs simulation
    def simulation(env, x):
        f, p, e, t = env.play(pcont=x)
        return f, p, e, p-e


    # code adapted from https://neat-python.readthedocs.io/en/latest/xor_example.html
    def eval_genomes(genomes, config) -> None:
        """
        Fitness function which sets
        """

        fitness_values = []
        for genome_id, genome in genomes:
            # genome.fitness = 4.0
            net = neat.nn.FeedForwardNetwork.create(genome, config)

            [genome.fitness, genome.player_energy, genome.enemy_energy,
                genome.individual_gain] = simulation(env, net)

            fitness_values.append(genome.fitness)

        # Calculate fitness metrics
        best_fitness = max(fitness_values)
        mean_fitness = np.mean(fitness_values)
        std_fitness = np.std(fitness_values)

        # Log the metrics to the file
        fitness_log_file.write(f"{best_fitness},{mean_fitness},{std_fitness}\n")


    def create_population(checkpoint_folder: str, config: neat.Config):
        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))

        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        cp_prefix = os.path.join(checkpoint_folder, CHECKPOINT_PREFIX)
        p.add_reporter(neat.Checkpointer(generation_interval=GENERATION_INTERVAL,
                                         filename_prefix=cp_prefix))
        os.makedirs(checkpoint_folder, exist_ok=True)
        return p


    def get_population(checkpoint_folder: str, config: neat.Config):
        os.makedirs(checkpoint_folder, exist_ok=True)
        files = os.listdir(checkpoint_folder)

        if len(files) == 0:
            return create_population(checkpoint_folder, config)

        def get_latest_file(f):
            return os.path.getctime(os.path.join(checkpoint_folder, f))

        filename = max(os.listdir(checkpoint_folder), key=get_latest_file)
        file = os.path.join(checkpoint_folder, filename)
        return neat.Checkpointer.restore_checkpoint(file)


    def run(config_file: str, checkpoint_folder: str):
        # Load configuration.
        species_set = neat.DefaultSpeciesSet
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             species_set, neat.DefaultStagnation,
                             config_file)

        p = get_population(checkpoint_folder, config)

        # Run for up to 300 generations.
        winner = p.run(eval_genomes, 30)

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))

        # Show output of the most fit genome against training data.
        print('\nOutput:')

        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

        [winner.fitness, winner.player_energy, winner.enemy_energy,
         winner.individual_gain] = simulation(env, winner_net)

        print("Winner fitness: {:.3f}, player_energy: {:.3f}, enemy_energy: {:.3f}, individual_gain: {:.3f}".format(winner.fitness, winner.player_energy, winner.enemy_energy,
              winner.individual_gain))
        # neat.Checkpointer.save_checkpoint(config, p, species_set, self.current_generation)
        # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
        # p.run(eval_genomes, 10)


    if __name__ == '__main__':
        # Determine path to configuration file. This path manipulation is
        # here so that the script will run successfully regardless of the
        # current working directory.
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config_specialist_NEAT')
        checkpoint_path = os.path.join(local_dir, experiment_name, 'checkpoints')

        run(config_path, checkpoint_path)

        fim = time.time() # prints total execution time for experiment
        print('\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
        print('\nExecution time: '+str(round((fim-ini)))+' seconds \n')

        file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
        file.close()

        env.state_to_log()  # checks environment state
