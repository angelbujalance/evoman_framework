# TODO: Came up with an alternative way to configure baseline NEAT

# imports framework
from evoman.environment import Environment
from neat_controller import PlayerControllerNEAT
from neat_population import CustomPopulation
from constants import ENEMY_GROUP_1, ENEMY_GROUP_2, enemy_group_to_str

# imports other libs
import time
import numpy as np
import glob
import os
import neat
import pickle
from neat.reporting import StdOutReporter

CURRENT_ENEMY_GROUP = ENEMY_GROUP_1

# Parameters for neat.Checkpointer
GENERATION_INTERVAL = 1
CHECKPOINT_PREFIX = 'neat-checkpoint-'
num_runs = 10
num_gens = 26

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

for i_run in range(num_runs):
    print("----------------------")
    print(f"Start running {i_run}")
    print("----------------------")

    experiment_name = f'NEAT_experiment/enemy_{enemy_group_to_str(CURRENT_ENEMY_GROUP)}/NEAT_run{i_run}'

    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 20

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                      enemies=CURRENT_ENEMY_GROUP,
                      playermode="ai",
                      player_controller=PlayerControllerNEAT(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False,
                      multiplemode="yes")

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
    fitness_log_file = open(f'{experiment_name}/results.csv', 'w')
    fitness_log_file.write("best_fitness,mean_fitness,std_fitness,gain\n")

    # runs simulation

    def simulation(env, x):
        f, p, e, t = env.play(pcont=x)
        return f, p, e, p-e

    # first implemenattion of dynamic mutation and crossover rates
    def adjust_mutation_and_crossover_rates(config, current_generation, max_generations):
        # we can change these later to what we'd like
        initial_mutation_rate = 0.5
        final_mutation_rate = 0.1
        initial_crossover_rate = 0.7
        final_crossover_rate = 0.3
        # Linear decrease of mutation rate over generations
        new_mutation_rate = initial_mutation_rate - \
            (initial_mutation_rate - final_mutation_rate) * \
            (current_generation / max_generations)
        new_crossover_rate = initial_crossover_rate - \
            (initial_crossover_rate - final_crossover_rate) * \
            (current_generation / max_generations)

        # Update mutation rates in the configuration
        config.genome_config.conn_add_prob = new_mutation_rate
        config.genome_config.conn_delete_prob = new_mutation_rate
        config.genome_config.node_add_prob = new_mutation_rate
        config.genome_config.node_delete_prob = new_mutation_rate
        config.genome_config.weight_mutate_rate = new_mutation_rate

        # Allow multiple structural mutations if needed
        config.genome_config.single_structural_mutation = False
        config.genome_config.crossover_rate = new_crossover_rate

        print(
            f"Adjusted mutation rate to {new_mutation_rate:.4f} and crossover rate to {new_crossover_rate:.4f} for generation {current_generation}")

    # code adapted from https://neat-python.readthedocs.io/en/latest/xor_example.html

    def eval_genomes(genomes, config) -> None:
        """
        Fitness function which sets
        """
        global current_generation
        # adjust_mutation_and_crossover_rates(config, current_generation, num_gens)

        fitness_values = []
        gain_values = []
        for genome_id, genome in genomes:
            # genome.fitness = 4.0
            net = neat.nn.FeedForwardNetwork.create(genome, config)

            [genome.fitness, genome.player_energy, genome.enemy_energy,
                genome.gain] = simulation(env, net)

            fitness_values.append(genome.fitness)
            gain_values.append(genome.gain)

        # Check how many genomes have valid fitness values
        valid_fitness_values = [fv for fv in fitness_values if fv is not None]

        print(
            f"Number of genomes including None scores: {len(fitness_values)}")
        print(
            f"Number of genomes with valid fitness: {len(valid_fitness_values)}")
        if valid_fitness_values:
            best_fitness = max(valid_fitness_values)
            mean_fitness = np.mean(valid_fitness_values)
            std_fitness = np.std(valid_fitness_values)
            best_gain = gain_values[fitness_values.index(best_fitness)]

            # Log the metrics to the file
            fitness_log_file.write(
                f"{best_fitness},{mean_fitness},{std_fitness},{best_gain}\n")
            fitness_log_file.flush()  # Ensure that the data is written to the file

            # Print statistics for the current generation
            print(f"Generation {current_generation}: Max Fitness = {best_fitness:.4f}, "
                  f"Average Fitness = {mean_fitness:.4f}, Std Dev = {std_fitness:.4f}")

        else:
            print(
                f"Warning: No valid fitness values found in generation {current_generation}.")

        current_generation += 1

    def create_population(checkpoint_folder: str, config: neat.Config):
        # Create the population, which is the top-level object for a NEAT run.
        p = CustomPopulation(config)

        # Add a stdout reporter to show progress in the terminal.
        # Add the StdOutReporter for console output
        reporter = StdOutReporter(True)
        p.add_reporter(reporter)

        # reporter = Reporter(True, i_run, enemy_number)
        # p.add_reporter(reporter)

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
        # print("files", files)

        if len(files) == 0:
            # print("no files yet")
            return create_population(checkpoint_folder, config)

        def get_latest_file(f):
            # print("file chosenaa: ", os.path.getctime(os.path.join(checkpoint_folder, f)))
            return os.path.getctime(os.path.join(checkpoint_folder, f))

        filename = max(os.listdir(checkpoint_folder), key=get_latest_file)
        # print("filename", filename)
        file = os.path.join(checkpoint_folder, filename)
        print("file", file)

        # print("return", neat.Checkpointer.restore_checkpoint(file))
        return neat.Checkpointer.restore_checkpoint(file)

    def get_weights_network(winner):
        weights = []

        for connection_key, connection in winner.connections.items():
            weight = connection.weight

            weights.append(weight)

        return weights

    def run(config_file: str, checkpoint_folder: str):
        global current_generation
        current_generation = 0  # Initialize the current generation counter
        # Load configuration.
        species_set = neat.DefaultSpeciesSet
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             species_set, neat.DefaultStagnation,
                             config_file)

        # Create the custom population
        # p = CustomPopulation(config)
        p = get_population(checkpoint_folder, config)

        # Add the statistics reporter to track stats over generations
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        # Run for up to 26 generations. Estimated with optuna
        winner, best = p.run(eval_genomes, num_gens)

        # Display the winning genome.
        # print('\nBest genome:\n{!s}'.format(winner))

        # Show output of the most fit genome against training data.
        # print('\nOutput:')

        best_file_name = f'best_individual_run{i_run}'

        with open(os.path.join(experiment_name, best_file_name), 'wb') as file_out:
            pickle.dump(best, file_out)

        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

        [winner.fitness, winner.player_energy, winner.enemy_energy,
         winner.individual_gain] = simulation(env, winner_net)

        print("Winner fitness: {:.3f}, player_energy: {:.3f}, enemy_energy: {:.3f}, individual_gain: {:.3f}".format(winner.fitness, winner.player_energy, winner.enemy_energy,
              winner.individual_gain))

        winner_weights = get_weights_network(winner)
        np.savetxt(os.path.join(experiment_name,
                   best_file_name + ".txt"), winner_weights)

        # Print statistics overview
        print('\n=== Overall Evolution Statistics ===')
        for i in range(num_gens):
            gen_stats = stats.get_fitness_stdev(i)
            print(f"Generation {i}: Max Fitness = {stats.most_fit_genomes[i].fitness}, "
                  f"Avg Fitness = {gen_stats[0]}, Std Dev = {gen_stats[1]}")

        # Final save and cleanup
        fim = time.time()  # prints total execution time for experiment
        print('\nExecution time: ' + str(round((fim - ini) / 60)) + ' minutes \n')
        print('\nExecution time: ' + str(round((fim - ini))) + ' seconds \n')
        file = open(experiment_name + '/neuroended', 'w')
        file.close()
        env.state_to_log()  # checks environment state

    if __name__ == '__main__':
        # Determine path to configuration file. This path manipulation is
        # here so that the script will run successfully regardless of the
        # current working directory.
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config_generalist_NEAT')
        checkpoint_path = os.path.join(
            local_dir, experiment_name, 'checkpoints')

        run(config_path, checkpoint_path)

        fim = time.time()  # prints total execution time for experiment
        print('\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
        print('\nExecution time: '+str(round((fim-ini)))+' seconds \n')

        # saves control (simulation has ended) file for bash loop file
        file = open(experiment_name+'/neuroended', 'w')
        file.close()

        env.state_to_log()  # checks environment state
