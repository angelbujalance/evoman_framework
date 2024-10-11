# TODO: Came up with an alternative way to configure baseline NEAT

# imports framework
from evoman.environment import Environment
from neat_controller import PlayerControllerNEAT
from neat_population import CustomPopulation
from enemy_groups import ENEMY_GROUP_1, ENEMY_GROUP_2, enemy_group_to_str

# imports other libs
import time
import numpy as np
import glob
import os
import neat
import pickle
from custom_neat_classes import EvomanPopulation


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

    env = Environment(experiment_name=experiment_name,
                        enemies=CURRENT_ENEMY_GROUP,
                        playermode="ai",
                        player_controller=PlayerControllerNEAT(n_hidden_neurons),
                        enemymode="static",
                        randomini = 'yes',
                        level=2,
                        speed="fastest",
                        savelogs="no",
                        logs="off",
                        multiplemode="yes",
                        contacthurt="player"
            )

    env.state_to_log()

    ini = time.time()

    run_mode = 'train'

    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors() + 1) * \
        n_hidden_neurons + (n_hidden_neurons + 1) * 5

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
        new_mutation_rate = initial_mutation_rate - (initial_mutation_rate - final_mutation_rate) * (current_generation / max_generations)
        new_crossover_rate = initial_crossover_rate - (initial_crossover_rate - final_crossover_rate) * (current_generation / max_generations)

        # Update mutation rates in the configuration
        config.genome_config.conn_add_prob = new_mutation_rate
        config.genome_config.conn_delete_prob = new_mutation_rate
        config.genome_config.node_add_prob = new_mutation_rate
        config.genome_config.node_delete_prob = new_mutation_rate
        config.genome_config.weight_mutate_rate = new_mutation_rate

        config.genome_config.single_structural_mutation = False  # Allow multiple structural mutations if needed
        config.genome_config.crossover_rate = new_crossover_rate
        
        print(f"Adjusted mutation rate to {new_mutation_rate:.4f} and crossover rate to {new_crossover_rate:.4f} for generation {current_generation}")



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
            net = neat.nn.FeedForwardNetwork.create(genome, config)

            [genome.fitness, genome.player_energy, genome.enemy_energy,
                genome.gain] = simulation(env, net)

            fitness_values.append(genome.fitness)
            gain_values.append(genome.gain)
        
        # Check how many genomes have valid fitness values
        valid_fitness_values = [fv for fv in fitness_values if fv is not None]

        if valid_fitness_values:
            best_fitness = max(valid_fitness_values)
            mean_fitness = np.mean(valid_fitness_values)
            std_fitness = np.std(valid_fitness_values)
            best_gain = gain_values[fitness_values.index(best_fitness)]
            # Print statistics for the current generation
            print(f"Generation {current_generation}: Max Fitness = {best_fitness:.4f}, "
                f"Average Fitness = {mean_fitness:.4f}, Std Dev = {std_fitness:.4f}")

        else:
            print(f"Warning: No valid fitness values found in generation {current_generation}.")

        current_generation += 1
        

    def get_weights_network(winner):
        weights = []

        for connection_key, connection in winner.connections.items():
            weight = connection.weight

            weights.append(weight)

        return weights

    def run(config_file: str):
        global current_generation
        current_generation = 0  # Initialization of current_generation
        species_set = neat.DefaultSpeciesSet
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            species_set, neat.DefaultStagnation,
                            config_file)

        p = CustomPopulation(config)
        winner, best = p.run(eval_genomes, num_gens)
        
        best_file_name = f'best_individual_run{i_run}'

        with open(os.path.join(experiment_name, best_file_name), 'wb') as file_out:
            pickle.dump(best, file_out)

        fim = time.time()  # prints total execution time for experiment
        print('\nExecution time: ' + str(round((fim - ini) / 60)) + ' minutes \n')
        print('\nExecution time: ' + str(round((fim - ini))) + ' seconds \n')
        
        env.state_to_log()

    if __name__ == '__main__':
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config_generalist_NEAT')
    

        run(config_path)

        fim = time.time()
        print('\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
        print('\nExecution time: '+str(round((fim-ini)))+' seconds \n')

        file = open(experiment_name+'/neuroended', 'w')
        file.close()

        env.state_to_log()
