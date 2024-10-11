from evoman.environment import Environment
from neat_controller import PlayerControllerNEAT
from neat_population import CustomPopulation
from enemy_groups import enemy_group_to_str

import time
import numpy as np
import os
import neat
import pickle


# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


class NeatRunner:
    def __init__(self, train_enemies: list, num_generations: int, run_idx: int,
                 training_base_folder: str = "trained",
                 testing_base_folder: str = "tested",
                 test_enemies: list = None,
                 config_file: str = "config_generalist_NEAT",
                 n_hidden_neurons: int = 10,
                 genome_type: type[neat.DefaultGenome] = neat.DefaultGenome,
                 reproduction_type: type[neat.DefaultReproduction] = neat.DefaultReproduction,
                 species_set_type: type[neat.DefaultSpeciesSet] = neat.DefaultSpeciesSet,
                 stagnation_type: type[neat.DefaultStagnation] = neat.DefaultStagnation,
                 use_adjusted_mutation_rate: bool = False
                 ):
        self.train_enemies = train_enemies
        self.test_enemies = test_enemies
        self.run_idx = run_idx
        self.num_generations = num_generations
        self.training_base_folder = os.path.join("results NEAT",
                                                 training_base_folder)
        self.testing_base_folder = os.path.join("results NEAT",
                                                testing_base_folder)

        self.config_file = config_file
        self.genome_type = genome_type
        self.reproduction_type = reproduction_type
        self.species_set_type = species_set_type
        self.stagnation_type = stagnation_type

        self.use_adjusted_mutation_rate = use_adjusted_mutation_rate

        self.config = self._create_config()
        self.n_hidden_neurons = n_hidden_neurons
        self.config.genome_config.num_hidden = n_hidden_neurons
        self.env, self.n_vars = self._create_environment()

        if self.is_training:
            experiment_name = self.get_input_folder()
            self.results_file = os.path.join(experiment_name, "results.txt")

            with open(self.results_file, 'w') as f:
                f.write("best_fitness,mean_fitness,std_fitness,gain\n")

        self.current_generation = 0

    @property
    def is_training(self):
        return self.test_enemies is None

    def _create_environment(self):
        experiment_name = self.get_input_folder()

        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name, exist_ok=True)

        # initializes simulation in individual evolution mode, for single static enemy.
        enemies = self.get_run_enemies()
        multiplemode = "yes" if len(enemies) > 1 else "no"

        env = Environment(experiment_name=experiment_name,
                          enemies=enemies,
                          playermode="ai",
                          player_controller=PlayerControllerNEAT(
                              self.n_hidden_neurons),
                          enemymode="static",
                          randomini='yes',
                          level=2,
                          speed="fastest",
                          savelogs="no",
                          logs="off",
                          multiplemode=multiplemode,
                          contacthurt="player"
                          )

        n_vars = (env.get_num_sensors() + 1) * \
            self.n_hidden_neurons + (self.n_hidden_neurons + 1) * 5

        env.state_to_log()  # checks environment state
        return env, n_vars

    def _create_config(self):
        return neat.Config(self.genome_type, self.reproduction_type,
                           self.species_set_type, self.stagnation_type,
                           self.config_file)

    def evaluate_from_genome_file(self, file):
        with open(file, "rb") as f:
            genome = pickle.load(f)

        net = neat.nn.FeedForwardNetwork.create(genome, self.config)
        f, p, e, gain = self.run_game(net)
        return f, p, e, gain

    def eval_genomes(self, genomes, config) -> None:
        self.adjust_mutation_and_crossover_rates()

        fitness_values = []
        gain_values = []

        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)

            [genome.fitness, genome.player_energy, genome.enemy_energy,
                genome.gain] = self._simulation(net)

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
            print(f"Generation {self.current_generation}: Max Fitness = {best_fitness:.4f}, "
                  f"Average Fitness = {mean_fitness:.4f}, Std Dev = {std_fitness:.4f}")

            with open(self.results_file, 'a') as f:
                f.write(
                    f"{best_fitness},{mean_fitness},{std_fitness},{best_gain}\n")

        else:
            print(
                f"Warning: No valid fitness values found in generation {self.current_generation}.")

        self.current_generation += 1

    def _simulation(self, pcont: neat.nn.FeedForwardNetwork):
        return self.run_game(pcont)

    def run_game(self, pcont: neat.nn.FeedForwardNetwork):
        f, p, e, t = self.env.play(pcont=pcont)
        gain = p - e
        return f, p, e, gain

    def get_input_folder(self):
        enemies = self.train_enemies
        return self._construct_path(self.training_base_folder, enemies)

    def get_output_folder(self):
        enemies = self.get_run_enemies()
        base_folder = self.training_base_folder if self.is_training else self.testing_base_folder
        return self._construct_path(base_folder, enemies)

    def _construct_path(self, base_folder, enemy_group):
        str_enemy_group = enemy_group_to_str(enemy_group)
        return os.path.join(f'{base_folder}',
                            f'enemies_{str_enemy_group}',
                            f'run_{self.run_idx}')

    def get_run_enemies(self):
        return self.train_enemies if self.is_training else self.test_enemies

    def adjust_mutation_and_crossover_rates(self):
        if not self.use_adjusted_mutation_rate:
            return

        # TODO: we can change these later to what we'd like
        initial_mutation_rate = 0.5
        final_mutation_rate = 0.1
        initial_crossover_rate = 0.7
        final_crossover_rate = 0.3

        # Linear decrease of mutation rate over generations
        new_mutation_rate = initial_mutation_rate - \
            (initial_mutation_rate - final_mutation_rate) * \
            (self.current_generation / self.num_generations)
        new_crossover_rate = initial_crossover_rate - \
            (initial_crossover_rate - final_crossover_rate) * \
            (self.current_generation / self.num_generations)

        # Update mutation rates in the configuration
        self.config.genome_config.conn_add_prob = new_mutation_rate
        self.config.genome_config.conn_delete_prob = new_mutation_rate
        self.config.genome_config.node_add_prob = new_mutation_rate
        self.config.genome_config.node_delete_prob = new_mutation_rate
        self.config.genome_config.weight_mutate_rate = new_mutation_rate

        # Allow multiple structural mutations if needed
        self.config.genome_config.single_structural_mutation = False
        self.config.genome_config.crossover_rate = new_crossover_rate

        print(
            f"Adjusted mutation rate to {new_mutation_rate:.4f} and crossover rate to {new_crossover_rate:.4f} for generation {self.current_generation}")

    def run_evolutionary_algorithm(self):
        ini = time.time()
        p = CustomPopulation(self.config)
        winner, best = p.run(self.eval_genomes, self.num_generations)

        best_file_name = f'best_individual_run{self.run_idx}'
        experiment_name = self.get_input_folder()

        file = os.path.join(experiment_name, best_file_name)

        with open(file, 'wb') as f:
            pickle.dump(best, f)

        fim = time.time()  # prints total execution time for experiment
        # print('\nExecution time: ' + str(round((fim - ini) / 60)) + ' minutes \n')
        # print('\nExecution time: ' + str(round((fim - ini))) + ' seconds \n')

        self.env.state_to_log()

        best_weights = self.get_weights_genome(winner)
        np.savetxt(os.path.join(experiment_name, "best.txt"), best_weights)

        file = open(os.path.join(experiment_name, 'neuroended'), 'w')
        file.close()

    def get_results(self):
        raise NotImplementedError()

    def get_weights_genome(self, genome: type[neat.DefaultGenome]):
        weights = []

        for connection_key, connection in genome.connections.items():
            weight = connection.weight

            weights.append(weight)

        return weights
