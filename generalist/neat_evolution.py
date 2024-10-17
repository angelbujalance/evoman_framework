# import time
import numpy as np
import os
import neat
import pickle
import pandas as pd

from evoman.environment import Environment
from neat_controller import PlayerControllerNEAT
from neat_population import CustomPopulation
from constants import enemy_folder, PATH_NEAT


# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


class NeatRunner:
    def __init__(self, train_enemies: list, num_generations: int,
                 model_folder: str = "trained",
                 results_folder: str = "tested",
                 test_enemies: list = None,
                 config_file: str = "config_generalist_NEAT",
                 n_hidden_neurons: int = 10,
                 genome_type: type[neat.DefaultGenome] = neat.DefaultGenome,
                 reproduction_type=neat.DefaultReproduction,
                 species_set_type=neat.DefaultSpeciesSet,
                 stagnation_type=neat.DefaultStagnation,
                 use_adjusted_mutation_rate: bool = True,
                 initial_mutation_rate=0.5,
                 final_mutation_rate=0.1,
                 initial_crossover_rate=0.7,
                 final_crossover_rate=.3
                 ):
        self.train_enemies = train_enemies
        self.test_enemies = test_enemies
        self.num_generations = num_generations
        self.model_base_folder = os.path.join(PATH_NEAT, model_folder)
        self.results_base_folder = os.path.join(PATH_NEAT, results_folder)

        self.config_file = config_file
        self.genome_type = genome_type
        self.reproduction_type = reproduction_type
        self.species_set_type = species_set_type
        self.stagnation_type = stagnation_type

        self.use_adjusted_mutation_rate = use_adjusted_mutation_rate

        self.config = self._create_config()
        self.n_hidden_neurons = n_hidden_neurons
        self.config.genome_config.num_hidden = n_hidden_neurons

        self.current_generation = 0
        self.run_idx = None

        self.logger = {}  # gen_id: [max_fitness, mean_fitness, std_fitness]

        self.initial_mutation_rate = initial_mutation_rate
        self.final_mutation_rate = final_mutation_rate
        self.initial_crossover_rate = initial_crossover_rate
        self.final_crossover_rate = final_crossover_rate

    @property
    def is_training(self):
        return self.test_enemies is None

    def _create_environment(self):
        experiment_name = self.get_input_folder()

        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name, exist_ok=True)

        # initializes simulation in individual evolution mode,
        # for single static enemy.
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
        self.env, self.n_vars = self._create_environment()
        os.makedirs(os.path.dirname(file), exist_ok=True)

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
            print(f"Generation {self.current_generation}: " +
                  f"Max Fitness = {best_fitness:.4f}, " +
                  f"Average Fitness = {mean_fitness:.4f}, " +
                  f"Std Dev = {std_fitness:.4f}")

        else:
            print(
                "Warning: No valid fitness values found in generation " +
                f"{self.current_generation}.")

        self.current_generation += 1

    def _simulation(self, pcont: neat.nn.FeedForwardNetwork):
        return self.run_game(pcont)

    def run_game(self, pcont: neat.nn.FeedForwardNetwork):
        f, p, e, t = self.env.play(pcont=pcont)
        gain = p - e
        return f, p, e, gain

    def get_input_folder(self):
        enemies = self.train_enemies
        return self._construct_path(self.model_base_folder, enemies)

    def get_output_folder(self):
        enemies = self.get_run_enemies()
        base_folder = (self.model_base_folder
                       if self.is_training
                       else self.results_base_folder)
        return self._construct_path(base_folder, enemies)

    def _construct_path(self, base_folder, enemy_group):
        return os.path.join(f'{base_folder}',
                            enemy_folder(enemy_group),
                            f'run_{self.run_idx}')

    def get_run_enemies(self):
        return self.train_enemies if self.is_training else self.test_enemies

    def _update_mutation_rate(self, new_mutation_rate: float):
        self.config.genome_config.conn_add_prob = new_mutation_rate
        self.config.genome_config.conn_delete_prob = new_mutation_rate
        self.config.genome_config.node_add_prob = new_mutation_rate
        self.config.genome_config.node_delete_prob = new_mutation_rate
        self.config.genome_config.weight_mutate_rate = new_mutation_rate

    def adjust_mutation_and_crossover_rates(self):
        if not self.use_adjusted_mutation_rate:
            return

        # TODO: we can change these later to what we'd like
        initial_mutation_rate = 1.0
        final_mutation_rate = 0.0
        initial_crossover_rate = 0.0
        final_crossover_rate = 1.0

        # Linear decrease of mutation rate over generations
        new_mutation_rate = initial_mutation_rate - \
            (initial_mutation_rate - final_mutation_rate) * \
            (self.current_generation / self.num_generations)
        new_crossover_rate = initial_crossover_rate - \
            (initial_crossover_rate - final_crossover_rate) * \
            (self.current_generation / self.num_generations)

        # Update mutation rates in the configuration
        #self._update_mutation_rate(new_mutation_rate)

        # Allow multiple structural mutations if needed
        self.config.genome_config.single_structural_mutation = False

        self.config.genome_config.weight_mutate_rate = new_mutation_rate
        self.config.genome_config.bias_mutate_rate = new_mutation_rate
        self.config.genome_config.activation_mutate_rate  = new_mutation_rate
        self.config.genome_config.response_mutate_rate = new_mutation_rate
        self.config.reproduction_config.survival_threshold = new_crossover_rate

        print(
            f"Adjusted mutation rate to {new_mutation_rate:.4f} and " +
            f"crossover rate to {new_crossover_rate:.4f} " +
            f"for generation {self.current_generation}")

    def run_evolutionary_algorithm(self, run_idx):
        self.run_idx = run_idx
        # ini = time.time()
        self.env, self.n_vars = self._create_environment()

        experiment_name = self.get_input_folder()
        self.results_file = os.path.join(experiment_name, "results.csv")

        p = CustomPopulation(self.config)

        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.StdOutReporter(True))

        winner, best = p.run(self.eval_genomes, self.num_generations)

        best_file_name = f'best_individual_run{run_idx}'

        file = os.path.join(experiment_name, best_file_name)

        print(file, 'file')

        with open(file, 'wb') as f:
            pickle.dump(best, f)

        # fim = time.time()  # prints total execution time for experiment
        # print('\nExecution time: ' + str(round((fim - ini) / 60)) +
        #       ' minutes \n')
        # print('\nExecution time: ' + str(round((fim - ini))) + ' seconds \n')

        self.env.state_to_log()

        best_weights = self.get_weights_genome(winner)
        np.savetxt(os.path.join(experiment_name, "best.txt"), best_weights)

        generations = list(range(len(stats.most_fit_genomes)))
        best_fitnesses = [stats.most_fit_genomes[i].fitness
                          for i in range(len(stats.most_fit_genomes))]
        mean_fitnesses = stats.get_fitness_mean()
        std_fitnesses = stats.get_fitness_stdev()

        df = pd.DataFrame({
            'Generation': generations,
            'Best Fitness': best_fitnesses,
            'Mean Fitness': mean_fitnesses,
            'Standard Deviation': std_fitnesses
        })

        df.to_csv(self.results_file, index=False)

        file = open(os.path.join(experiment_name, 'neuroended'), 'w')
        file.close()
        return winner

    def get_results(self):
        raise NotImplementedError()

    def get_weights_genome(self, genome: type[neat.DefaultGenome]):
        weights = []

        for connection_key, connection in genome.connections.items():
            weight = connection.weight

            weights.append(weight)

        return weights

    def set_params(self, n_hidden_neurons=None, bias_mutate_rate=None,
                   pop_size=None, elitism=None, response_mutate_rate=None,
                   weight_mutate_rate=None
                   ):
        """
        Override the values from the given config file by these values.
        """

        if n_hidden_neurons is not None:
            self.config.genome_config.num_hidden = n_hidden_neurons

        if bias_mutate_rate is not None:
            self.config.genome_config.bias_mutate_rate = bias_mutate_rate

        if response_mutate_rate is not None:
            self.config.genome_config.response_mutate_rate = response_mutate_rate

        if weight_mutate_rate is not None:
            self.config.genome_config.weight_mutate_rate = weight_mutate_rate

        if pop_size is not None:
            self.config.pop_size = pop_size

        if elitism is not None:
            self.config.reproduction_config.elitism = elitism