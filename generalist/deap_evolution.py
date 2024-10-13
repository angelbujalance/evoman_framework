import numpy as np
import os
from deap import base, creator, tools, algorithms
import random
import csv

from constants import PATH_DEAP
from evoman.environment import Environment
from demo_controller import player_controller
from enemy_groups import enemy_group_to_str


# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


class DeapRunner:
    def __init__(self, train_enemies: list, num_generations: int, run_idx: int,
                 training_base_folder: str = "trained",
                 testing_base_folder: str = "tested",
                 test_enemies: list = None, n_hidden_neurons=10):
        self.train_enemies = train_enemies
        self.test_enemies = test_enemies
        self.run_idx = run_idx
        self.num_generations = num_generations
        self.n_hidden_neurons = n_hidden_neurons
        self.training_base_folder = os.path.join(PATH_DEAP,
                                                 training_base_folder)
        self.testing_base_folder = os.path.join(PATH_DEAP, testing_base_folder)

        # Params to be set using `set_params`
        self.cxpb = None
        self.mutpb = None
        self.mu = None
        self.lambda_ = None

        self.env, self.n_vars = self._create_environment()

        if self.is_training:
            # Only activate deap in training mode
            self._init_deap_training()
            self.toolbox = self._create_toolbox()

        # Results
        self.final_pop = None
        self.hof = None
        self.logbook = None

    @property
    def is_training(self):
        return self.test_enemies is None

    def set_params(self, cxpb: float, mutpb: float, mu: float, lambda_: float):
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.mu = mu
        self.lambda_ = lambda_

    def _init_deap_training(self):
        # DEAP setup for evolutionary algorithm
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

    def _create_environment(self):
        experiment_name = self.get_input_folder()

        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)

        # initializes simulation in individual evolution mode,
        # for single static enemy.
        enemies = self.get_run_enemies()
        multiplemode = "yes" if len(enemies) > 1 else "no"
        env = Environment(experiment_name=experiment_name,
                          enemies=enemies,
                          playermode="ai",
                          player_controller=player_controller(
                              self.n_hidden_neurons),
                          enemymode="static",
                          level=2,
                          speed="fastest",
                          visuals=False,
                          multiplemode=multiplemode)

        n_vars = (env.get_num_sensors() + 1) * \
            self.n_hidden_neurons + (self.n_hidden_neurons + 1) * 5

        env.state_to_log()  # checks environment state
        return env, n_vars

    def _create_toolbox(self):
        toolbox = base.Toolbox()

        toolbox.register("attr_float", random.uniform, -1, 1)
        toolbox.register("individual", tools.initRepeat,
                         creator.Individual, toolbox.attr_float,
                         n=self.n_vars)
        toolbox.register("population", tools.initRepeat,
                         list, toolbox.individual)

        # Genetic operators
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", self.evaluate)

        return toolbox

    # Evaluation function for DEAP
    def evaluate(self, individual):
        return (self._simulation(individual),)

    def _simulation(self, pcont):
        f, p, e, t = self.run_game(pcont)
        return f

    def run_game(self, pcont):
        f, p, e, t = self.env.play(pcont=np.array(pcont))
        return f, p, e, t

    def get_input_folder(self):
        enemies = self.train_enemies
        return self._construct_path(self.training_base_folder, enemies)

    def get_output_folder(self):
        enemies = self.get_run_enemies()
        base_folder = (self.training_base_folder
                       if self.is_training
                       else self.testing_base_folder)
        return self._construct_path(base_folder, enemies)

    def _construct_path(self, base_folder, enemy_group):
        str_enemy_group = enemy_group_to_str(enemy_group)
        return os.path.join(f'{base_folder}',
                            f'enemies_{str_enemy_group}',
                            f'run_{self.run_idx}')

    def get_run_enemies(self):
        return self.train_enemies if self.is_training else self.test_enemies

    # Initializes the population
    def run_evolutionary_algorithm(self):
        npop = self.mu  # Use mu as the population size

        pop = self.toolbox.population(n=npop)

        # Configure statistics and logbook
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Hall of Fame to keep track of the best individual
        hof = tools.HallOfFame(1)

        # Run the DEAP evolutionary algorithm (Mu, Lambda)
        final_pop, logbook = algorithms.eaMuCommaLambda(
            pop, self.toolbox, self.mu, self.lambda_, self.cxpb, self.mutpb,
            self.num_generations, stats=stats, halloffame=hof, verbose=True)

        self.final_pop = final_pop
        self.hof = hof
        self.logbook = logbook
        return final_pop, hof, logbook

    def get_results(self):
        return self.final_pop, self.hof, self.logbook

    # Save logbook results to a CSV file
    def save_logbook(self, filename="logbook.csv"):
        # Extract the relevant keys (such as 'gen', 'nevals', 'avg', 'std',
        # 'min', 'max') from the logbook
        # Use the keys from the first logbook entry (assuming all entries
        # have the same keys)
        if self.logbook is None:
            return

        keys = self.logbook[0].keys()

        file = os.path.join(self.get_output_folder(), filename)

        with open(file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            for record in self.logbook:
                writer.writerow(record)
