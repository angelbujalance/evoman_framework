import numpy as np
import os
from deap import base, creator, tools, algorithms, cma
from deap.benchmarks.tools import hypervolume
import random
import csv

from constants import (enemy_folder, PATH_DEAP,
                       OUTPUT_FOLDER_TRAINING, OUTPUT_FOLDER_TESTING, USE_CMA)
from evoman.environment import Environment
from demo_controller import player_controller



# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


class DeapRunner:
    def __init__(self, train_enemies: list, num_generations: int,
                 test_enemies: list = None, n_hidden_neurons=10,
                 model_folder: str = OUTPUT_FOLDER_TRAINING,
                 results_folder: str = OUTPUT_FOLDER_TESTING,
                 use_cma: bool = USE_CMA):
        self.train_enemies = train_enemies
        self.test_enemies = test_enemies
        self.num_generations = num_generations
        self.n_hidden_neurons = n_hidden_neurons
        self.model_base_folder = os.path.join(PATH_DEAP, model_folder)
        self.results_base_folder = os.path.join(PATH_DEAP, results_folder)

        # Params to be set using `set_params`
        self.cxpb = None
        self.mutpb = None
        self.mu = None
        self.lambda_ = None
        self.use_cma = use_cma
        print(f"\n\n\nUSE_CMA: {use_cma}\n\n\n")

        # Results
        self.final_pop = None
        self.hof = None
        self.logbook = None

        self.run_idx = None
        self.env = None

    @property
    def is_training(self):
        return self.test_enemies is None

    def set_params(self, cxpb: float = None, mutpb: float = None, mu: float = None, lambda_: float = None, 
                        use_cma: bool = False, sigma: float = 1.0):
        # Set shared parameters
        self.mu = mu
        self.lambda_ = lambda_
        self.use_cma = use_cma

        if use_cma:
            # Only relevant for CMA-ES
            self.sigma = sigma
        else:
            # Only relevant for MuCommaLambda
            self.cxpb = cxpb
            self.mutpb = mutpb


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
        toolbox.register("evaluate", self.evaluate)

        if self.use_cma:
            print("\n\n\nUsing Single-Objective CMA-ES\n\n\n")
            strategy = cma.Strategy(
                centroid=[0.0] * self.n_vars,  # Initial centroid (can be set to zeros or other)
                sigma=self.sigma,  # Step size (standard deviation)
                lambda_=self.lambda_,  # Number of offspring
                mu=self.mu  # Number of parents
            )
            
            toolbox.register("generate", strategy.generate, creator.Individual)
            toolbox.register("update", strategy.update)
            self.strategy = strategy
            return toolbox

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

        return toolbox

    # Evaluation function for DEAP
    def evaluate(self, individual):
        return (self._simulation(individual),)

    def _simulation(self, pcont):
        f, p, e, t = self.run_game(pcont)
        return f

    def run_game(self, pcont):
        if self.env is None:
            self.env, self.n_vars = self._create_environment()

        f, p, e, t = self.env.play(pcont=np.array(pcont))
        return f, p, e, t

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

    # Initializes the population
    def run_evolutionary_algorithm(self, run_idx):
        self.run_idx = run_idx
        self.env, self.n_vars = self._create_environment()

        # Only activate deap in training mode
        self._init_deap_training()
        self.toolbox = self._create_toolbox()

        npop = self.mu  # Use mu as the population size

        # Create population based on algorithm type
        if self.use_cma:
            pop = self.toolbox.generate()
        else:
            pop = self.toolbox.population(n=npop)

        # Configure statistics and logbook
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Hall of Fame to keep track of the best individual
        hof = tools.HallOfFame(1)

        print(f"\n\n\nRUN_EVOLUTIONARY_ALGORITHM: USE_CMA: {self.use_cma}\n\n\n")
        if self.use_cma:
            final_pop, logbook = algorithms.eaGenerateUpdate(
            self.toolbox, ngen=self.num_generations, stats=stats,
            halloffame=hof, verbose=True)
        else:
            print("\n\n\nRUN_EVOLUTIONARY_ALGORITHM: WRONG BLOCK\n\n\n")
            # Run the DEAP evolutionary algorithm (Mu, Lambda)
            final_pop, logbook = algorithms.eaMuCommaLambda(
                pop, self.toolbox, self.mu, self.lambda_, self.cxpb, self.mutpb,
                self.num_generations, stats=stats, halloffame=hof, verbose=True)

        self.final_pop = final_pop
        self.hof = hof
        self.logbook = logbook

        experiment_name = self.get_input_folder()

        best_weights = hof[0]
        np.savetxt(os.path.join(experiment_name, "best.txt"), best_weights)

        file = open(os.path.join(experiment_name, 'neuroended'), 'w')
        file.close()
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
