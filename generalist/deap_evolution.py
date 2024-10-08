import sys
import time
import numpy as np
import os
from deap import base, creator, tools, algorithms
import random
import csv
import optuna

from evoman.environment import Environment
from demo_controller import player_controller
from enemy_groups import enemy_group_to_str


# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

N_HIDDEN_NEURONS = 10


class DeapRunner:
    def __init__(self, enemies: list, num_generations: int, run_idx: int,
                 output_base_folder: str = "DEAPexperimentE"):
        self.enemies = enemies
        self.run_idx = run_idx
        self.num_generations = num_generations
        self.output_base_folder = output_base_folder

        # Params to be set using `set_params`
        self.cxpb = None
        self.mutpb = None
        self.mu = None
        self.lambda_ = None

        self.env, self.n_vars = self._create_environment()
        self._init_deap()
        self.toolbox = self._create_toolbox()

        # Results
        self.final_pop = None
        self.hof = None
        self.logbook = None

    def set_params(self, cxpb: float, mutpb: float, mu: float, lambda_: float):
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.mu = mu
        self.lambda_ = lambda_

    def _init_deap(self):
        # DEAP setup for evolutionary algorithm
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = self._create_toolbox()
        return toolbox

    def _create_environment(self):
        experiment_name = self.get_run_folder()

        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)

        # initializes simulation in individual evolution mode, for single static enemy.
        env = Environment(experiment_name=experiment_name,
                          enemies=self.enemies,
                          playermode="ai",
                          player_controller=player_controller(
                              N_HIDDEN_NEURONS),
                          enemymode="static",
                          level=2,
                          speed="fastest",
                          visuals=False,
                          multiplemode="yes")

        n_vars = (env.get_num_sensors() + 1) * \
            N_HIDDEN_NEURONS + (N_HIDDEN_NEURONS + 1) * 5

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
        toolbox.register("evaluate", self._evaluate)

        return toolbox

    # Evaluation function for DEAP
    def _evaluate(self, individual):
        return (self._simulation(individual),)

    # same as the demo file
    def _simulation(self, x):
        f, p, e, t = self.env.play(pcont=np.array(x))
        return f

    def get_run_folder(self):
        str_enemy_group = enemy_group_to_str(self.enemies)
        return f'{self.output_base_folder}{str_enemy_group}/DEAP_run{self.run_idx}'

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
        # Extract the relevant keys (such as 'gen', 'nevals', 'avg', 'std', 'min', 'max') from the logbook
        # Use the keys from the first logbook entry (assuming all entries have the same keys)
        if self.logbook is None:
            return

        keys = self.logbook[0].keys()

        file = os.path.join(self.get_run_folder(), filename)

        with open(file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            for record in self.logbook:
                writer.writerow(record)


# def objective(trial):
#     """
#     Optuna Objective Function
#     """
#     # Suggest hyperparameters
#     cxpb = trial.suggest_float('cxpb', 0.0, 0.9)
#     mutpb = trial.suggest_float('mutpb', 0.0, 1.0 - cxpb)
#     mu = trial.suggest_int('mu', 50, 100)
#     lambda_ = trial.suggest_int('lambda_', 100, 200)

#     # Run the DEAP evolutionary algorithm
#     final_pop, hof, _ = run_evolutionary_algorithm(
#         cxpb, mutpb, mu, lambda_)

#     # Return the fitness of the best individual
#     return hof[0].fitness.values[0]


# if __name__ == '__main__':
#     run_mode = 'train'  # or 'test'

#     if run_mode == 'test':
#         bsol = np.loadtxt(experiment_name + '/best.txt')
#         print('\nRUNNING SAVED BEST SOLUTION\n')
#         env.update_parameter('speed', 'normal')
#         evaluate([bsol])
#         sys.exit(0)

#     # Record start time before Optuna study begins
#     start_time = time.time()

#     # Start the Optuna study for hyperparameter tuning
#     study = optuna.create_study(direction='maximize')
#     # Run 26 trials of hyperparameter optimization
#     study.optimize(objective, n_trials=26)

#     # Output the best parameters
#     best_params = study.best_params
#     print(f"Best Parameters: {best_params}")

#     # Once best parameters are found, you can run the evolutionary algorithm again with the best parameters:
#     final_pop, hof, logbook = run_evolutionary_algorithm(
#         best_params['cxpb'], best_params['mutpb'], best_params['mu'], best_params['lambda_'])

#     # Save the best individual
#     np.savetxt(experiment_name + '/best.txt', hof[0])

#     # Save logbook results
#     save_logbook(logbook, experiment_name + '/logbook.csv')

#     # Print execution time
#     end_time = time.time()
#     print('\nExecution time: ' +
    #       str(round((end_time - start_time) / 60)) + ' minutes \n')
    # print('\nExecution time: ' +
    #       str(round((end_time - start_time))) + ' seconds \n')

    # # Save control (simulation has ended) file for bash loop
    # with open(experiment_name + '/neuroended', 'w') as file:
    #     file.write('')

    # env.state_to_log()  # checks environment state
