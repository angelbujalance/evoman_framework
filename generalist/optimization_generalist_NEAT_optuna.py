import optuna
# Tree-structured Parzen Estimator (TPE) sampler for better search efficiency
from optuna.samplers import TPESampler

# Import existing modules
from evoman.environment import Environment
from neat_controller import PlayerControllerNEAT
import time
import numpy as np
import os
import neat
from neat_population import Population

from constants import ENEMY_GROUP_1, ENEMY_GROUP_2

CURRENT_ENEMY_GROUP = ENEMY_GROUP_1

# Fine-tune to obtain the best hyperparameter setting

# Define Optuna objective function


def objective(trial):
    """
    Objective function to optimize using Optuna.
    This function runs the NEAT algorithm with a set of hyperparameters and returns the best fitness.
    """
    # Hyperparameters to optimize
    n_hidden_neurons = trial.suggest_int(
        'num_hidden', 5, 20)  # Number of hidden neurons
    mutation_rate = trial.suggest_float(
        'mutation_rate', 0.01, 0.5)  # Mutation rate
    pop_size = trial.suggest_int('pop_size', 50, 200)  # Population size
    elitism = trial.suggest_int('elitism', 1, 30)  # Number of elitism
    generations = trial.suggest_int('generations', 5, 30)  # Number of elitism

    experiment_name = f'optuna/NEAT_experiment_optuna_trial_{trial.number}'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # Initialize environment
    env = Environment(experiment_name=experiment_name,
                      enemies=CURRENT_ENEMY_GROUP,
                      playermode="ai",
                      player_controller=PlayerControllerNEAT(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False,
                      multiplemode="yes")

    # Load configuration and set hyperparameters
    config_path = os.path.join(os.path.dirname(__file__),
                               'config_generalist_NEAT')
    checkpoint_path = os.path.join(experiment_name, 'checkpoints')

    def eval_genomes(genomes, config):
        fitness_values = []
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            [genome.fitness, genome.player_energy, genome.enemy_energy,
                genome.individual_gain] = simulation(env, net)
            fitness_values.append(genome.fitness)

        best_fitness = max(fitness_values)
        mean_fitness = np.mean(fitness_values)
        std_fitness = np.std(fitness_values)
        return best_fitness, mean_fitness, std_fitness

    def simulation(env, x):
        f, p, e, t = env.play(pcont=x)
        return f, p, e, p - e

    def run(config_file, checkpoint_folder):
        # Load NEAT configuration
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)

        p = neat.Population(config)

        # Modify NEAT parameters using Optuna-suggested values
        p.population_size = pop_size
        p.mutation_rate = mutation_rate

        # Run NEAT for the specified number of generations
        winner = p.run(eval_genomes, generations)

        # Run the simulation with the winning genome
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        [winner.fitness, winner.player_energy, winner.enemy_energy,
            winner.individual_gain] = simulation(env, winner_net)

        return winner.fitness

    # Run NEAT with current trial configuration
    best_fitness = run(config_path, checkpoint_path)

    # Return best fitness as the trial result
    return best_fitness


if __name__ == '__main__':
    # Create an Optuna study object to optimize the objective function
    study = optuna.create_study(direction="maximize", sampler=TPESampler())

    # Run the optimization, performing up to 50 trials
    study.optimize(objective, n_trials=50)

    # Print best trial results
    best_trial = study.best_trial
    print(f"Best trial: {best_trial.number}")
    print(f"Best fitness: {best_trial.value}")
    print(f"Best hyperparameters: {best_trial.params}")

    # Save the Optuna study results for further analysis
    study.trials_dataframe().to_csv('optuna_study_results.csv')
