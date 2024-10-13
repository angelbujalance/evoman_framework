import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from evoman.environment import Environment
from demo_controller import player_controller
from constants import (ENEMY_GROUP_1, ENEMY_GROUP_2, enemy_folder,
                       PATH_DEAP, OUTPUT_FOLDER_TRAINING,
                       NUM_RUNS)

N_GAMES = 5  # Specify the number of games to play


def read_results(experiment_dir, num_runs):
    all_best_fitness = []

    for i in range(num_runs):
        results_path = os.path.join(
            experiment_dir, f'run_{i}', 'logbook.csv')
        try:
            data = pd.read_csv(results_path)
            best_fitness = data['max'].tolist()

            # Collect data for all generations
            all_best_fitness.append(best_fitness)
        except Exception as e:
            print(f"Error reading {results_path}: {e}")

    # Find the best run by max fitness across all generations and runs
    all_best_fitness = np.array(all_best_fitness)
    max_fitness_value = np.max(all_best_fitness)
    best_run_idx, best_generation_idx = np.unravel_index(
        np.argmax(all_best_fitness), all_best_fitness.shape)

    print(
        f"Highest Best Fitness: {max_fitness_value}, "
        f"found in run {best_run_idx}, "
        f"generation {best_generation_idx}")

    return best_run_idx, best_generation_idx, max_fitness_value


def get_best_individual_weights(experiment_dir, best_run_idx):
    best_weights_path = os.path.join(
        experiment_dir, f'run_{best_run_idx}', 'best.txt')

    try:
        best_weights = np.loadtxt(best_weights_path)

        print(f"Best individual weights from run {best_run_idx}:")
        print(best_weights)

        return best_weights
    except Exception as e:
        print(f"Error reading {best_weights_path}: {e}")
        return None


def init_env(experiment_dir, enemies: list):
    n_hidden_neurons = 10
    env = Environment(experiment_name=experiment_dir,
                      enemies=enemies,
                      multiplemode="yes" if len(enemies) > 1 else "no",
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False,
                      randomini="yes"
                      )

    return env


# Function to simulate a game and return the gain
def play_games(env, best_weights, n_games):

    gains = []
    for i in range(n_games):

        f, p, e, t = env.play(pcont=best_weights)

        gain = p - e  # Player's life minus enemy's life

        gains.append(gain)

        print(f"Game {i+1}: Gain = {gain}")

    return gains


# Function to generate a boxplot comparing two approaches for individual gains
def plot_gain_boxplot(gains_approach_1, gains_approach_2, enemy_number):
    """
    Plots a boxplot for individual gains for two approaches against a
    specific enemy.
     gains_approach_1: List or array of individual gains for Approach 1
     gains_approach_2: List or array of individual gains for Approach 2
     enemy_number: The enemy number for the plot title
    """

    # Combine the gains from both approaches
    data = [gains_approach_1, gains_approach_2]
    labels = ["Approach 1", "Approach 2"]

    # Set up the plot
    sns.set(style="whitegrid")
    fig, ax = plt.subplots()

    # Plot the gains
    # Blue for Approach 1, Red for Approach 2
    sns.boxplot(data=data, ax=ax, palette=["#3498db", "#e74c3c"])
    sns.swarmplot(data=data, ax=ax, color="black", size=5)
    # Set plot labels and title
    ax.set_ylabel("Individual gain")
    ax.set_xticklabels(labels)

    # Create a title with T-statistic and p-value if provided
    title = f"Enemies {enemy_number}"
    ax.set_title(title)

    plt.show()


if __name__ == '__main__':
    # for enemy in [1, 2, 3, 4, 5, 6, 7, 8]:
    for enemy_group in [ENEMY_GROUP_1, ENEMY_GROUP_2]:
        # Directory containing all run folders
        experiment_dir = os.path.join(PATH_DEAP, OUTPUT_FOLDER_TRAINING,
                                      enemy_folder(enemy_group))

        # Find the best run and generation
        best_run_idx, best_generation_idx, max_fitness_value = read_results(
            experiment_dir, NUM_RUNS)

        # Retrieve the weights of the best individual from the best run
        best_weights = get_best_individual_weights(
            experiment_dir, best_run_idx)

        # Replace with actual gains for Approach 1
        gains_approach_1 = np.array([])

        if best_weights is None:
            continue

        # Initialize the environment
        env = init_env(experiment_dir, enemy_group)

        gains = play_games(env, best_weights, N_GAMES)

        print(f"All gains over {N_GAMES} games: {gains}")

        # Plot the boxplot for the gains
        plot_gain_boxplot(gains_approach_1, gains, enemy_number=enemy_group)

        # save the plot
        plt.savefig('gain_boxplot.png')
