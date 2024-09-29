import numpy as np
from scipy import stats
from best_individual_runs_DEAP import play_games, get_best_individual_weights, read_results

from evoman.environment import Environment
from demo_controller import player_controller

enemies = [2, 7, 8]


def ttest_neat(enemy):
    gain_stats = []

    for run in range(10):
        file_name = f'NEAT_experiment/enemy_{enemy}/NEAT_run{run}/results.txt'

        gains = np.genfromtxt(file_name, skip_header=1,
                              usecols=(3), delimiter=',')

        gain_stats.append((np.mean(gains), np.std(gains)))

    return gain_stats


def ttest_deap(enemy):
    N_GAMES = 5
    experiment_dir = 'DEAPAexperimentE2'  # Directory containing all run folders
    num_runs = 10  # Number of runs

    # Find the best run and generation
    best_run_idx, best_generation_idx, max_fitness_value = read_results(
        experiment_dir, num_runs)

    # Retrieve the weights of the best individual from the best run
    best_weights = get_best_individual_weights(experiment_dir, best_run_idx)

    if best_weights is None:
        return []

    n_hidden_neurons = 10
    env = Environment(experiment_name=experiment_dir,
                      enemies=[enemy],
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False,
                      randomini="yes"
                      )

    gains = play_games(env, best_weights, N_GAMES)
    return gains


if __name__ == "__main__":
    for enemy in enemies:
        gain_stats_NEAT = ttest_neat(enemy)
        gain_stats_DEAP = ttest_deap(enemy)

        print(gain_stats_DEAP)
        print(gain_stats_NEAT)

        stats.ttest_ind_from_stats()
