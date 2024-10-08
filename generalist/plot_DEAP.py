import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from enemy_groups import ENEMY_GROUP_1, ENEMY_GROUP_2, enemy_group_to_str

# Function to compute confidence intervals


def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    h = sem * stats.t.ppf((1 + confidence) / 2, n - 1)  # Margin of error
    return mean, mean - h, mean + h


def read_results(experiment_dir, num_runs, enemy):
    all_best_fitness = []
    all_mean_fitness = []
    all_std_fitness = []

    for i in range(num_runs):
        results_path = os.path.join(
            experiment_dir, f'DEAP_runE{enemy}{i}', 'logbook.csv')
        print(f"Trying to read: {results_path}")

        try:
            data = pd.read_csv(results_path)
            best_fitness = data['max'].tolist()
            mean_fitness = data['avg'].tolist()
            std_fitness = data['std'].tolist()

            # Collect data for all generations
            all_best_fitness.append(best_fitness)
            all_mean_fitness.append(mean_fitness)
            all_std_fitness.append(std_fitness)
        except Exception as e:
            print(f"Error reading {results_path}: {e}")

    return all_best_fitness, all_mean_fitness, all_std_fitness


def NEAT_results(enemy):

    # Generate file names for the different runs
    files = [
        f'NEAT_experiment/enemy_{enemy}/NEAT_run{run}/results_clean.txt' for run in range(10)]

    # Read the files and append them to a list of dataframes
    dfs = []
    for run, file_name in enumerate(files):
        df = pd.read_csv(file_name)
        df['run'] = run  # Add a new column to track the run
        dfs.append(df)

    # Combine all DataFrames into a single DataFrame
    combined_df = pd.concat([df.assign(run=i+1) for i, df in enumerate(dfs)])

    # Group the combined dataframe by generation to calculate the statistics
    grouped_mean_fitness = combined_df.groupby(
        'generation')['mean_fitness'].apply(list)
    grouped_best_fitness = combined_df.groupby(
        'generation')['best_fitness'].apply(list)

    # Calculate the mean and confidence intervals for mean fitness per generation
    mean_fitness_means = grouped_mean_fitness.apply(np.mean)
    mean_fitness_conf_ints = grouped_mean_fitness.apply(confidence_interval)

    # Calculate the mean and confidence intervals for best fitness per generation
    best_fitness_means = grouped_best_fitness.apply(np.mean)
    best_fitness_conf_ints = grouped_best_fitness.apply(confidence_interval)

    return mean_fitness_means, mean_fitness_conf_ints, best_fitness_means, best_fitness_conf_ints


# TODO: is the best fitness the average of the ten runs?

def plot_fitness(all_best_fitness, all_mean_fitness, all_std_fitness, experiment_dir, enemy_group: list):
    num_generations = len(all_mean_fitness[0])
    # Aggregate data across runs
    avg_best_fitness = np.mean(all_best_fitness, axis=0)
    std_best_fitness = np.std(all_best_fitness, axis=0)

    avg_mean_fitness = np.mean(all_mean_fitness, axis=0)
    avg_std_fitness = np.mean(all_std_fitness, axis=0)

    generations = range(num_generations)

    average_fitness_NEAT, average_fitness_NEAT_CI, best_fitness_NEAT, best_fitness_NEAT_CI, = NEAT_results(
        enemy_group)

    # Separate the mean, lower, and upper confidence intervals for mean fitness
    mean_fitness_values = [c[0] for c in average_fitness_NEAT_CI]
    mean_ci_lower = [c[1] for c in average_fitness_NEAT_CI]
    mean_ci_upper = [c[2] for c in average_fitness_NEAT_CI]

    # Separate the mean, lower, and upper confidence intervals for best fitness
    best_fitness_values = [c[0] for c in best_fitness_NEAT_CI]
    best_ci_lower = [c[1] for c in best_fitness_NEAT_CI]
    best_ci_upper = [c[2] for c in best_fitness_NEAT_CI]

    # Plot the average fitness with standard deviation for DEAP
    plt.plot(generations, avg_mean_fitness,
             label='Average DEAP', color='blue', ls="--")
    plt.fill_between(generations,
                     np.array(avg_mean_fitness) - np.array(avg_std_fitness),
                     np.array(avg_mean_fitness) + np.array(avg_std_fitness),
                     color='blue', alpha=0.2)

    # Plot the average fitness with standard deviation for NEAT
    plt.plot(generations[:25], average_fitness_NEAT,
             label='Average NEAT', color='green', ls="--")
    plt.fill_between(average_fitness_NEAT.index, mean_ci_lower, mean_ci_upper,
                     color='green', alpha=0.2)

    # Plot the best fitness for DEAP
    plt.plot(generations, avg_best_fitness, label='Best DEAP', color='blue')
    plt.fill_between(generations,
                     np.array(avg_best_fitness) - np.array(std_best_fitness),
                     np.array(avg_best_fitness) + np.array(std_best_fitness),
                     color='blue', alpha=0.2)

    # Plot the best fitness for NEAT
    plt.plot(generations[:25], best_fitness_NEAT,
             label='Best NEAT', color='green')
    plt.fill_between(best_fitness_NEAT.index, best_ci_lower, best_ci_upper,
                     color='green', alpha=0.2)

    # Add labels and title
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title(
        f'Fitness Over Generations for Enemy {enemy_group_to_str(enemy_group)}')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(f'fitness_plot_enemy_{enemy_group_to_str(enemy_group)}.png')
    plt.show()


if __name__ == '__main__':
    for enemy_group in [ENEMY_GROUP_1, ENEMY_GROUP_2]:
        # Directory containing all run folders
        experiment_dir = f'DEAPexperimentE{enemy_group_to_str(enemy_group)}'
        num_runs = 10  # Number of runs

        # Read the results from the files
        all_best_fitness, all_mean_fitness, all_std_fitness = read_results(
            experiment_dir, num_runs, enemy_group)

        # Plot the fitness results
        plot_fitness(all_best_fitness, all_mean_fitness,
                     all_std_fitness, experiment_dir, enemy_group)
