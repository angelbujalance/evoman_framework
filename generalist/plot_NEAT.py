import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

from enemy_groups import ENEMY_GROUP_1, ENEMY_GROUP_2, enemy_group_to_str

ENEMIES = ENEMY_GROUP_1

# Function to compute confidence intervals


def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    h = sem * stats.t.ppf((1 + confidence) / 2, n - 1)  # Margin of error
    return mean, mean - h, mean + h


# Generate file names for the different runs
files = [os.path.join('results', 'NEAT', 'tested',
                      f'enemy_{enemy_group_to_str(ENEMIES)}',
                      f'run_{run_idx}', 'results.csv')
         for run_idx in range(10)]

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

# Separate the mean, lower, and upper confidence intervals for mean fitness
mean_fitness_values = [c[0] for c in mean_fitness_conf_ints]
mean_ci_lower = [c[1] for c in mean_fitness_conf_ints]
mean_ci_upper = [c[2] for c in mean_fitness_conf_ints]

# Separate the mean, lower, and upper confidence intervals for best fitness
best_fitness_values = [c[0] for c in best_fitness_conf_ints]
best_ci_lower = [c[1] for c in best_fitness_conf_ints]
best_ci_upper = [c[2] for c in best_fitness_conf_ints]

# Plot the mean fitness with confidence interval
plt.figure(figsize=(10, 6))

# Plot the mean fitness line with confidence interval
plt.plot(mean_fitness_means.index, mean_fitness_values,
         label='Mean Fitness', color='blue', linewidth=2)
plt.fill_between(mean_fitness_means.index, mean_ci_lower,
                 mean_ci_upper, color='blue', alpha=0.2)

# Plot the best fitness line with confidence interval
plt.plot(best_fitness_means.index, best_fitness_values,
         label='Best Fitness', color='green', linewidth=2)
plt.fill_between(best_fitness_means.index, best_ci_lower,
                 best_ci_upper, color='green', alpha=0.2)

# Add labels and title
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Mean and Best Fitness Evolution with Confidence Intervals')
plt.legend()

plt.savefig("mean_best_fitness_evolution_with_CI.jpg",
            dpi=300, bbox_inches='tight')

# Show the plot
plt.tight_layout()
plt.show()
