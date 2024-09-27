import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def read_results(experiment_dir, num_runs):
    all_best_fitness = []
    all_mean_fitness = []
    all_std_fitness = []

    for i in range(num_runs):
        results_path = os.path.join(experiment_dir, f'DEAP_run{i}', 'logbook.csv')
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

#TODO: is the best fitness the average of the ten runs? 

def plot_fitness(all_best_fitness, all_mean_fitness, all_std_fitness, experiment_dir):
    num_generations = len(all_mean_fitness[0]) 
    # Aggregate data across runs
    avg_best_fitness = np.mean(all_best_fitness, axis=0)
    std_best_fitness = np.std(all_best_fitness, axis=0)

    avg_mean_fitness = np.mean(all_mean_fitness, axis=0)
    avg_std_fitness = np.mean(all_std_fitness, axis=0)

    generations = range(num_generations)

    # Plot the average fitness with standard deviation
    plt.plot(generations, avg_mean_fitness, label='Average Fitness', color='blue')
    plt.fill_between(generations,
                     np.array(avg_mean_fitness) - np.array(avg_std_fitness),
                     np.array(avg_mean_fitness) + np.array(avg_std_fitness),
                     color='blue', alpha=0.2, label='Standard Deviation')

    # Plot the best fitness
    plt.plot(generations, avg_best_fitness, label='Best Fitness', color='green')
    plt.fill_between(generations,
                     np.array(avg_best_fitness) - np.array(std_best_fitness),
                     np.array(avg_best_fitness) + np.array(std_best_fitness),
                     color='green', alpha=0.2, label='Best Fitness Std Dev')

    # Add labels and title
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Fitness Over Generations for Enemy 8 DEAP')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(os.path.join(experiment_dir, 'DEAP_fitness_plot.png'))
    plt.show()

if __name__ == '__main__':
    experiment_dir = 'DEAPAexperiment'  # Directory containing all run folders
    num_runs = 10  # Number of runs

    # Read the results from the files
    all_best_fitness, all_mean_fitness, all_std_fitness = read_results(experiment_dir, num_runs)

    # Plot the fitness results
    plot_fitness(all_best_fitness, all_mean_fitness, all_std_fitness, experiment_dir)



