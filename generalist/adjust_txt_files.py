import os
enemies = [1]
for enemy in enemies:
    for run in range(10):
        file_name = os.path.join('results', 'NEAT', 'trained', f'enemy_{enemy}',
                                 f'NEAT_run{run}', 'results.txt')

        with open(file_name, 'r') as file:
            lines = file.readlines()

        new_results = []
        lines = lines[1:]

        for gen, line in enumerate(lines):
            if not line.strip():  # skip any empty lines
                continue

            new_results.append(f'\n{gen},{line.strip()}')

        new_results = new_results[:31]
        header = "generation,best_fitness,mean_fitness,std_fitness,gain"
        new_results[0] = header

        file = os.path.join('results', 'NEAT', f'enemy_{enemy}',
                            f'NEAT_run{run}', 'results_clean.txt')

        with open(file, 'w') as file:
            file.writelines(new_results)
