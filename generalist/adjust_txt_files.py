import os
from constants import (PATH_NEAT, OUTPUT_FOLDER_TRAINING, NUM_RUNS,
                       ENEMY_GROUP_1, ENEMY_GROUP_2, enemy_folder)

if __name__ == "__main__":
    for train_enemy_group in [ENEMY_GROUP_1, ENEMY_GROUP_2]:
        for run in range(NUM_RUNS):
            file_name = os.path.join(PATH_NEAT, OUTPUT_FOLDER_TRAINING,
                                     enemy_folder(train_enemy_group),
                                     f'run_{run}',
                                     'results.txt')

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

            file = os.path.join(PATH_NEAT, OUTPUT_FOLDER_TRAINING,
                                enemy_folder(train_enemy_group),
                                f'run_{run}',
                                'results_clean.txt')

            with open(file, 'w') as file:
                file.writelines(new_results)
