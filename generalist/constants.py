import os
NUM_GENERATIONS = 30
NUM_RUNS = 10
NUM_TRIALS_NEAT = 50
NUM_TRIALS_DEAP = 26

PATH_NEAT = os.path.join("results", "NEAT")
PATH_DEAP = os.path.join("results", "DEAP")

OUTPUT_FOLDER_TUNING = "tuning"
OUTPUT_FOLDER_TUNING_BEST = "tuned"
OUTPUT_FOLDER_TRAINING = "trained"
OUTPUT_FOLDER_TESTING = "tested"

ENEMY_GROUP_1 = [3, 4, 5]
ENEMY_GROUP_2 = [1, 2, 6, 7, 8]
ALL_ENEMIES = [1, 2, 3, 4, 5, 6, 7, 8]


def enemy_group_to_str(arr: list):
    return "_".join([str(x) for x in arr])
