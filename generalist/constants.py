import os

DEBUG = False

NUM_GENERATIONS = 1 if DEBUG else 30
NUM_RUNS = 1 if DEBUG else 10
NUM_TRIALS_NEAT = 1 if DEBUG else 50
NUM_TRIALS_DEAP = 1 if DEBUG else 26

PATH_NEAT = os.path.join("results", "NEAT")
PATH_DEAP = os.path.join("results", "DEAP")

OUTPUT_FOLDER_TUNING = "tuning"
OUTPUT_FOLDER_TUNING_BEST = "tuned"
OUTPUT_FOLDER_TRAINING = "trained"
OUTPUT_FOLDER_TESTING = "tested"

ENEMY_GROUP_1 = [3, 4, 5]
ENEMY_GROUP_2 = [1, 2, 6, 7, 8]
ALL_ENEMIES = [1, 2, 3, 4, 5, 6, 7, 8]

# Make use of DEAP's CMA-ES algorithm
USE_CMA = True


def enemy_folder(enemy_group: list):
    return "enemies_" + enemy_group_to_str(enemy_group)


def enemy_group_to_str(enemy_group: list):
    return "_".join([str(x) for x in enemy_group])


# TUNING parameters
TUNING_POP_SIZE_MIN = 2 if DEBUG else 50
TUNING_POP_SIZE_MAX = 2 if DEBUG else 100
