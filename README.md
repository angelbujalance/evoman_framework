# Evoman
Evoman [[1]](#1) is a video game playing framework made in PyGame to be used as a testbed for optimization algorithms. It's inspired on the game MegaMan II.

A demo can be found here:  https://www.youtube.com/watch?v=ZqaMjd1E4ZI

Evolutionary Computing is used to train an AI-player to defeat the enemies. Each level contains one enemy, which has its own characteristics, such as the way it attacks. This enemy can shoot projectiles and the player can shoot to decrease the health of the enemy. The moves of the enemies are predefined and resemble the enemies from MegaMan II.

There are two methods, namely the specialist and the generalist. For the specialist, only a single objective is used for training. The objective is the enemy in this case. On the other hand, a generalist performs training on multiple enemies.

There are 20 values (sensors) in total which can determine the player's next action. 16 sensors are used for the distances (x, y) between the player and one of the maximum 8 projectiles. Another tuple (x, y) of two sensors is used for the distance between the player and the enemy. Finally, two sensors are used for the player's facing direction.

# Specialist
In this repository, we implemented two Evolutionary Algorithms (EA) using NEAT and comma-selection.

# Install
Use the requirements.txt file to install the dependencies.

# How to use
## Human controller
This game can be played with a human player against each of the enemies. The game will start with level 1, when the player or the enemy reaches a health of zero, the level is ended and will continue to the next one, regardless of who has one. This also happens in case of a time-out.

To play the game, use the following command:

```python human_demo.py```

## EA 1: NEAT
The following files are specific to this EA:

- config_specialist_NEAT: The configuration file to use with NEAT.
- neat_controller.py: The AI-player controller, which determines the next action.
- optimization_specialist_NEAT.py: Initializes the population and runs NEAT's genetic algorithm.

This algorithm trains on levels/enemies 1, ??? and 8.

To run this method, use the following command:

```python optimization_specialist_NEAT.py```

This will produce output files (logs and checkpoints) in the folder `optimization_NEAT`. If there already is a checkpoint file in here, this will be used to continue the algorithm.

## EA 2: Comma-selection with DEAP

The following files are specific to this EA:

- deap_controller.py: The AI-player controller, which determines the next action.
- optimization_specialist_DEAP.py: Initializes the population and runs the comma-selection algorithm using DEAP's framework.

To run this method, use the following command:

```python optimization_specialist_DEAP.py```

# References
<a id="1">[1]</a>
Karine da Silva Miras de Araújo and Fabrício Olivetti de França. 2016. An electronic-game framework for evaluating coevolutionary algorithms. arXiv.NE/1604.00644. https://arxiv.org/abs/1604.00644
