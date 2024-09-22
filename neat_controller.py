from evoman.controller import Controller
import neat


class PlayerControllerNEAT(Controller):
    def __init__(self, _n_hidden: int):
        self.n_hidden = [_n_hidden]

    def set(self, genome: neat.nn.FeedForwardNetwork, n_inputs: int):
        # TODO
        # in the demo controller, the biases and weights were set here
        ...

    def control(self, inputs: list, cont: neat.nn.FeedForwardNetwork = None):
        """
        Evaluates the next move of actions.
        The actions are: left, right, jump, shoot and release.

        inputs: the list of value inputs for each of the sensors
        cont: NEAT net to be evaluated

        Returns:

        A boolean list corresponding to each of the actions,
        which defines the action's activity.
        """
        # print("player_controller control", cont.fitness, len(inputs))
        # Normalises the input using min-max scaling
        inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))

        output = cont.activate(inputs)
        left, right, jump, shoot, release = self.extract_net_output(output)
        return [left, right, jump, shoot, release]

    def extract_net_output(self, output: list):
        if output[0] > 0.5:
            left = 1
        else:
            left = 0

        if output[1] > 0.5:
            right = 1
        else:
            right = 0

        if output[2] > 0.5:
            jump = 1
        else:
            jump = 0

        if output[3] > 0.5:
            shoot = 1
        else:
            shoot = 0

        if output[4] > 0.5:
            release = 1
        else:
            release = 0

        return left, right, jump, shoot, release
