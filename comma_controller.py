from evoman.controller import Controller
import numpy as np


class PlayerControllerCOMMA(Controller):
    def __init__(self, _n_hidden: int):
        self.n_hidden = _n_hidden

    def set_weights(self, individual_weights: list, n_inputs: int):
        """
        Extract the weights from the DEAP individual and set up the network
        """
        # Total number of neurons
        n_hidden = self.n_hidden
        n_outputs = 5

        # Extract weights for the input-to-hidden layer
        start, end = 0, (n_inputs + 1) * n_hidden  # +1 for bias
        self.w_input_hidden = np.array(individual_weights[start:end]).reshape((n_inputs + 1, n_hidden))

        # Extract weights for the hidden-to-output layer
        start, end = end, end + (n_hidden + 1) * n_outputs  # +1 for bias
        self.w_hidden_output = np.array(individual_weights[start:end]).reshape((n_hidden + 1, n_outputs))

    def control(self, inputs: list, individual_weights: list = None):
        """
        Evaluate the next move
        """
        # Normalize inputs to range [0, 1] and add bias
        inputs = (inputs - np.min(inputs)) / (np.max(inputs) - np.min(inputs))
        inputs = np.append(inputs, 1)

        # Forward pass + add bias
        hidden_activations = np.tanh(np.dot(inputs, self.w_input_hidden))
        hidden_activations = np.append(hidden_activations, 1)

        output = np.tanh(np.dot(hidden_activations, self.w_hidden_output))
        
        left, right, jump, shoot, release = self.extract_net_output(output)
        return [left, right, jump, shoot, release]






    """
    def control(self, inputs: list, individual_weights):
        
        Evaluates the next move of actions.
        The actions are: left, right, jump, shoot and release.

        inputs: the list of value inputs for each of the sensors

        Returns:

        A boolean list corresponding to each of the actions,
        which defines the action's activity.
        
        inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))

        output = cont.activate(inputs)
        left, right, jump, shoot, release = self.extract_net_output(output)
        return [left, right, jump, shoot, release]

    """
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