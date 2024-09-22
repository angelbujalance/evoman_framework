from evoman.controller import Controller
import neat


class PlayerControllerNEAT(Controller):
    def __init__(self, _n_hidden):
        self.n_hidden = [_n_hidden]

    def set(self, genome, n_inputs):
        # TODO
        # in the demo controller, the biases and weights were set here
        ...

    def control(self, params, cont: neat.nn.FeedForwardNetwork = None):
        # Normalises the input using min-max scaling
        params = (params-min(params))/float((max(params)-min(params)))

        output = cont.activate(params)
        left, right, jump, shoot, release = self.extract_net_output(output)
        return [left, right, jump, shoot, release]

    def extract_net_output(self, output):
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
