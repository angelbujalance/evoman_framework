from evoman.controller import Controller
import neat


class PlayerControllerNEAT(Controller):
    def __init__(self, _n_hidden):
        self.n_hidden = [_n_hidden]
        self.net = None  # NOTE: I guess the net can be initialized here

    def set(self, genome, n_inputs):
        # TODO
        # in the demo controller, the biases and weights were set here
        # maybe set self.net here?
        pass

    # NOTE: not sure how to use cont / if it's needed with NEAT
    def control(self, params, cont=None):
        # Normalises the input using min-max scaling
        params = (params-min(params))/float((max(params)-min(params)))

        output = self.net.activate(params)
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
