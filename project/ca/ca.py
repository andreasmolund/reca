import numpy as np

import util


class CA:
    """The class that holds the 1D, Boolean cellular automation.
    Thinking about generating the rules at initialization."""

    def __init__(self, rule, k=2, n=3):
        self.rule = rule
        self.k = k
        self.n = n
        self.transition = util.get_rule(rule, k, n)
        self.radius = (n - 1) / 2

    def step(self, config):
        size = config.shape[0]
        next_config = np.empty(size, dtype='int')
        indices = np.arange(-self.radius, self.radius + 1)

        for c in xrange(size):
            template = config.take(indices + c, mode='wrap')
            neighborhood = 0
            for bit in template:
                neighborhood = (neighborhood << 1) | bit

            # neighborhood = 0b0
            # power = self.n - 1
            # for d in xrange(-self.radius, self.radius + 1):
            #     state = config[(c + d) % size]
            #     neighborhood += 2**power * state
            #     power -= 1
            next_state = self.transition[neighborhood]
            next_config[c] = next_state

        return next_config

    def copy(self):
        return CA(self.rule, self.k, self.n)


