import util
import scipy as sp
import numpy as np


class CA:
    """The class that holds the 1D, Boolean cellular automation.
    Thinking about generating the rules at initialization."""

    def __init__(self, rule, k=2, n=3, visual=False):
        self.transition = util.get_rule(rule, k, n)
        self.n = n
        self.radius = (n - 1) / 2
        self.visual = visual

    def step(self, config):
        size = len(config)
        next_config = sp.zeros([size], dtype=np.dtype(int))

        for c in xrange(size):
            neighborhood = 0b0
            power = self.n - 1
            for d in xrange(-self.radius, self.radius + 1):
                state = config[(c + d) % size]
                neighborhood += 2**power * state
                power -= 1
            next_state = self.transition[neighborhood]
            next_config[c] = next_state

        return next_config


# Useful
#   pylab's subplot for plotting, e.g. as a func of time
#   pylab's imshow function
#       cmap option in imshow is to specify colour scheme
