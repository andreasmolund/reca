import util
import scipy as sp
import numpy as np


class CA:
    """The class that holds the 1D, Boolean cellular automation.
    Thinking about generating the rules at initialization."""

    def __init__(self, dim, rule, init_config, iterations, k=2, n=3, visual=False):
        self.transition = util.get_rule(rule, k, n)
        self.set_config(init_config)
        self.iterations = iterations
        self.n = n
        self.radius = (n - 1) / 2
        self.visual = visual
        self.time = 0

        print "Initialized CA with dimension", dim, \
            "and rule", rule,\
            "for", iterations, "steps."

    def set_config(self, config):
        self.config = config
        self.size = len(config)

    def step(self):
        next_config = sp.zeros([self.size], dtype=np.dtype(int))

        for c in xrange(self.size):
            neighborhood = 0b0
            power = self.n - 1
            for d in xrange(-self.radius, self.radius + 1):
                state = self.config[(c + d) % self.size]
                neighborhood += 2**power * state
                power -= 1
            next_state = self.transition[neighborhood]
            next_config[c] = next_state

        self.config = next_config
        self.time += 1

    def step_all(self):
        for i in xrange(self.iterations):
            if self.visual:
                util.print_config_1dim(self.config)
            self.step()


# Useful
#   pylab's subplot for plotting, e.g. as a func of time
#   pylab's imshow function
#       cmap option in imshow is to specify colour scheme
