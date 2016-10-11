import util
import scipy as sp
import numpy as np


class CA:
    """The class that holds the 1D, Boolean cellular automation.
    Thinking about generating the rules at initialization."""

    def __init__(self, dim, rule, init_config, steps, k=2, n=3, visual=False):
        self.transition = util.get_rule(rule, k, n)
        self.config = init_config
        self.size = len(init_config)
        self.steps = steps
        self.n = n
        self.radius = (n - 1) / 2
        self.visual = visual

        print "Initialized with dimension", dim, \
            "and rule", rule,\
            "for", steps, "steps."

    def _step(self):
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

    def start(self):
        print self.config

        for i in xrange(self.steps):
            self._step()
            print self.config


# Useful
#   pylab's subplot for plotting, e.g. as a func of time
#   pylab's imshow function
#       cmap option in imshow is to specify colour scheme
