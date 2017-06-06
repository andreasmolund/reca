# Module for simulating the CA as a stand-alone
# ECASim = Elementary cellular automata simulation
import getopt
import math
import sys

import ca.util as cutil
from ca.eca import ECA
from stats.plotter import plot_temporal


def sim(raw_args):
    """Internal test for the ECA"""

    size = 5
    rule = 90
    simple = False
    iterations = 0

    if len(raw_args) > 1:
        # length, simple, rule
        opts, args = getopt.getopt(raw_args[1:],
                                   "s:r:I:",
                                   ['simple'])
        for o, a in opts:
            if o == '-s':
                size = int(a)
            elif o == '--simple':
                simple = True
            elif o == '-r':
                rule = int(a)
            elif o == '-I':
                iterations = int(a)

    if iterations == 0:
        iterations = int(math.ceil((size + 1) / 2))
    config = cutil.config_simple(size) if simple else cutil.config_rand(size)
    automation = ECA(rule)

    state_vector = []
    state_vector.extend(config)
    for t in xrange(iterations):
        # util.print_config_1dim(config)
        config = automation.step(config)
        state_vector.extend(config)

    plot_temporal([state_vector],
                  1,
                  size,
                  1,
                  1 + iterations,
                  sample_nr=0)


if __name__ == '__main__':
    # sys.argv = ['ecasim.py', '-s', 'no', '-r', '52', '-l', '10', '-i' '10']
    sim(sys.argv)

