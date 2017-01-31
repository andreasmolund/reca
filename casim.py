# Module for simulating the CA as a stand-alone
# CASim = CA simulation
import getopt
import math
import sys

import ca.util as cutil
from ca.ca import CA
from plotter import plot_temporal


def sim(raw_args):
    """Internal test for the CA"""

    length = 5
    rule = 90
    simple = True
    iterations = 0

    if len(raw_args) > 1:
        # length, simple, rule
        opts, args = getopt.getopt(raw_args[1:], "l:s:r:i:")
        for o, a in opts:
            if o == '-l':
                length = int(a)
            elif o == '-s':
                simple = a in ['True', 'true', 'y', 'yes', 'seff', 'ofc']
            elif o == '-r':
                rule = int(a)
            elif o == '-i':
                iterations = int(a)

    if iterations == 0:
        iterations = int(math.ceil((length + 1) / 2))
    config = cutil.config_simple(length) if simple else cutil.config_rand(length)
    automation = CA(rule, n=3)

    state_vector = []
    state_vector.extend(config)
    for t in xrange(iterations):
        # util.print_config_1dim(config)
        config = automation.step(config)
        state_vector.extend(config)

    plot_temporal([state_vector],
                  1,
                  length,
                  1,
                  1 + iterations,
                  sample_nr=0)


if __name__ == '__main__':
    # sys.argv = ['casim.py', '-s', 'no', '-r', '52', '-l', '10', '-i' '10']
    sim(sys.argv)

