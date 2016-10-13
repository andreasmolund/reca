# Module for simulating the CA as a stand-alone
# CASim = CA simulation
import sys
import getopt
from ca import CA
import numpy as np
import util
import math


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
                simple = a in ['True', 'true', 'y', 'yes', 'seff']
            elif o == '-r':
                rule = int(a)
            elif o == '-i':
                iterations = int(a)

    if iterations == 0:
        iterations = int(math.ceil((length + 1) / 2))
    init_config = util.config_simple(length) if simple else util.config_rand(length)
    automation = CA(1, rule, np.asarray(init_config), iterations)

    for t in xrange(iterations):
        util.print_config_1dim(automation.config)
        automation.step()


if __name__ == '__main__':
    sim(sys.argv)

