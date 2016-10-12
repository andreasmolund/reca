import sys
import getopt
from ca import CA
import numpy as np
import util
import math


length = 21
rule = 90
simple = True


def test(raw_args):
    """Internal test for the CA"""

    # length, simple, rule
    opts, args = getopt.getopt(raw_args[1:], "l:s:r:")
    for o, a in opts:
        if o == '-l':
            length = int(a)
        elif o == '-s':
            simple = a in ['True', 'true', 'y']
        elif o == '-r':
            rule = int(a)

    steps = int(math.ceil(length / 2))
    init_config = util.init_config_simple(length) if simple else util.init_config_rand(length)
    automation = CA(1, rule, np.asarray(init_config), steps)

    for t in xrange(steps):
        util.print_config_1dim(automation.config)
        automation.step()
        #time.sleep(0.05)


if __name__ == '__main__':
    test(sys.argv)

