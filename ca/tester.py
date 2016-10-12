from ca import CA
import numpy as np
import util
import math


size = 101
steps = int(math.ceil(size / 2))


def test():
    """Internal test for the CA"""
    init_config = util.init_config_simple(size)
    rule = 90

    automation = CA(1, rule, np.asarray(init_config), steps)

    for t in xrange(steps):
        util.print_config_1dim(automation.config)
        automation.step()
        #time.sleep(0.05)


if __name__ == '__main__':
    test()

