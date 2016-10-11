from ca import CA
import numpy as np
import time
import util

def test():
    """ Internal test for the CA """
    init_config = util.rand_init_config(11)
    rule = 250
    start = int(round(time.time() * 1000))
    automation = CA(1, rule, np.asarray(init_config), 5)
    automation.start()
    print "Time to generate and enum:", (int(round(time.time() * 1000)) - start)

if __name__ == '__main__':
    test()

