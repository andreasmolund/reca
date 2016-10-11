from ca import CA
import numpy as np
import util
import pylab as pl


def test():
    """Internal test for the CA"""
    size = 21
    init_config = util.rand_init_config(size)
    rule = 90

    automation = CA(1, rule, np.asarray(init_config), size)
    # automation.start()

    for t in xrange(size):
        automation.step()

        pl.ion()
        modelfigure = pl.figure()
        automation.draw()
        modelfigure.canvas.manager.window.update()
        pl.show()


def init_gui():



if __name__ == '__main__':
    init_gui()
    test()

