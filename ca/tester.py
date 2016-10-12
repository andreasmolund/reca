from ca import CA
from graphics import CAWindow
import numpy as np
import util
import Queue
import threading
import time
import math


size = 101
steps = int(math.ceil(size / 2))


# def update_gui(queue):
#     win = CAWindow(width=size, height=steps)
#     while True:
#         entity = queue.get()
#         win.update_time(entity)
#
#
# time_queue = Queue.Queue()
# gui_thread = threading.Thread(target=update_gui, args=[time_queue])
# gui_thread.daemon = True
# gui_thread.start()


def test():
    """Internal test for the CA"""
    init_config = util.rand_init_config(size)
    rule = 90

    automation = CA(1, rule, np.asarray(init_config), steps)
    # automation.start()

    for t in xrange(steps):
        util.print_config_1dim(automation.config)
        automation.step()
        #time_queue.put(automation.config)
        #time.sleep(0.05)


if __name__ == '__main__':
    test()

