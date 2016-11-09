# Module to generate configurations corresponding to different problems
# with labels/classes/fasit

import random as rn
import numpy as np


def density(quantity, size=5, on_probability=0.5):
    """Generates initial configurations for the density classification,
    along with correct classifications

    :param quantity: number of configurations to generate
    :param size: length
    :return: configurations, labels
    """
    configs = []
    labels = []

    for c in xrange(quantity):
        config = []
        ones = 0
        zeros = 0
        for i in xrange(size):
            state = 1 if rn.random() < on_probability else 0
            if state == 1:
                ones += 1
            else:
                zeros += 1
            config.append(state)
        configs.append(config)
        labels.append(1 if ones > zeros else 0)
    return configs, labels


def parity(quantity, size=4):
    """Generates initial configurations for the parity problem,
    along with the corresponding solutions

    :param quantity: number of configurations to generate
    :param size: length
    :return: configurations, labels
    """
    configs = []
    labels = []

    for c in xrange(quantity):
        config = []
        ones = 0
        for i in xrange(size):
            state = rn.randint(0, 1)
            if state == 1:
                ones += 1
            config.append(state)
        configs.append(config)
        labels.append(1 if ones % 2 == 0 else 0)
    return configs, labels


def bit_memory_task(quantity, bits, distractor_period):
    """

    :param quantity: the number of tasks to generate
    :param bits:
    :param distractor_period:
    :return: tasks, labels
    """
    tasks = []
    labels = []

    for i in xrange(quantity):
        y = 2  # y being 0, 1, 2 means which of the output nodes y1, y2, y3 that is activated, respectively
        task = []
        label = []
        for t in xrange(bits + distractor_period + 1 + bits):
            if t < bits:
                bit = (i & 2**t) / 2**t
                a1 = - bit + 1
                a2 = bit
            else:
                a1 = 0
                a2 = 0
                # if rn.random() > 0.5:
                #     a1 = 1
                #     a2 = 0
                # else:
                #     a1 = 0
                #     a2 = 1

            if bits <= t < bits + distractor_period or t > bits + distractor_period:
                # Distractor signal
                a3 = 1
            else:
                a3 = 0

            if bits + distractor_period == t:
                # Cue signal
                a4 = 1
            else:
                a4 = 0

            task.append(np.asarray([a1, a2, a3, a4]))

            if t > bits + distractor_period:
                y1 = task[t - (bits + distractor_period) - 1][0]
                y = 0 if y1 == 1 else 1

            # if i == 9:
            #     print "%d %d %d %d -> %d" % (a1, a2, a3, a4, y)
            label.append(y)
        tasks.append(task)
        labels.append(label)
    return tasks, labels

