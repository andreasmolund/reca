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
    a1 = 0
    a2 = 0
    a3 = 0  # Distractor signal
    a4 = 0  # Cue signal
    y1 = 0
    y2 = 0
    y3 = 1
    tasks = []
    labels = []

    for i in xrange(quantity):
        task = []
        label = []
        for t in xrange(bits + distractor_period + 1 + bits):
            if rn.random() > 0.5:
                a1 = 1
                a2 = 0
            else:
                a1 = 0
                a2 = 1

            if bits <= t < bits + distractor_period or t > bits + distractor_period:
                # In the distractor period
                a3 = 1
            else:
                a3 = 0

            if bits + distractor_period == t:
                # Cue signal
                a4 = 1
            else:
                a4 = 0

            task.append(np.asarray([a1, a2, a3, a4]))
            label.append(np.asarray([y1, y2, y3]))

            if t > bits + distractor_period:
                y1 = task[t - (bits + distractor_period) - 1][0]
                y2 = task[t - (bits + distractor_period) - 1][1]
                y3 = 0

            print "%d %d %d %d\t\t%d %d %d" % (a1, a2, a3, a4, y1, y2, y3)
        tasks.append(task)
        labels.append(label)
    return tasks, labels

