# Utility functions for CA related stuff
import random as rn
import math


def print_config_1dim(config, prefix="", postfix=""):
    """Prints the configuration by blocks of characters

    :param config: a 1D list of states to print
    :param prefix:
    :param postfix:
    """
    line = ""
    on = unichr(0x2588) + unichr(0x2588)
    off = "  "
    for i in xrange(len(config)):
        line += on if config[i] == 0b1 else off
    print prefix + line + postfix


def config_simple(size=21):
    """ Sets all states to 0b0, except for the middle one that is set to 0b1

    :param size: size or length of the wanted 1D configuration
    :return: configuration
    """
    config = []
    for i in xrange(size):
        config.append(0b0)
    config[int(math.ceil(size/2))] = 0b1
    return config


def config_rand(size=20, k=2):
    """Generates a random configuration

    :param size:
    :param k:
    :return:
    """
    config = []
    ones = 0
    zeros = 0
    for i in xrange(size):
        state = rn.randint(0, k - 1)
        if state == 1:
            ones += 1
        else:
            zeros += 1
        config.append(state)
    return config


def get_all_rules(k, n):
    """Traverses the whole rule space for the given k and n

    :param k: number of states
    :param n: number of neighbors
    """

    rules = []

    for x in xrange(k**(k**n)):
        rule = {}
        for y in xrange(k**n):
            rule[y] = 0b1 if x > 0 and x & 2**y == 2**y else 0b0
        rules.append(rule)

    return rules


def get_elementary_rule(wolfram_code):
    """Gets the transition function for the rule number

    :param wolfram_code: the rule
    """
    base2rule = format(wolfram_code, '08b')

    return {
        (1, 1, 1): int(base2rule[0]),
        (1, 1, 0): int(base2rule[1]),
        (1, 0, 1): int(base2rule[2]),
        (1, 0, 0): int(base2rule[3]),
        (0, 1, 1): int(base2rule[4]),
        (0, 1, 0): int(base2rule[5]),
        (0, 0, 1): int(base2rule[6]),
        (0, 0, 0): int(base2rule[7])
    }

    # return [(wolframcode/pow(k, i)) % k for i in range(pow(k, n))]

