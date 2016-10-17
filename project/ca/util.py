# Utility functions for CA related stuff
import random as rn
import math


def print_config_1dim(config):
    """Prints the configuration by blocks of characters

    :param config: a 1D list of states to print
    """
    line = ""
    on = "  "
    off = unichr(0x2588) + unichr(0x2588)
    for i in xrange(len(config)):
        line += on if config[i] == 0b1 else off
    print line


def config_simple(size=21):
    """ Sets all states to 0b0, except for the center most one that is set to 0b1

    :param size: size or length of the wanted 1D configuration
    :param k:
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
    """Traverses the whole rule space

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


def get_rule(rule=0, k=2, n=3):
    """Gets the transition function for the rule number

    :param rule: the rule
    :param k: number of states
    :param n: number of neighbors
    """
    rules = get_all_rules(k, n)
    rule = rules[rule]

    return rule

