import util


class ECA:
    """The class that holds the elementary cellular automaton."""

    def __init__(self, rule):
        self.rule = rule
        self.transition = util.get_elementary_rule(rule)

    def step(self, config):
        """Applying the transition function one time,
        and returning the next state vector.

        :param config:
        :return:
        """
        size = len(config)
        next_config = [0b0] * size
        transition = self.transition

        for c in xrange(size):
            next_config[c] = transition[(config[(c - 1) % size],
                                         config[c % size],
                                         config[(c + 1) % size])]

        return next_config


