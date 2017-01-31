import util


class CA:
    """The class that holds the 1D, Boolean cellular automation."""

    def __init__(self, rule, k=2, n=3):
        self.rule = rule
        self.k = k
        self.n = n
        self.transition = util.get_rule(rule, k, n)
        self.radius = (n - 1) / 2

    def step(self, config):
        """Applying the transition function one time,
        and returning the next state vector.

        :param config:
        :return:
        """
        size = len(config)
        next_config = [0b0] * size

        for c in xrange(size):
            neighborhood = 0b0
            for d in xrange(-self.radius, self.radius + 1):
                state = config[(c + d) % size]
                neighborhood = (neighborhood << 1) | state
            next_state = self.transition[neighborhood]
            next_config[c] = next_state

        return next_config


