import util

class CA:
    """The class that holds the automata.
    Thinking about generating the rules at initialization."""

    def __init__(self, dim, rule, init_state, steps, visual=False):
        self.dim = dim
        self.transition = util.exhaustive_enum(rule)
        self.init_state = init_state
        self.steps = steps
        self.visual = visual
        print("Initialized with dimension ", dim,
              ", rule ", rule,
              " for ", steps, " steps.")

    def _step(self):
        print("*performes one time step*")

    def _

