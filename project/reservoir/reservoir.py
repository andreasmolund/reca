import scipy as sp
import numpy as np



def compute(reservoir):
    """Uses the reservoir provided, and lets it digest the initial configuration

    :return: all state vectors concatenated
    """
    concat = reservoir.config
    for i in xrange(reservoir.iterations):
        reservoir.step()
        concat = np.append(concat, reservoir.config)
    return concat
