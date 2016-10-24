# Utilities related to reservoir handling


def classify_output(vector):
    new = []
    for o in xrange(len(vector)):
        val = vector[o]
        new.append(0 if val < 0.5 else 1)
    return new
