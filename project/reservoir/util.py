# Utilities related to reservoir handling

def classify_output(vector):
    new = []
    for o in xrange(len(vector)):
        val = vector[o]
        if 0.5 < val < 1.5:
            new.append(1)
        elif -0.5 < val < 0.5:
            new.append(0)
        else:
            new.append(2)
    return new