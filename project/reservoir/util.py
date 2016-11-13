# Utilities related to reservoir handling


def classify_output(vector):
    return [0 if val < 0.5 else 1 for val in vector]
