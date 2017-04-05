# Encoder for real valued input.

from encoders.classic import ClassicEncoder


def quantize_activation(vector, p1=0.25, p2=0.5, p3=0.75):
    q = []
    for value in vector:
        if value < p1:
            q.extend([0, 0])
        elif value < p2:
            q.extend([0, 1])
        elif value < p3:
            q.extend([1, 1])
        else:
            q.extend([1, 0])
    return q


class RealEncoder(ClassicEncoder):
    """
    Encoder to use for floating point input.
    What differs this from ClassicEncoder is the translation stage/function.
    This class uses the same type of random mapping,
    however, it must binarize or quantize input elements before mapping them onto automata

    """

    def __init__(self,
                 n_random_mappings,
                 input_size,
                 input_area,
                 automaton_area,
                 input_offset=0,
                 verbose=0,
                 group_len=1,
                 quantize=quantize_activation):
        self.quantize = quantize
        self.quantize_len = len(quantize([1]))
        super(RealEncoder, self).__init__(n_random_mappings,
                                          self.quantize_len * input_size,
                                          self.quantize_len * input_area,
                                          self.quantize_len * automaton_area,
                                          input_offset,
                                          verbose,
                                          group_len=self.quantize_len)

    def _overwrite(self, master, second):
        return super(RealEncoder, self)._overwrite(self.quantize(master), second)


def quantize_japvow(vector):
    q = []
    # for value in vector:
    #     if value < -0.2501245:
    #         q.extend([0, 0, 1])
    #     elif value < -0.066513:
    #         q.extend([0, 1, 0])
    #     elif value < 0.14068925:
    #         q.extend([0, 1, 1])
    #     else:
    #         q.extend([1, 0, 1])

    for value in vector:
        if value < -0.2984444:
            q.extend([0, 0, 0, 1])
        elif value < -0.1379516:
            q.extend([0, 0, 1, 1])
        elif value < 0.004259:
            q.extend([0, 1, 1, 1])
        elif value < 0.2079332:
            q.extend([0, 1, 1, 0])
        else:
            q.extend([0, 1, 0, 0])

    # 14: [0, 0, 0, 1] [0, 1, 0, 1] [0, 1, 1, 0] [1, 0, 0, 1] [1, 0, 0, 0]

    return q


def quantize_cifar(vector):
    q = []
    for value in vector:
        if value < 70:
            q.extend([1, 1, 1])
        elif value < 117:
            q.extend([0, 1, 1])
        elif value < 167:
            q.extend([0, 0, 1])
        else:
            q.extend([0, 0, 0])
    return q
