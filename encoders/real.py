# Encoder for real valued input.

from encoders.classic import ClassicEncoder


def quantize_activation(value):
    if value < 0.05:
        representation = [0, 0, 0]
    elif value < 0.1:
        representation = [0, 0, 1]
    elif value < 0.15:
        representation = [0, 1, 1]
    else:
        representation = [1, 1, 1]
    return representation


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
                 quantize=quantize_activation):
        self.quantize = quantize
        self.quantize_len = len(quantize(1))
        super(RealEncoder, self).__init__(n_random_mappings,
                                          input_size,
                                          input_area,
                                          self.quantize_len * automaton_area,
                                          input_offset,
                                          verbose)

    def _overwrite(self, master, second):
        automaton_offset = 0  # Offset index
        for i, r in enumerate(self.random_mappings):
            master_i = i % self.input_size
            second_i = automaton_offset + self.quantize_len * r
            second[second_i:second_i + self.quantize_len] = self.quantize(master[master_i])

            if master_i + 1 == self.input_size:
                automaton_offset += self._automaton_area  # Adjusting offset
        return second

    @staticmethod
    def extend_state_vectors(state_vectors, appendices):
        """
        Extends (in the beginning) each of the state vectors with the corresponding appendix.
        :param state_vectors:
        :param appendices:
        :return:
        """
        extends = []
        for appendix_sequence, state_sequence in zip(appendices, state_vectors):
            sequence = []
            for appendix, state_vector in zip(appendix_sequence, state_sequence):
                extended = appendix.tolist()
                extended.extend(state_vector)
                sequence.append(extended)
            extends.append(sequence)
        return extends


def quantize_japvow(value):
    if value < -0.2501245:
        representation = [0, 0, 1]
    elif value < -0.066513:
        representation = [0, 0, 1]
    elif value < 0.14068925:
        representation = [0, 1, 1]
    else:
        representation = [1, 1, 1]

    # if value < -0.2984444:
    #     representation = [0, 0, 0, 1]
    # elif value < -0.1379516:
    #     representation = [0, 1, 0, 1]
    # elif value < 0.004259:
    #     representation = [0, 1, 1, 0]
    # elif value < 0.2079332:
    #     representation = [1, 0, 0, 1]
    # else:
    #     representation = [1, 0, 0, 0]

    # 14: [0, 0, 0, 1] [0, 1, 0, 1] [0, 1, 1, 0] [1, 0, 0, 1] [1, 0, 0, 0]

    return representation


def quantize_cifar(value):
    if value < 70:
        representation = [1, 1, 1]
    elif value < 117:
        representation = [0, 1, 1]
    elif value < 167:
        representation = [0, 0, 1]
    else:
        representation = [0, 0, 0]
    return representation
