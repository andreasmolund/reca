# Encoder for real valued input.

from encoders.classic import ClassicEncoder


def quantize_japvow(value):
    if value < -0.2501245:
        return [0, 0, 0]
    elif value < -0.066513:
        return [0, 0, 1]
    elif value < 0.14068925:
        return [0, 1, 1]
    else:
        return [1, 1, 1]


def quantize_cifar(value):
    if value < 70:
        return [1, 1, 1]
    elif value < 117:
        return [0, 1, 1]
    elif value < 167:
        return [0, 0, 1]
    else:
        return [0, 0, 0]

quantize = quantize_japvow

quantize_l = len(quantize(1))


class RealEncoder(ClassicEncoder):
    """
    Encoder to use for floating point input.
    What differs this from ClassicEncoder is the translation stage/function.
    This class uses the same type of random mapping,
    however, it must binarize or quantize input elements before mapping them onto automata

    """

    def __init__(self, n_random_mappings, input_size, input_area, automaton_area, input_offset=0, verbose=0):
        super(RealEncoder, self).__init__(n_random_mappings,
                                          input_size,
                                          input_area,
                                          quantize_l * automaton_area,
                                          input_offset,
                                          verbose)

    def _overwrite(self, master, second):
        automaton_offset = 0  # Offset index
        for i, r in enumerate(self.random_mappings):
            master_i = i % self.input_size
            second_i = automaton_offset + quantize_l * r
            second[second_i:second_i + quantize_l] = quantize(master[master_i])

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
