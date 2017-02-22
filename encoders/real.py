# Encoder for real valued input.

from encoders.classic import ClassicEncoder


def quantize(value):
    if value < -0.2501245:
        return [1, 1, 1]
    elif value < -0.066513:
        return [0, 1, 1]
    elif value < 0.14068925:
        return [0, 0, 1]
    else:
        return [0, 0, 0]


class RealEncoder(ClassicEncoder):
    """
    Encoder to use for floating point input.
    What differs this from ClassicEncoder is the translation stage/function.
    This class uses the same type of random mapping,
    however, it must binarize or quantize input elements before mapping them onto automata

    """

    def _overwrite(self, master, second):
        automaton_offset = 0  # Offset index
        for i, r in enumerate(self.random_mappings):
            master_i = i % self.input_size
            second_i = automaton_offset + 3 * r
            second[second_i:second_i + 3] = quantize(master[master_i])

            if master_i + 1 == self.input_size:
                automaton_offset += self._automaton_area  # Adjusting offset
        return second
