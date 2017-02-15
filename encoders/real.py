# Encoder for real valued input.

from encoders.classic import ClassicEncoder
from reservoir.util import binarize


class RealEncoder(ClassicEncoder):
    """
    Encoder to use for floating point input.
    What differs this from ClassicEncoder is the translation stage/function.
    This class uses the same type of random mapping,
    however, it must binarize input elements before mapping them onto automata

    """

    def overwrite(self, master, second):
        mapped = super(RealEncoder, self).overwrite(master, second)
        binarized = binarize(mapped)
        return binarized
