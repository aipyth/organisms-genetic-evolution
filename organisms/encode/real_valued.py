import numpy as np
from organism import Organism


class RealValued:
    """A real-valued encoding for an organism's genome.

    This class implements a real-valued encoding for an organism's genome,
    which concatenates the values of the genome's elements along the second axis.

    Attributes:
        None

    Methods:
        __call__: Returns a real-valued array representing the organism's genome.

    Usage:
        encoding = ReadValued()
        genome_array = encoding(organism)

    """

    def __call__(self, organism: Organism) -> np.ndarray:
        """Return a real-valued array representing the organism's genome.

        Args:
            organism (Organism): An organism object containing a genome.

        Returns:
            np.ndarray: A real-valued array representing the organism's genome.

        """
        return np.concatenate(organism.genome, axis=1)

    @staticmethod
    def reverse(encoding: np.ndarray, organism: Organism) -> np.ndarray:
        genome = []
        splits = np.split(
            encoding,
            np.cumsum(np.multiply(organism._size[:-1], organism._size[1:])))

        for i, s in enumerate(splits[::2]):
            genome.append((
                s.reshape((organism._size[i], organism._size[i + 1])),
                s.reshape((organism._size[i + 1], 1)),
            ))
        return genome
