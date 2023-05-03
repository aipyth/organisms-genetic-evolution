from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
    from organism import Organism


class Encoding:

    def __call__(self, organism: Organism) -> np.ndarray:
        pass

    def reverse(self, encoding: np.ndarray,
                organism: Organism) -> list[tuple[np.ndarray, np.ndarray]]:
        pass


class RealValued(Encoding):
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
        flattened = []
        for w, b in zip(organism.weights, organism.biases):
            flattened.extend([
                w.flatten(),
                b.flatten(),
            ])
        return np.concatenate(flattened)

    def reverse(self, encoding: np.ndarray,
                organism: Organism) -> list[tuple[np.ndarray, np.ndarray]]:
        weights = []
        biases = []
        sizes = organism._size
        index = 0
        for i in range(len(sizes) - 1):
            w_size = sizes[i] * sizes[i + 1]
            b_size = sizes[i + 1]
            weights.append(encoding[index:index + w_size].reshape(
                (sizes[i], sizes[i + 1])))
            index += w_size
            biases.append(encoding[index:index + b_size].reshape(
                (sizes[i + 1], 1)))
            index += b_size

        return list(zip(weights, biases))
        # genome = []
        # print({
        #     'encoding':
        #     encoding,
        #     'multiply':
        #     np.multiply(organism._size[:-1], organism._size[1:]),
        #     'cumsum':
        #     np.cumsum(np.multiply(organism._size[:-1], organism._size[1:])),
        # })
        # splits = np.split(
        #     encoding,
        #     np.cumsum(np.multiply(organism._size[:-1], organism._size[1:])))
        # print(len(splits), splits)

        # for i, s in enumerate(splits[::2]):
        #     genome.append((
        #         s.reshape((organism._size[i], organism._size[i + 1])),
        #         splits[i + 1].reshape((organism._size[i + 1], 1)),
        #     ))
        # return genome
