import random
import numpy as np
import functools
import utils


class Organism:

    def __init__(
        self,
        r: float = 0.0,
        genome: list[np.ndarray] = [],
    ):
        # self._x = np.array([0, 0])
        self.x = np.array([0, 0])
        self.r = r
        self.v = 0
        self.a = 0
        self.genome = genome
        self._activation_function = np.tanh
        self._energy = 5
        # self._energy = 25
        self._age = 0
        self._size = None
        self._name = utils.generate_name(random.randint(1, 4))
        self._parents = []

    def __str__(self):
        return f"<Organism {[g.shape for g in self.genome]}>"

    def add_parent(self, organism):
        self._parents.append(organism)
        return self

    @property
    def name(self):
        return self._name

    @property
    def weights(self):
        return list(map(lambda x: x[0], self.genome))

    @property
    def biases(self):
        return list(map(lambda x: x[1], self.genome))

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, value):
        """Increases the amount of energy in the organism."""
        self._energy += max(value, 0)

    def set_genome(self, genome: list[tuple[np.ndarray, np.ndarray]]):
        self.genome = genome
        return self

    def set_genome_size(self, size: list[int]):
        self._size = size
        if self.genome is None or len(self.genome) == 0:
            mu = 0
            sigma = 1
            # initialize weights and bias
            self.genome = [(
                np.random.normal(mu, sigma, [size[i], size[i + 1]]),
                np.random.normal(mu, sigma, (size[i + 1], 1)),
            ) for i in range(len(size) - 1)]
        return self

    def set_activation_function(self, func):
        self._activation_function = func

    @property
    def is_alive(self):
        return self._energy > -1

    @property
    def age(self):
        return self._age

    def evaluate(self, x):
        movement = functools.reduce(
            lambda z, p: self._activation_function(p[0].T @ z + p[1]),
            self.genome, x)
        # TODO: in case of low energy the possible movement is restricted
        # TODO: energy is wasted by the amount of movement made
        # though the movement is mapped to environment bounds and physics
        # we will neglect this fact for now and assume that the energy
        # is wasted by fact of organism's thinking and commands
        decrease_by = 0.08
        # self._energy = self._energy - decrease_by if self.is_alive else self._energy
        self._energy = self._energy - decrease_by
        self._age += 1
        # return movement * (self._energy > 0)
        return movement


if __name__ == "__main__":
    org1 = Organism()
    org1.set_genome_size([2, 2, 4, 10])
    print(org1)
    x = np.array([0.1, 0.2]).reshape((2, 1))
    print(f"{x=}")
    print(org1.evaluate(x))
