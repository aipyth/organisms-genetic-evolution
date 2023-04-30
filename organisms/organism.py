import numpy as np
import functools


class Organism:

    def __init__(self,
                 x: float | None = None,
                 y: float | None = None,
                 r: float | None = None,
                 genome: list[np.ndarray] | None = None):
        self.x = x
        self.y = y
        self.r = r
        self.genome = genome or []
        self._activation_function = np.tanh
        self._energy = 25
        self._age = 0

    def __str__(self):
        return f'<Organism {[g.shape for g in self.genome]}>'

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, value):
        """Increases the amount of energy in the organism."""
        self._energy += value if value > 0 else 0

    def set_genome_size(self, size: list[int]):
        mu = 0
        sigma = 1
        # initialize weights and bias
        self.genome = [(np.random.normal(mu, sigma, [size[i], size[i + 1]]),
                        np.random.normal(mu, sigma, (size[i + 1], 1)))
                       for i in range(len(size) - 1)]

    def set_activation_function(self, func):
        self._activation_function = func

    @property
    def is_alive(self):
        return self._energy > 0

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
        decrease_by = 1
        self._energy = self._energy - decrease_by if self.is_alive else self._energy
        return movement * (self._energy > 0)


if __name__ == '__main__':
    org1 = Organism()
    org1.set_genome_size([2, 2, 4, 10])
    print(org1)
    x = np.array([0.1, 0.2]).reshape((2, 1))
    print(f'{x=}')
    print(org1.evaluate(x))
