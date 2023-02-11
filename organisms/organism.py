import numpy as np
import functools


class Organism:

    def __init__(self, genome: list[np.ndarray] | None = None):
        self.genome = genome or []
        self._activation_function = np.tanh

    def __str__(self):
        return f'<Organism {[g.shape for g in self.genome]}>'

    def set_genome_size(self, size: list[int]):
        mu = 0
        sigma = 1
        # initialize weights and bias
        self.genome = [(np.random.normal(mu, sigma, [size[i], size[i + 1]]),
                        np.random.normal(mu, sigma, (size[i + 1], 1)))
                       for i in range(len(size) - 1)]

    def set_activation_function(self, func):
        self._activation_function = func

    def evaluate(self, x):
        return functools.reduce(
            lambda z, p: self._activation_function(p[0].T @ z + p[1]),
            self.genome, x)


def main():
    org1 = Organism()
    org1.set_genome_size([2, 2, 4, 10])
    print(org1)
    x = np.array([0.1, 0.2]).reshape((2, 1))
    print(f'{x=}')
    print(org1.evaluate(x))


if __name__ == '__main__':
    main()
