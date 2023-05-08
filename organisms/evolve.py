from typing import List, Callable
from organism import Organism
import numpy as np
import encode


def get_organism_fitness(organism: Organism) -> float:
    return organism.energy


def new_organism_from_encoding(encoding: encode.Encoding, string) -> Organism:
    org = Organism()
    org.set_genome(encoding.reverse(string, org))
    return org


class Evolve:

    def __init__(self, encoding: encode.Encoding,
                 selection: Callable[[List[Organism]], List[Organism]],
                 crossover: Callable[[Organism, Organism], List[Organism]],
                 mutation: Callable[[Organism],
                                    Organism], elitism: int) -> None:
        self._encoding = encoding
        self._selection = selection
        self._crossover = crossover
        self._mutation = mutation
        self._elitism = elitism

    def __call__(self, organisms: List[Organism]) -> List[Organism]:
        # 1. Selection
        selected = self._selection(organisms)
        next_population: list[Organism] = []
        # print(f"selection {len(selected)}")

        # 2. Elitism
        next_population.extend(selected[:self._elitism] if self._elitism <
                               len(selected) else selected)
        # print(f"next_population after elitism {next_population}")

        # 3. Crossover
        for i in range(0, len(selected), 2):
            parent_a = selected[i]
            parent_b = selected[i + 1]
            children = self._crossover(self._encoding(parent_a),
                                       self._encoding(parent_b))

            new_organisms = [Organism() for _ in range(len(children))]
            new_organisms = [
                o.add_parent(parent_a).add_parent(parent_b).set_genome_size(
                    parent_a._size) for o in new_organisms
            ]
            new_organisms = [
                new_organisms[i].set_genome(
                    self._encoding.reverse(enc, new_organisms[i]))
                for i, enc in enumerate(children)
            ]
            next_population.extend(new_organisms)

            # next_population.extend([
            #     new_organism_from_encoding(
            #         self._encoding,
            #         enc).set_genome_size(parent_a._size).add_parent(
            #             parent_a).add_parent(parent_b) for enc in children
            # ])

        # print(f"number of organisms after crossover {len(next_population)}")

        # 4. Mutation
        for org in next_population:
            new_genome = self._mutation(self._encoding(org))
            org.set_genome(self._encoding.reverse(new_genome, org))

        return next_population


class TruncationSelection:

    def __init__(self, n: int = 10):
        self.n = n

    def __call__(self, population: List[Organism]) -> List[Organism]:
        # TODO: N as a hyperparameter
        # print(population)
        fitness_values = np.array(
            [get_organism_fitness(x) for x in population])
        top_indices = np.argsort(fitness_values)[-self.n:]
        return [population[ind] for ind in top_indices]


class SBXCrossover:

    def __init__(self, mu: float = 5) -> None:
        self._mu = mu

    def __call__(self, parent_a: np.ndarray,
                 parent_b: np.ndarray) -> List[Organism]:
        # encoding_a = parent_a.()
        # encoding_b = parent_b.encoding()

        offspring_a = np.zeros(parent_a.shape)
        offspring_b = np.zeros(parent_b.shape)

        # for each gene perform SBX crossover
        for i, _ in enumerate(parent_a):
            u = np.random.rand()
            if u <= 0.5:
                beta = (2 * u)**(1 / (self._mu + 1))
            else:
                beta = (1 / (2 * (1 - u)))**(1 / (self._mu + 1))

            # calculate the offspring values
            offspring_a[i] = 0.5 * (((1 + beta) * parent_a[i]) +
                                    ((1 - beta) * parent_b[i]))
            offspring_b[i] = 0.5 * (((1 - beta) * parent_a[i]) +
                                    ((1 + beta) * parent_b[i]))

        return [
            offspring_a,
            offspring_b,
        ]


class GaussianMutation:

    def __init__(self, mu: float, sigma: float, p: float) -> None:
        self._mu = mu
        self._sigma = sigma
        self._p = p

    def __call__(self, individual: np.ndarray) -> Organism:
        """
        Apply Gaussian mutation to an individual's genome.

        Args:
            individual (numpy array): The individual's genome.
            mu (float): The mean of the normal distribution.
            sigma (float): The standard deviation of the normal distribution.
            p (float): The probability of mutation.

        Returns:
            The mutated individual's genome.
        """
        genome_size = individual.shape[0]
        mutated_individual = individual.copy()
        for i in range(genome_size):
            if np.random.random() < self._p:
                mutated_individual[i] += np.random.normal(
                    self._mu, self._sigma)
        return mutated_individual
