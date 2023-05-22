from typing import Any, List, Callable
from organism import Organism
import numpy as np
import encode


def get_organism_fitness(organism: Organism) -> float:
    return organism.energy


def new_organism_from_encoding(encoding: encode.Encoding, string) -> Organism:
    org = Organism()
    org.set_genome(encoding.reverse(string, org))
    return org


class Fitness:

    def __call__(self, organisms: list[Organism]) -> list[float]:
        raise NotImplementedError()


class Selection:
    """
    Selection operator interface
    """

    def __call__(self, population: list[Organism]) -> list[Organism]:
        raise NotImplementedError()


class Crossover:

    def __call__(self, parent_a: np.ndarray,
                 parent_b: np.ndarray) -> list[np.ndarray]:
        raise NotImplementedError()


class Mutation:

    def __call__(self, organism: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class NonUniformMutation(Mutation):

    def __init__(self, b: float, p: float, T: int) -> None:
        self._b = b
        self._p = p
        self._T = T
        self._t = 0

    def set_time(self, t):
        self._t = t

    def __call__(self, individual: np.ndarray) -> np.ndarray:
        t = self._t
        mutated_individual = individual.copy()
        for i in range(individual.shape[0]):
            if np.random.random() < self._p:
                r = np.random.random()
                delta = self._delta(self._T - t, r)
                if np.random.random() < 0.5:
                    mutated_individual[i] += delta * (individual[i] -
                                                      individual[i].min())
                else:
                    mutated_individual[i] -= delta * (individual[i].max() -
                                                      individual[i])
        return mutated_individual

    def _delta(self, t: int, r: float) -> float:
        return r * (1 - t / self._T)**self._b


class Evolve:

    def __init__(self, encoding: encode.Encoding, selection: Selection,
                 crossover: Crossover, mutation: Mutation,
                 elitism: int) -> None:
        self._encoding = encoding
        self._selection = selection
        self._crossover = crossover
        self._mutation = mutation
        self._elitism = elitism

    def __call__(self, organisms: List[Organism], time: int) -> List[Organism]:
        # Case for NonUniform Mutation
        if isinstance(self._mutation, NonUniformMutation):
            self._mutation.set_time(time)

        # 1. Selection
        selected = self._selection(organisms)
        next_population: list[Organism] = []

        # 2. Elitism
        next_population.extend(selected[:self._elitism] if self._elitism <
                               len(selected) else selected)

        # 3. Crossover
        for i in range(0, len(selected), 2):
            i = np.random.randint(0, len(selected))
            j = np.random.randint(0, len(selected))
            parent_a = selected[i]
            parent_b = selected[j]
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

        # 4. Mutation
        for org in next_population:
            new_genome = self._mutation(self._encoding(org))
            org.set_genome(self._encoding.reverse(new_genome, org))

        return next_population


class FoodConsumnptionToDistanceFitness(Fitness):

    def __call__(self, organisms: list[Organism]) -> list[float]:

        def fitness(organism: Organism):
            return organism.consumed_food_energy / organism.distance_traveled

        return list(map(fitness, organisms))


class EnergyFitness(Fitness):

    def __call__(self, organisms: list[Organism]) -> list[float]:

        def fitness(organism: Organism):
            return organism.energy

        return list(map(fitness, organisms))


class TruncationSelection(Selection):

    def __init__(self, fitness: Fitness, n: int = 10):
        self.n = n
        self.fitness = fitness

    def __call__(self, population: List[Organism]) -> List[Organism]:
        # TODO: N as a hyperparameter
        # fitness_values = np.array(
        #     [self.fitness(x) for x in population])
        fitness_values = np.array(self.fitness(population))
        top_indices = np.argsort(fitness_values)[-self.n:]
        return [population[ind] for ind in top_indices]


class SBXCrossover(Crossover):

    def __init__(self, n: float = 5) -> None:
        self.n = n

    def __call__(self, parent_a: np.ndarray,
                 parent_b: np.ndarray) -> List[np.ndarray]:
        offspring_a = np.zeros(parent_a.shape)
        offspring_b = np.zeros(parent_b.shape)

        # for each gene perform SBX crossover
        for i, _ in enumerate(parent_a):
            u = np.random.rand()
            if u <= 0.5:
                # beta = (2 * u)**(1 / (self._mu + 1))
                beta = np.power(2 * u, 1 / (self.n + 1))
            else:
                # beta = (1 / (2 * (1 - u)))**(1 / (self.n + 1))
                beta = 1 / np.power(-2 * u, self.n + 1)

            # calculate the offspring values
            offspring_a[i] = 0.5 * ((parent_a[i] + parent_b[i]) -
                                    beta * np.abs(parent_b[i] - parent_a[i]))
            offspring_b[i] = 0.5 * ((parent_a[i] + parent_b[i]) +
                                    beta * np.abs(parent_b[i] - parent_a[i]))
            # offspring_a[i] = 0.5 * (((1 + beta) * parent_a[i]) +
            #                         ((1 - beta) * parent_b[i]))
            # offspring_b[i] = 0.5 * (((1 - beta) * parent_a[i]) +
            #                         ((1 + beta) * parent_b[i]))

        return [
            offspring_a,
            offspring_b,
        ]


class BLXCrossover(Crossover):

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha

    def __call__(self, parent_a: np.ndarray,
                 parent_b: np.ndarray) -> List[np.ndarray]:
        offspring_a = np.zeros(parent_a.shape)
        offspring_b = np.zeros(parent_b.shape)

        # for each gene perform BLX crossover
        for i in range(parent_a.shape[0]):
            # Calculate the interval difference
            d = np.abs(parent_a[i] - parent_b[i])
            I = self.alpha * d

            # Create offspring within range [min(a,b) - I, max(a,b) + I]
            min_val = min(parent_a[i], parent_b[i]) - I
            max_val = max(parent_a[i], parent_b[i]) + I

            offspring_a[i] = np.random.uniform(low=min_val, high=max_val)
            offspring_b[i] = np.random.uniform(low=min_val, high=max_val)

        return [offspring_a, offspring_b]


class ArithmeticCrossover(Crossover):

    def __call__(self, parent_a: np.ndarray,
                 parent_b: np.ndarray) -> List[np.ndarray]:
        offspring_a = np.zeros(parent_a.shape)
        offspring_b = np.zeros(parent_b.shape)

        # Randomly generate a weight
        rand_weight = np.random.rand()

        # for each gene perform Arithmetic Crossover
        for i in range(parent_a.shape[0]):
            offspring_a[i] = rand_weight * parent_a[i] + (
                1 - rand_weight) * parent_b[i]
            offspring_b[i] = rand_weight * parent_b[i] + (
                1 - rand_weight) * parent_a[i]

        return [offspring_a, offspring_b]


class UniformCrossover(Crossover):

    def __call__(self, parent_a: np.ndarray,
                 parent_b: np.ndarray) -> List[np.ndarray]:
        offspring_a = np.zeros(parent_a.shape)
        offspring_b = np.zeros(parent_b.shape)

        # for each gene perform Uniform Crossover
        for i in range(parent_a.shape[0]):
            if np.random.rand() < 0.5:
                offspring_a[i] = parent_a[i]
                offspring_b[i] = parent_b[i]
            else:
                offspring_a[i] = parent_b[i]
                offspring_b[i] = parent_a[i]

        return [offspring_a, offspring_b]


class GaussianMutation(Mutation):

    def __init__(self, mu: float, sigma: float, p: float) -> None:
        self._mu = mu
        self._sigma = sigma
        self._p = p

    def __call__(self, individual: np.ndarray) -> np.ndarray:
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


class UniformMutation(Mutation):

    def __init__(self, low: float, high: float, p: float) -> None:
        self._low = low
        self._high = high
        self._p = p

    def __call__(self, individual: np.ndarray) -> np.ndarray:
        mutated_individual = individual.copy()
        for i in range(individual.shape[0]):
            if np.random.random() < self._p:
                mutated_individual[i] += np.random.uniform(
                    self._low, self._high)
        return mutated_individual
