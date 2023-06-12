import encodings
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

    def __call__(self, parent_a: Organism,
                 parent_b: Organism) -> list[Organism]:
        raise NotImplementedError()

    def use_encoding(self, encoding: encode.Encoding):
        self._encoding = encoding

    def _create_organisms_from_encoding(self, offsprings, parent_a: Organism,
                                        parent_b: Organism) -> list[Organism]:
        new_organisms = [Organism() for _ in range(len(offsprings))]
        new_organisms = [
            o.add_parent(parent_a).add_parent(parent_b).set_genome_size(
                parent_a._size) for o in new_organisms
        ]
        new_organisms = [
            new_organisms[i].set_genome(
                self._encoding.reverse(enc, new_organisms[i]))
            for i, enc in enumerate(offsprings)
        ]
        return new_organisms


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


class Evolve:

    def __init__(self, encoding: encode.Encoding, selection: Selection,
                 crossover: Crossover, mutation: Mutation,
                 elitism: int) -> None:
        self._encoding = encoding
        self._selection = selection
        self._crossover = crossover
        self._crossover.use_encoding(self._encoding)
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
        elites_selector = TruncationSelection(EnergyFitness(), self._elitism)
        elites = elites_selector(organisms)
        next_population.extend(elites)
        # next_population.extend(selected[:self._elitism] if self._elitism <
        #                        len(selected) else selected)

        # 3. Crossover
        for i in range(0, len(selected), 2):
            i = np.random.randint(0, len(selected))
            j = np.random.randint(0, len(selected))
            parent_a = selected[i]
            parent_b = selected[j]
            children = self._crossover(parent_a, parent_b)
            next_population.extend(children)

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


class SBXCrossover(Crossover):

    def __init__(self, n: float = 5) -> None:
        self.n = n

    def __call__(self, parent_a: Organism,
                 parent_b: Organism) -> list[Organism]:
        encoding_a = self._encoding(parent_a)
        encoding_b = self._encoding(parent_b)
        offspring_a = np.zeros(encoding_a.shape)
        offspring_b = np.zeros(encoding_b.shape)

        # for each gene perform SBX crossover
        for i, _ in enumerate(encoding_a):
            u = np.random.rand()
            if u <= 0.5:
                # beta = (2 * u)**(1 / (self._mu + 1))
                beta = np.power(2 * u, 1 / (self.n + 1))
            else:
                # beta = (1 / (2 * (1 - u)))**(1 / (self.n + 1))
                beta = 1 / np.power(-2 * u, self.n + 1)

            # calculate the offspring values
            offspring_a[i] = 0.5 * (
                (encoding_a[i] + encoding_b[i]) -
                beta * np.abs(encoding_b[i] - encoding_a[i]))
            offspring_b[i] = 0.5 * (
                (encoding_a[i] + encoding_b[i]) +
                beta * np.abs(encoding_b[i] - encoding_a[i]))
            # offspring_a[i] = 0.5 * (((1 + beta) * parent_a[i]) +
            #                         ((1 - beta) * parent_b[i]))
            # offspring_b[i] = 0.5 * (((1 - beta) * parent_a[i]) +
            #                         ((1 + beta) * parent_b[i]))

        return self._create_organisms_from_encoding([
            offspring_a,
            offspring_b,
        ], parent_a, parent_b)


class SASBXCrossover(Crossover):

    def __init__(self, alpha: float = 1.5, init_n: float = 2) -> None:
        self.alpha = alpha
        self.n = init_n
        self.decider = FoodConsumnptionToDistanceFitness()
        self._lower_eta_bound = 1
        self._upper_eta_bound = 10

    def is_better_than_parents(self, child: Organism):
        if len(child._parents) < 2:
            return False
        parent_a = child._parents[0]
        parent_b = child._parents[1]
        evaluations = self.decider([parent_a, parent_b, child])
        return (evaluations[2] > evaluations[0]
                and evaluations[2] > evaluations[1])

    def correct_eta(self, child, parent_a, parent_b, alpha):
        encoding_c = self._encoding(child)
        encoding_pa = self._encoding(parent_a)
        encoding_pb = self._encoding(parent_b)
        parent_enc_diff = encoding_pb - encoding_pa
        parent_enc_diff[parent_enc_diff == 0] = 1 << 32
        beta = np.linalg.norm(1 +
                              (2 *
                               (encoding_c - encoding_pb)) / parent_enc_diff)
        if beta > 1:
            eta_s = -1 + (child.eta + 1) * np.log(beta) / np.log(1 + alpha *
                                                                 (beta - 1))
        else:
            eta_s = (1 + child.eta) / alpha - 1
        # eta_s[eta_s < self._lower_eta_bound] = self._lower_eta_bound
        # eta_s[eta_s > self._upper_eta_bound] = self._upper_eta_bound
        eta_s = min(self._upper_eta_bound, max(self._lower_eta_bound, eta_s))
        return eta_s

    def correct_eta_with_u(self, child, parent_a, parent_b, alpha, u):
        encoding_c = self._encoding(child)
        encoding_cs = self._encoding(child.sibling)
        encoding_pa = self._encoding(parent_a)
        encoding_pb = self._encoding(parent_b)
        if np.linalg.norm(encoding_c -
                          encoding_pa) < np.linalg.norm(encoding_c -
                                                        encoding_pb):
            encoding_pn = encoding_pa
            encoding_pd = encoding_pb
        else:
            encoding_pn = encoding_pb
            encoding_pd = encoding_pa
        parent_enc_diff = encoding_pd - encoding_pn
        parent_enc_diff[parent_enc_diff == 0] = 1 << 32
        # beta = np.linalg.norm(
        #     1 + (2 * (encoding_c - encoding_pn)) / parent_enc_diff)
        beta = np.mean(1 + (2 * (encoding_c - encoding_pn)) / parent_enc_diff)
        siblings_enc_diff = encoding_cs - encoding_c
        siblings_enc_diff[siblings_enc_diff == 0] = 1 << 32
        # beta_a = np.linalg.norm(
        #     1 + (alpha * (beta - 1) * parent_enc_diff) / siblings_enc_diff)
        beta_a = np.mean(1 + (2 * alpha *
                              (encoding_cs - encoding_pn)) / siblings_enc_diff)
        # print('beta, beta_a', beta, beta_a)
        if beta > 1:
            eta_s = -1 - np.log(2 * (1 - u)) / np.log(beta_a)
            # eta_s = -1 - np.log(beta_a) / np.log(2 * (1 - u))
        else:
            eta_s = -1 + np.log(2 * u) / np.log(beta_a)
            # eta_s = -1 + np.log(beta_a) / np.log(2 * u)
            # eta_s=(1 + child.eta) / alpha - 1
        # eta_s[eta_s < self._lower_eta_bound] = self._lower_eta_bound
        # eta_s[eta_s > self._upper_eta_bound] = self._upper_eta_bound
        # print('eta after correction', eta_s)
        eta_s = min(self._upper_eta_bound, max(self._lower_eta_bound, eta_s))
        return eta_s

    def __call__(self, parent_a: Organism,
                 parent_b: Organism) -> list[Organism]:

        encoding_a = self._encoding(parent_a)
        encoding_b = self._encoding(parent_b)
        offspring_a = np.zeros(encoding_a.shape)
        offspring_b = np.zeros(encoding_b.shape)

        # get the offspring's etas
        if parent_a.eta is None:
            # parent_a.eta = np.full(encoding_a.shape, self.n)
            parent_a.eta = self.n
        if parent_b.eta is None:
            # parent_b.eta = np.full(encoding_b.shape, self.n)
            parent_b.eta = self.n
        # eta_a = parent_a.eta.copy()
        # eta_b = parent_b.eta.copy()
        eta_a = parent_a.eta
        eta_b = parent_b.eta
        # print('eta a, b:', eta_a, eta_b)
        eta = (eta_a + eta_b) / 2

        # decide if the offsprings evaluated better than their parents
        # if self.is_better_than_parents(parent_a):
        #     eta_a = self.correct_eta(
        #         parent_a, parent_a._parents[0], parent_a._parents[1], self.alpha)
        # if self.is_better_than_parents(parent_b):
        #     eta_b = self.correct_eta(
        #         parent_b, parent_b._parents[0], parent_b._parents[1], self.alpha)
        # if len(parent_a._parents) == 2:
        #     eta_a = self.correct_eta(
        #         parent_a, parent_a._parents[0], parent_a._parents[1], self.alpha if
        #         self.is_better_than_parents(parent_a) else 1 / self.alpha)
        # if len(parent_b._parents) == 2:
        #     eta_b = self.correct_eta(
        #         parent_b, parent_b._parents[0], parent_b._parents[1], self.alpha if
        #         self.is_better_than_parents(parent_a) else 1 / self.alpha)
        #     print(eta_b)

        # eta = np.zeros(encoding_a.shape)
        # for i, _ in enumerate(encoding_a.shape):
        #     eta[i] = eta_a[i] if np.random.random() > 0.5 else eta_b[i]

        # for each gene perform SBX crossover
        for i, _ in enumerate(encoding_a):
            u = np.random.rand()
            parent = parent_a if np.random.random() > 0.5 else parent_b
            if len(parent._parents) == 2:
                eta = self.correct_eta_with_u(
                    parent, parent._parents[0], parent._parents[1],
                    self.alpha if self.is_better_than_parents(parent) else 1 /
                    self.alpha, u)
            if u <= 0.5:
                beta = np.power(2 * u, 1 / (eta + 1))
            else:
                beta = np.power(1 / (2 * (1 - u)), 1 / (eta + 1))

            # calculate the offspring values
            offspring_a[i] = 0.5 * (((1 + beta) * encoding_a[i]) +
                                    ((1 - beta) * encoding_b[i]))
            offspring_b[i] = 0.5 * (((1 - beta) * encoding_a[i]) +
                                    ((1 + beta) * encoding_b[i]))

        new_organisms = self._create_organisms_from_encoding([
            offspring_a,
            offspring_b,
        ], parent_a, parent_b)
        new_organisms[0].sibling = new_organisms[1]
        new_organisms[1].sibling = new_organisms[0]
        return new_organisms


class SimpleAdaptiveSBXCrossover(Crossover):

    def __init__(self, alpha: float = 1.1, init_n: float = 2) -> None:
        self.alpha = alpha
        self.n = init_n
        self.decider = FoodConsumnptionToDistanceFitness()
        self._lower_eta_bound = 1
        self._upper_eta_bound = 10

    def is_better_than_parents(self, child: Organism):
        if len(child._parents) < 2:
            return False
        parent_a = child._parents[0]
        parent_b = child._parents[1]
        evaluations = self.decider([parent_a, parent_b, child])
        return (evaluations[2] > evaluations[0]
                and evaluations[2] > evaluations[1])

    def correct_eta(self, child, parent_a, parent_b, alpha):
        encoding_c = self._encoding(child)
        encoding_pa = self._encoding(parent_a)
        encoding_pb = self._encoding(parent_b)
        parent_enc_diff = encoding_pb - encoding_pa
        parent_enc_diff[parent_enc_diff == 0] = 1 << 32
        beta = np.linalg.norm(1 +
                              (2 *
                               (encoding_c - encoding_pb)) / parent_enc_diff)
        if beta > 1:
            eta_s = alpha * child.eta
            # eta_s = -1 + (child.eta + 1) * np.log(beta) / np.log(
            #     1 + alpha * (beta - 1)
            # )
        else:
            eta_s = alpha * child.eta
            # eta_s = (1 + child.eta) / alpha - 1
        # eta_s[eta_s < self._lower_eta_bound] = self._lower_eta_bound
        # eta_s[eta_s > self._upper_eta_bound] = self._upper_eta_bound
        eta_s = min(self._upper_eta_bound, max(self._lower_eta_bound, eta_s))
        return eta_s

    def correct_eta_with_u(self, child, parent_a, parent_b, alpha, u):
        encoding_c = self._encoding(child)
        encoding_cs = self._encoding(child.sibling)
        encoding_pa = self._encoding(parent_a)
        encoding_pb = self._encoding(parent_b)
        if np.linalg.norm(encoding_c -
                          encoding_pa) < np.linalg.norm(encoding_c -
                                                        encoding_pb):
            encoding_pn = encoding_pa
            encoding_pd = encoding_pb
        else:
            encoding_pn = encoding_pb
            encoding_pd = encoding_pa
        parent_enc_diff = encoding_pd - encoding_pn
        parent_enc_diff[parent_enc_diff == 0] = 1 << 32
        beta = np.linalg.norm(1 +
                              (2 *
                               (encoding_c - encoding_pn)) / parent_enc_diff)
        siblings_enc_diff = encoding_cs - encoding_c
        siblings_enc_diff[siblings_enc_diff == 0] = 1 << 32
        # beta_a = np.linalg.norm(
        #     1 + (alpha * (beta - 1) * parent_enc_diff) / siblings_enc_diff)
        beta_a = np.linalg.norm(1 + (2 * alpha * (encoding_cs - encoding_pn)) /
                                siblings_enc_diff)
        # print('beta, beta_a', beta, beta_a)
        if beta > 1:
            eta_s = -1 - np.log(2 * (1 - u)) / np.log(beta_a)
            # eta_s = -1 - np.log(beta_a) / np.log(2 * (1 - u))
        else:
            eta_s = -1 + np.log(2 * u) / np.log(beta_a)
            # eta_s = -1 + np.log(beta_a) / np.log(2 * u)
            # eta_s=(1 + child.eta) / alpha - 1
        # eta_s[eta_s < self._lower_eta_bound] = self._lower_eta_bound
        # eta_s[eta_s > self._upper_eta_bound] = self._upper_eta_bound
        # print('eta after correction', eta_s)
        eta_s = min(self._upper_eta_bound, max(self._lower_eta_bound, eta_s))
        return eta_s

    def __call__(self, parent_a: Organism,
                 parent_b: Organism) -> list[Organism]:

        encoding_a = self._encoding(parent_a)
        encoding_b = self._encoding(parent_b)
        offspring_a = np.zeros(encoding_a.shape)
        offspring_b = np.zeros(encoding_b.shape)

        # get the offspring's etas
        if parent_a.eta is None:
            # parent_a.eta = np.full(encoding_a.shape, self.n)
            parent_a.eta = self.n
        if parent_b.eta is None:
            # parent_b.eta = np.full(encoding_b.shape, self.n)
            parent_b.eta = self.n
        # eta_a = parent_a.eta.copy()
        # eta_b = parent_b.eta.copy()
        eta_a = parent_a.eta
        eta_b = parent_b.eta
        # print('eta a, b:', eta_a, eta_b)
        eta = (eta_a + eta_b) / 2

        # decide if the offsprings evaluated better than their parents
        # if self.is_better_than_parents(parent_a):
        #     eta_a = self.correct_eta(
        #         parent_a, parent_a._parents[0], parent_a._parents[1], self.alpha)
        # if self.is_better_than_parents(parent_b):
        #     eta_b = self.correct_eta(
        #         parent_b, parent_b._parents[0], parent_b._parents[1], self.alpha)
        if len(parent_a._parents) == 2:
            eta_a = self.correct_eta(
                parent_a, parent_a._parents[0], parent_a._parents[1],
                self.alpha if self.is_better_than_parents(parent_a) else 1 /
                self.alpha)
        if len(parent_b._parents) == 2:
            eta_b = self.correct_eta(
                parent_b, parent_b._parents[0], parent_b._parents[1],
                self.alpha if self.is_better_than_parents(parent_a) else 1 /
                self.alpha)

        eta = (eta_a + eta_b) / 2

        # eta = np.zeros(encoding_a.shape)
        # for i, _ in enumerate(encoding_a.shape):
        #     eta[i] = eta_a[i] if np.random.random() > 0.5 else eta_b[i]

        # for each gene perform SBX crossover
        for i, _ in enumerate(encoding_a):
            u = np.random.rand()
            # parent = parent_a if np.random.random() > 0.5 else parent_b
            # if len(parent._parents) == 2:
            #     eta = self.correct_eta_with_u(
            #         parent,
            #         parent._parents[0],
            #         parent._parents[1],
            #         self.alpha if self.is_better_than_parents(parent)
            #         else 1 / self.alpha,
            #         u
            #     )
            if u <= 0.5:
                beta = np.power(2 * u, 1 / (eta + 1))
            else:
                beta = np.power(1 / (2 * (1 - u)), 1 / (eta + 1))

            # calculate the offspring values
            offspring_a[i] = 0.5 * (((1 + beta) * encoding_a[i]) +
                                    ((1 - beta) * encoding_b[i]))
            offspring_b[i] = 0.5 * (((1 - beta) * encoding_a[i]) +
                                    ((1 + beta) * encoding_b[i]))

        new_organisms = self._create_organisms_from_encoding([
            offspring_a,
            offspring_b,
        ], parent_a, parent_b)
        new_organisms[0].sibling = new_organisms[1]
        new_organisms[1].sibling = new_organisms[0]
        return new_organisms


class BLXCrossover(Crossover):

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha

    def __call__(self, parent_a: np.ndarray,
                 parent_b: np.ndarray) -> List[np.ndarray]:
        encoding_a = self._encoding(parent_a)
        encoding_b = self._encoding(parent_b)
        offspring_a = np.zeros(encoding_a.shape)
        offspring_b = np.zeros(encoding_b.shape)

        # for each gene perform BLX crossover
        for i in range(encoding_a.shape[0]):
            # Calculate the interval difference
            d = np.abs(encoding_a[i] - encoding_b[i])
            I = self.alpha * d

            # Create offspring within range [min(a,b) - I, max(a,b) + I]
            min_val = min(encoding_a[i], encoding_b[i]) - I
            max_val = max(encoding_a[i], encoding_b[i]) + I

            offspring_a[i] = np.random.uniform(low=min_val, high=max_val)
            offspring_b[i] = np.random.uniform(low=min_val, high=max_val)

        return self._create_organisms_from_encoding([offspring_a, offspring_b],
                                                    parent_a, parent_b)


class ArithmeticCrossover(Crossover):

    def __call__(self, parent_a: np.ndarray,
                 parent_b: np.ndarray) -> List[np.ndarray]:
        encoding_a = self._encoding(parent_a)
        encoding_b = self._encoding(parent_b)
        offspring_a = np.zeros(encoding_a.shape)
        offspring_b = np.zeros(encoding_b.shape)

        # Randomly generate a weight
        rand_weight = np.random.rand()

        # for each gene perform Arithmetic Crossover
        for i in range(encoding_a.shape[0]):
            offspring_a[i] = rand_weight * encoding_a[i] + (
                1 - rand_weight) * encoding_b[i]
            offspring_b[i] = rand_weight * encoding_b[i] + (
                1 - rand_weight) * encoding_a[i]

        return self._create_organisms_from_encoding([offspring_a, offspring_b],
                                                    parent_a, parent_b)


class UniformCrossover(Crossover):

    def __call__(self, parent_a: np.ndarray,
                 parent_b: np.ndarray) -> List[np.ndarray]:
        encoding_a = self._encoding(parent_a)
        encoding_b = self._encoding(parent_b)
        offspring_a = np.zeros(encoding_a.shape)
        offspring_b = np.zeros(encoding_b.shape)

        # for each gene perform Uniform Crossover
        for i in range(encoding_a.shape[0]):
            if np.random.rand() < 0.5:
                offspring_a[i] = encoding_a[i]
                offspring_b[i] = encoding_b[i]
            else:
                offspring_a[i] = encoding_b[i]
                offspring_b[i] = encoding_a[i]

        return self._create_organisms_from_encoding([offspring_a, offspring_b],
                                                    parent_a, parent_b)


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
