from typing import Any, List
import numpy as np
from rtree import index
from organism import Organism

MAX_VELOCITY = 0.01
MAX_DTHETA = 0.2


class Vision:

    @property
    def organism_input_shape(self):
        raise NotImplementedError()

    def __call__(self, target: Organism, organisms: list[Organism],
                 food: list[tuple[float, float]]) -> np.ndarray:
        raise NotImplementedError()


class Simple2DContinuousEnvironment:

    def __init__(self,
                 width: int,
                 height: int,
                 vision: Vision,
                 vision_range: float = 1,
                 organism_size: float = 0.01,
                 food_size: float = 0.03,
                 food_energy: float = 10,
                 food_appearance_prob: float = 0.2):
        self.width = width
        self.height = height
        self.organisms: dict[int, Organism] = {}
        self._organism_counter = 0
        self.food: dict[int, tuple[float, float]] = {}
        self._food_counter = 0
        self.organism_idx = index.Index()
        self.food_idx = index.Index()
        self._vision = vision
        self.food_energy = food_energy
        self.food_size = food_size
        self.organism_size = organism_size
        self._food_appearance_prob = food_appearance_prob
        self.vision_range = vision_range

    def add_food(self):
        # generate random location for the food
        x = np.random.uniform(0, self.width)
        y = np.random.uniform(0, self.height)
        # add the food to the environment
        self.food[self._food_counter] = (x, y)
        # insert into the rtree
        self.food_idx.insert(self._food_counter, (x, y, x, y))
        self._food_counter += 1

    def remove_food(self, food: tuple[float, float]):
        for i, (x, y) in self.food.items():
            if (x, y) == food:
                # self.food.pop(i)
                del self.food[i]
                self.food_idx.delete(i, (x, y, x, y))
                break

    def detect_collision(self, organism):
        """Check for collision with food and increase energy if collision occurs."""
        # get organism's coordinates
        o = next(o for i, o in self.organisms.items() if o == organism)
        pos = o.x
        # get all food in a region
        foods = [
            food
            for food in self.food_idx.intersection((o.x[0] - self.vision_range,
                                                    o.x[1] - self.vision_range,
                                                    o.x[0] + self.vision_range,
                                                    o.x[1] +
                                                    self.vision_range))
        ]
        for i in foods:
            if np.linalg.norm(np.array(self.food[i]) -
                              o.x) <= self.food_size + self.organism_size:
                organism.energy += self.food_energy
                # print('organism ate a piece of food')
                self.remove_food(self.food[i])

    def add_organism(self,
                     organism,
                     x: float | None = None,
                     y: float | None = None):
        if x is None or y is None:
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)
        self.organisms[self._organism_counter] = organism
        organism.x = np.array([x, y])
        self.organism_idx.insert(self._organism_counter, (x, y, x, y))
        self._organism_counter += 1
        # print('added new organism')

    def remove_organism(self, organism):
        # TODO: remove by id (which removes cycle to find the id)
        for i, o in self.organisms.items():
            if o is organism:
                # self.organisms.pop(i)
                del self.organisms[i]
                # print('removed from the list')
                self.organism_idx.delete(i, (o.x[0], o.x[1], o.x[0], o.x[1]))
                # print('removed from index tree')
                break

    def organism_result_to_coordinates(
            self, organism, result: np.ndarray) -> tuple[float, float]:
        """
        This function takes in an organism and a result which is a numpy array.
        It then updates the coordinates of the organism and returns a tuple of
        floats that represent the updated coordinates for the organism's
        position.

        Args:
            organism: An object representing the organism whose coordinates are
                      to be updated.
            result: A numpy array consisting of two floats representing
                    the change in the coordinates for the organism.


        Returns:
            A tuple of two floats representing the updated x and y coordinates
            for the organism's position.

        Raises:
            No specific exceptions are raised.
        """
        i, o = next((i, o) for i, o in self.organisms.items() if o is organism)
        x = organism.x[0]
        y = organism.x[1]
        self.organism_idx.delete(i, (x, y, x, y))
        v = organism.v
        a = organism.a
        theta = organism.r
        dtheta, da = result.reshape((2, ))
        dtheta = max(-MAX_DTHETA, min(MAX_DTHETA, dtheta))
        a = a + da
        v = v + a
        v = max(-MAX_VELOCITY, min(MAX_VELOCITY, v))
        theta = theta + dtheta
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        x += dx
        y += dy
        x = np.clip(x, 0, self.width)
        y = np.clip(y, 0, self.height)
        organism.x = np.array([x, y]).reshape((2, ))
        organism.r = theta
        organism.v = v
        organism.a = a
        self.organism_idx.insert(i, (x, y, x, y))
        return x, y

    def to_organism_input(self, organism: Organism) -> np.ndarray:
        """
        This method takes in an organism as input and returns a numpy array
        representing the vision matrix of the organism.

        Args:
            organism: An instance of the Organism class whose vision matrix is
            to be calculated.

        Returns:
            Numpy array representing the vision matrix of the organism.

        Raises:
            No specific exceptions are raised.
        """
        # find the organism's position
        i, o = next((i, o) for i, o in self.organisms.items() if o is organism)
        # get nearby organisms from the rtree
        organisms = [
            self.organisms[i] for i in self.organism_idx.intersection((
                o.x[0] - self.vision_range, o.x[1] - self.vision_range,
                o.x[0] + self.vision_range, o.x[1] + self.vision_range))
        ]
        food = [
            self.food[i]
            for i in self.food_idx.intersection((o.x[0] - self.vision_range,
                                                 o.x[1] - self.vision_range,
                                                 o.x[0] + self.vision_range,
                                                 o.x[1] + self.vision_range))
        ]
        # eliminate the target organism to compute matrix for
        organisms = list(filter(lambda os: os is not organism, organisms))
        return self._vision(organism, organisms, food)

    def tick(self):
        """
        This method simulates the passage of time in the environment. It iterates through each organism and performs the following operations:
            - Detects for the organism if it's colliding with any other organism.
            - Evaluates the organism based on the vision matrix of the organisms surroundings.
            - Updates the position of the organism based on the evaluation result.
        Afterwards, with a probability of 0.1 a new food is added to the environment.

        Args:
            None

        Returns:
            None
        """
        for organism in list(self.organisms.values()):
            self.detect_collision(organism)
            result = organism.evaluate(self.to_organism_input(organism))
            # print('input shape', self.to_organism_input(organism).shape)
            # print('result shape', result.shape)
            self.organism_result_to_coordinates(organism, result)
            if not organism.is_alive:
                self.remove_organism(organism)
        # add food with some probability
        food_to_spawn = np.random.exponential(self._food_appearance_prob)
        for _ in range(int(food_to_spawn)):
            self.add_food()


class SectorVision(Vision):

    def __init__(self,
                 distance: float = 1.,
                 distance_sectors: int = 6,
                 angle_sectors: int = 6) -> None:
        self._distance = distance
        self._distance_sectors = distance_sectors
        self._angle_sectors = angle_sectors

    @property
    def organism_input_shape(self):
        return self._distance_sectors * self._angle_sectors

    def __call__(self, target: Organism, organisms: list[Organism],
                 food: list[tuple[float, float]]) -> np.ndarray:
        organism_coordinates = target.x

        def to_angle(x):
            return np.arctan2(*(x - organism_coordinates).T)

        organisms_coordinates = np.array([o.x for o in organisms])
        food = np.array(food)

        distances_to_organisms = np.linalg.norm(
            organism_coordinates - organisms_coordinates,
            axis=1) if len(organisms_coordinates) > 0 else np.array([])
        distances_to_food = np.linalg.norm(
            organism_coordinates -
            food, axis=1) if len(food) > 0 else np.array([])

        angles_to_organisms = to_angle(organisms_coordinates) if len(
            organisms_coordinates) > 0 else np.array([])
        angles_to_food = to_angle(food) if len(food) > 0 else np.array([])

        angles_to_organisms = np.where(angles_to_organisms < 0,
                                       2 * np.pi + angles_to_organisms,
                                       angles_to_organisms) / (2 * np.pi)
        angles_to_food = np.where(angles_to_food < 0, 2 * np.pi +
                                  angles_to_food, angles_to_food) / (2 * np.pi)

        encoding_matrix = np.zeros(
            (self._distance_sectors, self._angle_sectors))

        dist_boundaries = np.linspace(0, self._distance,
                                      self._distance_sectors)
        angle_boundaries = np.linspace(0, 1, self._angle_sectors)

        # other organisms vision
        # dist_indices = np.searchsorted(dist_boundaries, distances_to_organisms) - 1
        # angle_indices = np.searchsorted(angle_boundaries, angles_to_organisms) - 1
        # encoding_matrix[dist_indices, angle_indices] = -1
        # food vision
        dist_indices = np.searchsorted(dist_boundaries, distances_to_food) - 1
        angle_indices = np.searchsorted(angle_boundaries, angles_to_food) - 1
        encoding_matrix[dist_indices, angle_indices] = 1

        return encoding_matrix.reshape(
            (self._distance_sectors * self._angle_sectors, 1))


class NearestFoodParticleVision(Vision):

    def __init__(self) -> None:
        pass

    @property
    def organism_input_shape(self):
        return 2

    def __call__(self, target: Organism, organisms: list[Organism],
                 food: list[tuple[float, float]]) -> np.ndarray:
        if not food:
            return np.array([[0], [0]])
        f = np.array(food)
        # define nearest food particle
        distances = np.linalg.norm(target.x - f, axis=1)
        min_distance_idx = distances.argmin()
        angle = np.arctan2(*(f[min_distance_idx] - target.x))
        direction = angle - target.r
        return np.array([[distances[min_distance_idx]], [direction]])


# def distance_point_line(point, line):
#     """Calculate the distance from a point to a line segment."""
#     line_start = line[0]
#     line_end = line[1]
#     v = line_end - line_start
#     w = point - line_start
#     c1 = np.dot(w, v)
#     c2 = np.dot(v, v)
#     if c1 <= 0:
#         return np.linalg.norm(point - line_start)
#     if c2 <= c1:
#         return np.linalg.norm(point - line_end)
#     b = c1 / c2
#     pb = line_start + b * v
#     return np.linalg.norm(point - pb)
