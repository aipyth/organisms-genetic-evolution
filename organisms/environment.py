from typing import List
import numpy as np
from rtree import index
from organism import Organism


class Simple2DContinuousEnvironment:
    def __init__(
        self, width: int, height: int, distance_sectors: int = 6, angle_sectors: int = 6
    ):
        self.width = width
        self.height = height
        self.organisms = []
        self.food = []
        self.organism_idx = index.Index()
        self.food_idx = index.Index()
        self._distance_sectors = distance_sectors
        self._angle_sectors = angle_sectors
        self.food_energy = 10
        self.food_size = 0.03

    @property
    def organism_input_dimension(self):
        return self._distance_sectors * self._angle_sectors

    def add_food(self):
        # generate random location for the food
        x = np.random.uniform(0, self.width)
        y = np.random.uniform(0, self.height)
        # add the food to the environment
        self.food.append((x, y))
        # insert into the rtree
        self.food_idx.insert(len(self.food) - 1, (x, y, x, y))

    def remove_food(self, food):
        for i, (x, y) in enumerate(self.food):
            if (x, y) == food:
                self.food.pop(i)
                self.food_idx.delete(i, (x, y, x, y))
                break

    def detect_collision(self, organism):
        """Check for collision with food and increase energy if collision occurs."""
        vision_range = 10
        # get organism's coordinates
        x, y, _ = next((x, y, o)
                       for x, y, o in self.organisms if o == organism)
        # get all food in a region
        foods = [
            food
            for food in self.food_idx.intersection(
                (x - vision_range, y - vision_range,
                 x + vision_range, y + vision_range)
            )
        ]
        for food in foods:
            if np.linalg.norm(np.array(food) - np.array((x, y))) <= self.food_size:
                # print('organism ate a piece of food')
                organism.energy += self.food_energy
                # organism.energy_inc(self.food_energy)
                self.remove_food(food)

    def add_organism(self, organism, x: float = None, y: float = None):
        if x is None or y is None:
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)
        self.organisms.append((x, y, organism))
        self.organism_idx.insert(len(self.organisms) - 1, (x, y, x, y))

    def remove_organism(self, organism):
        for i, (x, y, o) in enumerate(self.organisms):
            if o == organism:
                self.organisms.pop(i)
                self.organism_idx.delete(i, (x, y, x, y))
                break

    def organism_result_to_coordinates(
        self, organism, result: np.ndarray
    ) -> tuple[float, float]:
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
        i, (x, y, _) = next(
            (i, (x, y, o))
            for i, (x, y, o) in enumerate(self.organisms)
            if o == organism
        )
        dx, dy = result
        x += dx
        y += dy
        x = np.clip(x, 0, self.width)
        y = np.clip(y, 0, self.height)
        self.organism_idx.delete(i, (x, y, x, y))
        self.organisms[i] = (x, y, organism)
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
        i, (x, y, _) = next(
            (i, (x, y, o))
            for i, (x, y, o) in enumerate(self.organisms)
            if o == organism
        )
        # get nearby organisms from the rtree
        vision_range = 10
        organisms = [
            self.organisms[i]
            for i in self.organism_idx.intersection(
                (x - vision_range, y - vision_range,
                 x + vision_range, y + vision_range)
            )
        ]
        # eliminate the target organism to compute matrix for
        organisms = list(filter(lambda os: os[2] != organism, organisms))
        vision_distance = 10
        vision_matrix = sector_vision(
            (x, y),
            organisms,
            vision_distance,
            self._distance_sectors,
            self._angle_sectors,
        )
        return vision_matrix

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
        for x, y, organism in self.organisms:
            self.detect_collision(organism)
            result = organism.evaluate(self.to_organism_input(organism))
            self.organism_result_to_coordinates(organism, result)
        # add food with some probability
        if np.random.rand() < 0.1:
            self.add_food()


def sector_vision(
    organism_coordinates: tuple[float, float],
    organisms: List[tuple[float, float, Organism]],
    distance: float,
    distance_sectors: int = 6,
    angle_sectors: int = 6,
):
    organism_coordinates = np.array(organism_coordinates)
    metric = np.linalg.norm

    def to_angle(x):
        return np.arctan2(*(x - organism_coordinates))

    distances = map(
        lambda o: metric(organism_coordinates -
                         np.array([o[0], o[1]])), organisms
    )

    angles = map(lambda o: to_angle(np.array([o[0], o[1]])), organisms)
    # get only non-negative angles [0. 2pi]
    list(angles)
    # angles normalization to [0, 1]
    angles = np.fromiter(
        map(lambda x: 2 * np.pi + x if x < 0 else x, angles),
        dtype=np.float64,
    ) / (2 * np.pi)
    # angles /= 2 * np.pi

    encoding_matrix = np.zeros((distance_sectors, angle_sectors))

    # compute the vision matrix
    dist_boundaries = np.linspace(0, distance, distance_sectors)
    angle_boundaries = np.linspace(0, 1, angle_sectors)
    for d, a in zip(distances, angles):
        sector = np.searchsorted(dist_boundaries, d)
        angle = np.searchsorted(angle_boundaries, a)
        encoding_matrix[sector, angle] = 0.5  # set .5 for as other organisms
    return encoding_matrix.reshape((distance_sectors * angle_sectors, 1))


def distance_point_line(point, line):
    """Calculate the distance from a point to a line segment."""
    line_start = line[0]
    line_end = line[1]
    v = line_end - line_start
    w = point - line_start
    c1 = np.dot(w, v)
    c2 = np.dot(v, v)
    if c1 <= 0:
        return np.linalg.norm(point - line_start)
    if c2 <= c1:
        return np.linalg.norm(point - line_end)
    b = c1 / c2
    pb = line_start + b * v
    return np.linalg.norm(point - pb)
