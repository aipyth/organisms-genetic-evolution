import numpy as np
# import pandas as pd
import uuid

from organism import Organism

# import plotly.express as px


class Simple2DContinuousEnvironment:

    def __init__(self,
                 coordinates_bounds: tuple[float, float] = (0, 1),
                 vision_distance: float = 0.1,
                 viscosity: float = 0.01):
        self.environment_shape = 2
        self._viscosity = viscosity
        self._coordinates_bounds = coordinates_bounds
        self._vision_distance = vision_distance
        self.walls = [
            np.array([
                [0, 0],
                [0, 1],
            ]),
            np.array([
                [0, 1],
                [1, 1],
            ]),
            np.array([
                [1, 1],
                [1, 0],
            ]),
            np.array([[1, 0], [0, 0]]),
        ]
        self.organism_size = 0.05

    def set_walls(self, walls: list[np.ndarray]):
        self.walls = walls

    def set_organism_size(self, size: float):
        self.organism_size = size

    def propagate_organisms(self, organisms: list[Organism]):

        def generate_random_point():
            return np.random.uniform(0, 1, 2)

        self.organisms = {str(uuid.uuid4()): o for o in organisms}
        self.organisms_coordinates = {
            k: generate_random_point()
            for (k, _) in self.organisms.items()
        }

    def set_environment_viscosity(self, viscosity):
        self._viscosity = viscosity

    def organism_result_to_coordinates(self, code: str,
                                       result: np.ndarray) -> np.ndarray:
        if code not in self.organisms_coordinates:
            raise KeyError("invalid organism code accessed")
        coordinates_bounds = self._coordinates_bounds  # (min, max)
        last_coordinates = self.organisms_coordinates.get(code)
        coordinates = last_coordinates + \
            self._viscosity * result.flatten()
        # keep in the environment bounds
        # coordinates = np.minimum(coordinates, coordinates_bounds[1])
        # coordinates = np.maximum(coordinates, coordinates_bounds[0])
        # Check for collisions with walls and other organisms
        # if self._detect_collision(coordinates, code):
        # Use the direction of least resistance to avoid collisions
        # direction = self._direction_of_least_resistance(
        #     coordinates, last_coordinates)
        # coordinates = last_coordinates + self.organism_constraint * direction
        # coordinates += self._correct_organism_direction(
        #     code, coordinates, last_coordinates)
        # print('last coords', last_coordinates)
        # print('calculated', coordinates)
        # print(
        #     'cirrection',
        #     self._correct_organism_direction(code, coordinates,
        #                                      last_coordinates))
        coordinates = last_coordinates + self._correct_organism_direction(
            code, coordinates, last_coordinates)
        # print('corrected', coordinates)

        # Keep the organism in the environment bounds after collision avoidance
        coordinates = np.minimum(coordinates, coordinates_bounds[1])
        coordinates = np.maximum(coordinates, coordinates_bounds[0])

        # Update the organism's coordinates
        self.organisms_coordinates[code] = coordinates

        return coordinates

    def _correct_organism_direction(
            self, code: str, coordinates: np.ndarray,
            last_coordinates: np.ndarray) -> np.ndarray:
        direction = coordinates - last_coordinates
        unit_direction = direction / np.linalg.norm(direction)

        # print('---------------------------------------------')
        # print('organism currect location', last_coordinates)
        # print('directing to', coordinates)
        # print('direction', direction)
        # print('unit direction', unit_direction)

        resistance_forces = np.zeros(unit_direction.shape)

        for cd, coord in self.organisms_coordinates.items():
            if cd == code:
                continue
            distance = np.linalg.norm(coordinates - coord)
            # print(f'distance to organism {coord} is {distance}')
            if distance > self.organism_size:
                continue
            # direction -= distance
            normal_vector = coordinates - coord
            unit_normal_vector = normal_vector / np.linalg.norm(normal_vector)
            print(f'distance to organism {coord} is {distance}')
            # print('adding resistance force of other organism',
            #       unit_normal_vector)
            print('adding resistance force of other organism',
                  normal_vector)
            resistance_forces += normal_vector
        # for wall in self.walls:
        #     if not check_collision_with_wall(coordinates, wall[0], wall[1],
        #                                      self.organism_size):
        #         continue
        #     normal_vector = np.array(
        #         [wall[1, 1] - wall[0, 1], wall[0, 0] - wall[1, 0]])
        #     unit_normal_vector = normal_vector / np.linalg.norm(normal_vector)
        #     # print('adding resistance force of wall', unit_normal_vector)
        #     resistance_forces += unit_normal_vector
        #     # dot_product = np.dot(unit_direction, unit_normal_vector)
        #     # if dot_product < min_dot_product:
        #     #     min_dot_product = dot_product
        #     #     min_normal_vector = unit_normal_vector

        print('resulting force', direction + resistance_forces)
        # print('resulting force', direction * resistance_forces)
        # return direction * resistance_forces
        return direction + resistance_forces

    def to_organism_input(self, code: str) -> np.ndarray:
        if code not in self.organisms_coordinates:
            raise KeyError("invalid organism code accessed")
        coordinates = self.organisms_coordinates.get(code)
        # get nearest neighbours in k distance
        max_distance = self._vision_distance
        # TODO: sectors as environment parameter
        distance_sectors = 3
        angle_sectors = 3
        nearest_organisms_code_distance = dict(
            filter(
                lambda x: x[1] < max_distance and x[1] != 0,
                {
                    k: np.sum(np.sqrt(np.power(coordinates - v, 2)))
                    for (k, v) in self.organisms_coordinates.items()
                }.items(),
            ))

        def to_angle(x):
            y = self.organisms_coordinates.get(x[0])
            return np.arctan2(*(y - coordinates))

        nearest_organisms_angle = map(to_angle,
                                      nearest_organisms_code_distance.items())
        # print(list(nearest_organisms_angle))
        distances = np.fromiter(nearest_organisms_code_distance.values(),
                                dtype=np.float64)
        # get only non-negative angles [0. 2pi]
        angles = np.fromiter(
            map(lambda x: 2 * np.pi + x
                if x < 0 else x, nearest_organisms_angle),
            dtype=np.float64,
        )
        # angles normalization
        angles /= 2 * np.pi
        # compute the vision matrix
        encoding_matrix = np.zeros((distance_sectors, angle_sectors))
        dist_boundaries = np.linspace(0, max_distance, distance_sectors + 1)
        angle_boundaries = np.linspace(0, 1, angle_sectors + 1)
        for d, a in zip(distances, angles):
            sector = np.searchsorted(dist_boundaries, d) - 1
            angle = np.searchsorted(angle_boundaries, a) - 1
            encoding_matrix[sector,
                            angle] = 0.5  # set .5 for as other organisms

        # walls
        dist_to_walls = np.fromiter(
            map(lambda x: distance_point_line(coordinates.flatten(), x),
                self.walls),
            dtype=np.float64,
        )
        direction_walls = np.fromiter(map(lambda x: np.arctan2(*(x[0] - x[1])),
                                          self.walls),
                                      dtype=np.float64)
        direction_walls /= 2 * np.pi
        for d, a in zip(dist_to_walls, direction_walls):
            if d >= max_distance:
                continue
            sector = np.searchsorted(dist_boundaries, d) - 1
            angle = np.searchsorted(angle_boundaries, a) - 1
            encoding_matrix[sector, angle] += 1

        return encoding_matrix.reshape((distance_sectors * angle_sectors, 1))

    def tick(self):
        self.organisms_coordinates = {
            k: self.organism_result_to_coordinates(
                k, v.evaluate(self.to_organism_input(k)))
            for (k, v) in self.organisms.items()
        }


def check_collision_with_wall(organism_location, wall_start, wall_end,
                              threshold_distance):
    """
    Check if the organism is in collision with the wall.
    """
    # print(organism_location, wall_start, wall_end, threshold_distance)
    wall_vector = wall_end - wall_start
    closest_point = (wall_start +
                     np.dot(organism_location - wall_start, wall_vector) /
                     np.dot(wall_vector, wall_vector) * wall_vector)
    closest_point = np.maximum(closest_point, wall_start)
    closest_point = np.minimum(closest_point, wall_end)
    distance = np.linalg.norm(closest_point - organism_location)
    return distance < threshold_distance


def check_collision_with_other_organism(organism_location,
                                        other_organism_location,
                                        threshold_distance):
    """
    Check if the organism is in collision with any of the other organisms.
    """
    distance = np.linalg.norm(organism_location - other_organism_location)
    if distance < threshold_distance:
        return True
    return False


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


if __name__ == "__main__":
    env = Simple2DContinuousEnvironment()
    ORGANISMS_NUM = 5

    orgs = []
    for i in range(ORGANISMS_NUM):
        org = Organism()
        # input is 9 as we devide our vision spectre into 9 subsectors
        # output is 2: x and y acceleration
        org.set_genome_size([9, 4, 2])
        orgs.append(org)

    env.propagate_organisms(orgs)

    print(f'{env.organisms_coordinates=}')

    while True:
        env.tick()
        input()

    # df = pd.DataFrame(columns=['x', 'y', 'name', 'iteration'])
    # iterations = 1000
    # for i in range(iterations):
    #     env.tick()
    #     for (k, v) in env.organisms_coordinates.items():
    #         df.append({
    #             'x': v[0],
    #             'y': v[1],
    #             'name': k,
    #             'iteration': i
    #         },
    #                   ignore_index=True)

    # fig = px.scatter(
    #     df,
    #     x="x",
    #     y="y",
    #     animation_frame="iteration",
    #     # animation_group="country",
    #     # size="pop",
    #     color="name",
    #     # hover_name="country",
    #     # log_x=True,
    #     # size_max=55,
    #     range_x=[0, 1],
    #     range_y=[0, 1])
    # fig.show()
