import numpy as np
import random
import matplotlib.pyplot as plt


def poisson_disk_sampling(width, height, radius, rejection_samples=30):
    cell_size = radius / np.sqrt(2)
    grid_width = int(np.ceil(width / cell_size))
    grid_height = int(np.ceil(height / cell_size))
    grid = [[None for _ in range(grid_height)] for _ in range(grid_width)]

    def grid_coords(point):
        return int(point[0] // cell_size), int(point[1] // cell_size)

    def fits(point, grid_coords):
        grid_x, grid_y = grid_coords
        yrange = list(range(max(grid_y-2, 0), min(grid_y+2, grid_height-1)))
        xrange = list(range(max(grid_x-2, 0), min(grid_x+2, grid_width-1)))
        for x in xrange:
            for y in yrange:
                neighbour = grid[x][y]
                if neighbour is not None:
                    dist = (point[0]-neighbour[0])**2 + \
                        (point[1]-neighbour[1])**2
                    if dist < radius**2:
                        return False
        return True

    def generate_point_around(point):
        radius1 = random.uniform(radius, 2*radius)
        angle = 2 * np.pi * random.uniform(0, 1)
        new_x = point[0] + radius1 * np.cos(angle)
        new_y = point[1] + radius1 * np.sin(angle)
        return new_x, new_y

    initial_point = width * random.uniform(0, 1), height * random.uniform(0, 1)
    to_process = [initial_point]
    while to_process:
        point = to_process.pop(0)
        point_grid_coords = grid_coords(point)
        if fits(point, point_grid_coords):
            grid[point_grid_coords[0]][point_grid_coords[1]] = point
            for _ in range(rejection_samples):
                new_point = generate_point_around(point)
                if 0 <= new_point[0] < width and 0 <= new_point[1] < height:
                    new_point_grid_coords = grid_coords(new_point)
                    if fits(new_point, new_point_grid_coords):
                        to_process.append(new_point)
    return [point for row in grid for point in row if point is not None]


if __name__ == '__main__':
    points = poisson_disk_sampling(20, 20, 5)
    px, py = zip(*points)

    opoints = poisson_disk_sampling(20, 20, 8)
    opx, opy = zip(*opoints)

    plt.scatter(px, py)
    plt.scatter(opx, opy)
    plt.show()
