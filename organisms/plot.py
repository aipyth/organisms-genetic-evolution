from matplotlib import pyplot as plt
import matplotlib.lines as lines
from matplotlib.patches import Circle
import matplotlib

import numpy as np
import os
from tqdm import tqdm

import threading

matplotlib.use('agg')


def plot_frame_wrapper(self, args):
    progress_bar = args[0]
    plot_frame(*(args[1:]))
    progress_bar.update(1)
    return True


def create_frames(organisms, food, width, height, organism_size, food_size,
                  results_dir):
    iterations = organisms['iteration'].max()
    progress_bar = tqdm(total=iterations)
    threads_pool: list[threading.Thread] = []
    pool_size = 20
    # run plot_frame in a separate thread
    for i in range(iterations):
        t = threading.Thread(target=plot_frame,
                             args=(organisms[organisms['iteration'] == i],
                                   food[food['iteration'] == i], i, width,
                                   height, organism_size, food_size,
                                   results_dir))
        t.start()
        threads_pool.append(t)
        if len(threads_pool) >= pool_size:
            threads_pool[0].join()
            threads_pool.pop(0)
            progress_bar.update(1)
            # for i, pt in enumerate(threads_pool[:]):
            #     if not pt.is_alive():
            #         threads_pool.pop(i)
            #         progress_bar.update(1)
    # with Pool(20) as p:
    #     p.map(plot_frame_wrapper,
    #           [(progress_bar, organisms[organisms['iteration'] == i],
    #             food[food['iteration'] == i], i, width, height, organism_size,
    #             food_size, results_dir) for i in range(iterations)])


def plot_organism(x1, y1, theta, ax, organism_size):

    circle = Circle([x1, y1],
                    organism_size,
                    edgecolor="g",
                    facecolor="lightgreen",
                    zorder=8)
    ax.add_artist(circle)

    edge = Circle([x1, y1],
                  organism_size,
                  facecolor="None",
                  edgecolor="darkgreen",
                  zorder=8)
    ax.add_artist(edge)

    tail_len = organism_size * 2

    x2 = np.cos(theta) * tail_len + x1
    y2 = np.sin(theta) * tail_len + y1

    ax.add_line(
        lines.Line2D([x1, x2], [y1, y2],
                     color='darkgreen',
                     linewidth=0.3,
                     zorder=10))


def plot_food(x1, y1, ax, food_size):

    circle = Circle(
        [x1, y1],
        food_size,
        edgecolor="darkslateblue",
        facecolor="mediumslateblue",
        zorder=5,
    )
    ax.add_artist(circle)


def plot_frame(organisms, food, i, xlim, ylim, organism_size, food_size,
               folder):

    fig, ax = plt.subplots()
    fig.set_size_inches(6.8, 5.4)
    ax.grid()

    ax.set_xlim([0 - xlim * 0.1, xlim + xlim * 0.1])
    ax.set_ylim([0 - ylim * 0.1, ylim + ylim * 0.1])

    for ind in organisms.index:
        plot_organism(organisms["x"][ind], organisms["y"][ind],
                      organisms["theta"][ind], ax, organism_size)

    for ind in food.index:
        plot_food(food["x"][ind], food["y"][ind], ax, food_size)

    ax.set_aspect("equal")
    # frame = plt.gca()
    # frame.axes.get_xaxis().set_ticks(np.linspace(0, xlim, 9))
    # frame.axes.get_yaxis().set_ticks(np.linspace(0, ylim, 9))

    fig.text(0.025, 0.95, f"Generation: {organisms['generation'].max()}")
    fig.text(0.025, 0.90, f"Time: {i}")

    fig.savefig(f"{folder}/{i}.png", dpi=100)
    plt.close(fig)


def generate_video(result_dir: str,
                   framerate: int = 24,
                   output: str = 'output.mp4'):
    os.system(f"ffmpeg -framerate {framerate} -i {result_dir}%d.png {output}")
