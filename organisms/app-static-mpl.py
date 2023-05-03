from matplotlib import pyplot as plt
import matplotlib.lines as lines
from matplotlib.patches import Circle
from tqdm import tqdm
from datetime import datetime
import os
import pandas as pd

from environment import Simple2DContinuousEnvironment
from organism import Organism
import evolve
import encode

ORGANISMS_NUM = 15
WIDTH = 1
HEIGHT = 1
ITERATIONS = 300

PLOT_FOOD_SIZE = 0.003
PLOT_ORG_SIZE = 0.009

TIMEGEN = str(datetime.now())
os.mkdir(TIMEGEN)


def main():
    env = Simple2DContinuousEnvironment(WIDTH, HEIGHT)
    ev = evolve.Evolve(
        encoding=encode.RealValued(),
        selection=evolve.TruncationSelection(),
        crossover=evolve.SBXCrossover(mu=5),
        mutation=evolve.GaussianMutation(mu=0, sigma=0.01, p=0.3),
        elitism=5,
    )

    genome_size = [env.organism_input_dimension, 4, 2]

    print(f"Adding {ORGANISMS_NUM} to the environment")
    for i in range(ORGANISMS_NUM):
        org = Organism()
        org.set_genome_size(genome_size)
        env.add_organism(org)

    print('organisms after added to the env', env.organisms)

    organisms_df = pd.DataFrame(columns=["x", "y", "name", "iteration"])
    food_df = pd.DataFrame(columns=["x", "y", "name", "iteration"])
    for i in tqdm(range(ITERATIONS)):
        # perform one step in time
        env.tick()
        print('organisms before tick', env.organisms)
        print('tick')
        print('organisms after tick', env.organisms)
        next_organisms = ev(env.organisms)
        for org in next_organisms:
            if org.x is None and org.y is None:
                env.add_organism(org)
        # store organisms and food locations
        organisms_df = add_stats_record(env.organisms, organisms_df, i)
        food_df = add_stats_record(env.food, food_df, i, "food")
        plot_frame(organisms_df, food_df, i)


def add_stats_record(source, df, iteration, name=None):
    for r in source:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "x": r[0] if type(r) == tuple else r.x,
                        "y": r[1] if type(r) == tuple else r.y,
                        "name": name if name is not None else r.name,
                        "iteration": iteration
                    },
                    index=["name", "iteration"],
                ),
            ],
            ignore_index=True,
        )
    return df


# PLOTTING


def plot_organism(x1, y1, theta, ax):

    circle = Circle([x1, y1],
                    PLOT_ORG_SIZE,
                    edgecolor="g",
                    facecolor="lightgreen",
                    zorder=8)
    ax.add_artist(circle)

    edge = Circle([x1, y1],
                  PLOT_ORG_SIZE,
                  facecolor="None",
                  edgecolor="darkgreen",
                  zorder=8)
    ax.add_artist(edge)

    # tail_len = 0.075

    # x2 = cos(radians(theta)) * tail_len + x1
    # y2 = sin(radians(theta)) * tail_len + y1

    # ax.add_line(lines.Line2D([x1, x2], [y1, y2],
    # color='darkgreen', linewidth=1, zorder=10))


def plot_food(x1, y1, ax):

    circle = Circle(
        [x1, y1],
        PLOT_FOOD_SIZE,
        edgecolor="darkslateblue",
        facecolor="mediumslateblue",
        zorder=5,
    )
    ax.add_artist(circle)


def plot_frame(organisms_df, food_df, i):

    fig, ax = plt.subplots()
    fig.set_size_inches(6.8, 5.4)

    plt.xlim([WIDTH * 0.1, WIDTH + WIDTH * 0.1])
    plt.ylim([WIDTH * 0.1, WIDTH + WIDTH * 0.1])

    organisms = organisms_df[organisms_df["iteration"] == i]
    food = food_df[food_df["iteration"] == i]

    # print(organisms)

    # organisms.map(lambda o: plot_organism(o['x'], o['y'], 0, ax))
    # PLOT ORGANISMS
    for ind in organisms.index:
        plot_organism(organisms["x"][ind], organisms["y"][ind], 0, ax)

    # PLOT FOOD PARTICLES
    for ind in food.index:
        plot_food(food["x"][ind], food["y"][ind], ax)

    # MISC PLOT SETTINGS
    ax.set_aspect("equal")
    # frame = plt.gca()
    # frame.axes.get_xaxis().set_ticks([])
    # frame.axes.get_yaxis().set_ticks([])

    plt.figtext(0.025, 0.95, f"GENERATION: null")
    plt.figtext(0.025, 0.90, f"T_STEP: {i}")

    plt.savefig(f"{TIMEGEN}/{i}.png", dpi=100)
    plt.close()


def generate_video():
    os.chdir(TIMEGEN)
    os.system(f"ffmpeg -framerate 10 -i %d.png output.mp4")


if __name__ == "__main__":
    main()
    generate_video()
