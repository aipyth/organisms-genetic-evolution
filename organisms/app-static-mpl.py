from matplotlib import pyplot as plt
import matplotlib.lines as lines
from matplotlib.patches import Circle
from tqdm import tqdm
from datetime import datetime
import os
import pandas as pd
import numpy as np

from environment import Simple2DContinuousEnvironment, SectorVision, NearestFoodParticleVision
from organism import Organism
import evolve
import encode

ORGANISMS_NUM = 6
WIDTH = 4
HEIGHT = 4
ITERATIONS = 400
GENERATION_TIME = 20  # time for each generation to live

ORGANISM_SIZE = 0.018
FOOD_SIZE = 0.008

ORGANISM_VISION_RANGE = 0.4

TIMEGEN = str(datetime.now())

RESULTS_DIR = 'results/'
FRAMES_DIR = 'frames/'
ORGANISMS_LOCATIONS_FILE = 'organisms_locations.csv'
FOOD_LOCATIONS_FILE = 'food_locations.csv'
EATEN_FOOD_FILE = 'eaten_food.csv'
ORGANISMS_STATS_FILE = 'organisms_stats.csv'

# create the results dir with current timestamp
os.mkdir(os.path.join(RESULTS_DIR, TIMEGEN))
os.chdir(os.path.join(RESULTS_DIR, TIMEGEN))
# and create the directory for frames in png
os.mkdir(FRAMES_DIR)


def main():
    # vision = SectorVision(
    #     distance=ORGANISM_VISION_RANGE,
    #     distance_sectors=6,
    #     angle_sectors=5,
    # )
    vision = NearestFoodParticleVision()
    env = Simple2DContinuousEnvironment(width=WIDTH,
                                        height=HEIGHT,
                                        vision=vision,
                                        vision_range=ORGANISM_VISION_RANGE,
                                        organism_size=ORGANISM_SIZE,
                                        food_size=FOOD_SIZE,
                                        food_energy=1,
                                        food_appearance_prob=1)
    ev = evolve.Evolve(
        encoding=encode.RealValued(),
        selection=evolve.TruncationSelection(n=4),
        crossover=evolve.SBXCrossover(mu=5),  # type: ignore
        mutation=evolve.GaussianMutation(mu=0, sigma=0.1,
                                         p=0.3),  # type: ignore
        elitism=4,
    )
    generation = 0

    genome_size = [vision.organism_input_shape, 4, 2]

    print(f"Genome size set: {genome_size}")

    print(f"Adding {ORGANISMS_NUM} to the environment")
    for _ in range(ORGANISMS_NUM):
        org = Organism()
        org.set_genome_size(genome_size)
        env.add_organism(org)

    for _ in range(np.random.randint(ORGANISMS_NUM, ORGANISMS_NUM * 2)):
        env.add_food()

    organisms_loc_df = pd.DataFrame(
        [], columns=['index', "x", "y", "name", "iteration"])
    food_loc_df = pd.DataFrame(
        [], columns=['index', "x", "y", "name", "iteration"])
    eaten_stats = pd.DataFrame(
        [],
        columns=['organism_id', 'food_location', 'energy_taken', 'iteration'])
    organisms_stats = pd.DataFrame([],
                                   columns=[
                                       'id', 'name', 'genome', 'age',
                                       'parents', 'iteration', 'energy'
                                   ])
    with tqdm(total=ITERATIONS, desc="Simulating organisms") as pbar:
        for i in range(ITERATIONS):
            # perform one step in time
            eaten_on_step = env.tick()

            # evolution
            if i % GENERATION_TIME == 0 and i != 0:
                next_organisms = ev(list(env.organisms.values()))
                for org in next_organisms:
                    if org not in env.organisms.values():
                        env.add_organism(org)
                generation += 1

            # store organisms and food locations
            organisms_loc_df = add_locations_record(env.organisms,
                                                    organisms_loc_df,
                                                    i,
                                                    generation=generation,
                                                    concatenate=True)
            food_loc_df = add_locations_record(env.food,
                                               food_loc_df,
                                               i,
                                               name="food",
                                               concatenate=True)
            eaten_on_step['iteration'] = i
            eaten_stats = pd.concat([eaten_stats, eaten_on_step],
                                    ignore_index=True)
            organisms_stats = pd.concat([
                organisms_stats,
                pull_organisms_stats(env.organisms, iteration=i)
            ],
                                        ignore_index=True)
            pbar.set_postfix({
                "Number of organisms": len(env.organisms),
                "Gen": generation,
            })
            pbar.update(1)
            # TODO: run plot_frame in a separate thread
            plot_frame(organisms_loc_df, food_loc_df, i)
    organisms_loc_df.to_csv(ORGANISMS_LOCATIONS_FILE, index=False)
    food_loc_df.to_csv(FOOD_LOCATIONS_FILE, index=False)
    eaten_stats.to_csv(EATEN_FOOD_FILE, index=False)
    organisms_stats.to_csv(ORGANISMS_STATS_FILE, index=False)


def pull_organisms_stats(organisms: dict[int, Organism], iteration: int):
    df = pd.DataFrame([],
                      columns=[
                          'id', 'name', 'genome', 'age', 'parents',
                          'iteration', 'energy'
                      ])
    for i, o in organisms.items():
        df = pd.concat(
            [
                df,
                pd.DataFrame({
                    'id': i,
                    'name': o.name,
                    'genome': o.genome,
                    'age': o.age,
                    # 'parent': o._parents,
                    'iteration': iteration,
                    'energy': o.energy,
                })
            ],
            ignore_index=True)
    return df


def add_locations_record(source: dict,
                         df,
                         iteration,
                         name=None,
                         generation=None,
                         concatenate=True):
    records = []
    for i, r in source.items():
        x = r[0] if isinstance(r, tuple) else r.x[0]
        y = r[1] if isinstance(r, tuple) else r.x[1]
        theta = r.r if isinstance(r, Organism) else 0
        record = {
            "index": i,
            "x": x,
            "y": y,
            "theta": theta,
            "name": name if name is not None else r.name,
            "iteration": iteration,
            "generation": generation,
        }
        if isinstance(r, Organism):
            record["v"] = r.v
            record["a"] = r.a
            record["energy"] = r.energy
            record['age'] = r.age
        records.append(record)

    new_df = pd.DataFrame(records)
    return pd.concat([df, new_df],
                     ignore_index=False) if concatenate else new_df


# PLOTTING


def plot_organism(x1, y1, theta, ax):

    circle = Circle([x1, y1],
                    ORGANISM_SIZE,
                    edgecolor="g",
                    facecolor="lightgreen",
                    zorder=8)
    ax.add_artist(circle)

    edge = Circle([x1, y1],
                  ORGANISM_SIZE,
                  facecolor="None",
                  edgecolor="darkgreen",
                  zorder=8)
    ax.add_artist(edge)

    tail_len = ORGANISM_SIZE * 2

    x2 = np.cos(theta) * tail_len + x1
    y2 = np.sin(theta) * tail_len + y1

    ax.add_line(
        lines.Line2D([x1, x2], [y1, y2],
                     color='darkgreen',
                     linewidth=0.3,
                     zorder=10))


def plot_food(x1, y1, ax):

    circle = Circle(
        [x1, y1],
        FOOD_SIZE,
        edgecolor="darkslateblue",
        facecolor="mediumslateblue",
        zorder=5,
    )
    ax.add_artist(circle)


def plot_frame(organisms_df, food_df, i):

    fig, ax = plt.subplots()
    fig.set_size_inches(6.8, 5.4)
    ax.grid()

    plt.xlim([0 - WIDTH * 0.1, WIDTH + WIDTH * 0.1])
    plt.ylim([0 - WIDTH * 0.1, WIDTH + WIDTH * 0.1])

    organisms = organisms_df[organisms_df["iteration"] == i]
    food = food_df[food_df["iteration"] == i]
    # PLOT ORGANISMS
    for ind in organisms.index:
        plot_organism(organisms["x"][ind], organisms["y"][ind],
                      organisms["theta"][ind], ax)

    # PLOT FOOD PARTICLES
    for ind in food.index:
        plot_food(food["x"][ind], food["y"][ind], ax)

    # MISC PLOT SETTINGS
    ax.set_aspect("equal")
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticks(np.linspace(0, WIDTH, 9))
    frame.axes.get_yaxis().set_ticks(np.linspace(0, HEIGHT, 9))

    plt.figtext(0.025, 0.95, f"Generation: {organisms['generation'].max()}")
    plt.figtext(0.025, 0.90, f"Time: {i}")

    plt.savefig(f"{FRAMES_DIR}/{i}.png", dpi=100)
    plt.close()


def generate_video():
    framerate = 24
    os.system(
        f"ffmpeg -framerate {framerate} -i {FRAMES_DIR}%d.png output.mp4")


if __name__ == "__main__":
    main()
    generate_video()
