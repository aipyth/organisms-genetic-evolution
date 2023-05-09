import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm

import encode
import evolve
from environment import Simple2DContinuousEnvironment
from organism import Organism

ORGANISMS_NUM = 15
WIDTH = 1
HEIGHT = 1
ITERATIONS = 200
GENERATION_TIME = 20  # time for each generation to live

PLOT_FOOD_SIZE = 0.003
PLOT_ORG_SIZE = 0.009

TIMEGEN = str(datetime.now())
# os.mkdir(TIMEGEN)


def main():
    env = Simple2DContinuousEnvironment(WIDTH, HEIGHT)
    ev = evolve.Evolve(
        encoding=encode.RealValued(),
        selection=evolve.TruncationSelection(),
        crossover=evolve.SBXCrossover(mu=5),
        mutation=evolve.GaussianMutation(mu=0, sigma=0.08, p=0.3),
        elitism=5,
    )

    genome_size = [env.organism_input_dimension, 4, 2]

    print(f"Adding {ORGANISMS_NUM} to the environment")
    for _ in range(ORGANISMS_NUM):
        org = Organism()
        org.set_genome_size(genome_size)
        env.add_organism(org)

    # print('organisms after added to the env', env.organisms)

    organisms_df = pd.DataFrame(columns=["x", "y", "name", "iteration"])
    food_df = pd.DataFrame(columns=["x", "y", "name", "iteration"])
    with tqdm(total=ITERATIONS, desc="Simulating organisms") as pbar:
        # for i in tqdm(range(ITERATIONS), desc="Simulating organisms"):
        for i in range(ITERATIONS):
            # perform one step in time
            env.tick()

            if i % GENERATION_TIME == 0 and i != 0:
                next_organisms = ev(env.organisms)
                for org in next_organisms:
                    if org.x is None and org.y is None:
                        env.add_organism(org)

            # store organisms and food locations
            organisms_df = add_stats_record(env.organisms, organisms_df, i)
            food_df = add_stats_record(env.food, food_df, i, "food")
            pbar.set_postfix({"Number of organisms": len(env.organisms)})
            pbar.update(1)

    print(organisms_df)

    plot_evolution(organisms_df, food_df)


def add_stats_record(source, df, iteration, name=None):
    records = []
    for r in source:
        x = r[0] if isinstance(r, tuple) else r.x
        y = r[1] if isinstance(r, tuple) else r.y
        record = {
            "x": x,
            "y": y,
            "name": name if name is not None else r.name,
            "iteration": iteration
        }
        records.append(record)

    new_df = pd.DataFrame(records)
    return pd.concat([df, new_df], ignore_index=False)


# def add_stats_record(source, df, iteration, name=None):
#     for r in source:
#         df = pd.concat(
#             [
#                 df,
#                 pd.DataFrame(
#                     {
#                         "x": r[0] if type(r) == tuple else r.x,
#                         "y": r[1] if type(r) == tuple else r.y,
#                         "name": name if name is not None else r.name,
#                         "iteration": iteration
#                     },
#                     index=["name", "iteration"],
#                 ),
#             ],
#             ignore_index=True,
#         )
#     return df


def plot_evolution(organisms_df, food_df):
    trace1 = go.Scatter(
        x=organisms_df['x'],
        y=organisms_df['y'],
        # animation_frame="iteration",
        # animation_group="name",
        mode='markers',
        name='Organisms',
        range_x=[-0.1, 1.1],
        range_y=[-0.1, 1.1],
        marker=dict(size=5, line=dict(width=2, color='DarkSlateGrey')),
    )

    # create scatter plot trace for food
    trace2 = go.Scatter(
        x=food_df['x'],
        y=food_df['y'],
        # animation_frame="iteration",
        # animation_group="name",
        mode='markers',
        name='Food',
        range_x=[-0.1, 1.1],
        range_y=[-0.1, 1.1],
        marker=dict(size=5, line=dict(width=2, color='Red')),
    )

    # create figure layout
    layout = go.Layout(title='Organisms in the environment',
                       xaxis=dict(title='X-axis'),
                       yaxis=dict(title='Y-axis'),
                       updatemenus=[
                           dict(type="buttons",
                                buttons=[
                                    dict(label="Play",
                                         method="animate",
                                         args=[None]),
                                    dict(label="Pause",
                                         method="animate",
                                         args=[[None], {
                                             "frame": {
                                                 "duration": 0,
                                                 "redraw": False
                                             },
                                             "mode": "immediate"
                                         }])
                                ],
                                showactive=True,
                                direction="left",
                                x=0.05,
                                y=1.1),
                       ],
                       sliders=[
                           dict(active=1,
                                steps=[
                                    dict(method="animate",
                                         args=[
                                             None, {
                                                 "frame": {
                                                     "duration": 500,
                                                     "redraw": False
                                                 },
                                                 "mode": "immediate",
                                                 "fromcurrent": True
                                             }
                                         ],
                                         label=str(i))
                                    for i in range(ITERATIONS)
                                ],
                                x=0.1,
                                y=1.1)
                       ])
    frames = [
        go.Frame(data=[
            go.Scatter(x=organisms_df.loc[organisms_df['iteration'] == i]['x'],
                       y=organisms_df.loc[organisms_df['iteration'] == i]['y'],
                       mode='markers',
                       name='Organisms'),
            go.Scatter(x=food_df.loc[food_df['iteration'] == i]['x'],
                       y=food_df.loc[food_df['iteration'] == i]['y'],
                       mode='markers',
                       name='Food')
        ]) for i in range(ITERATIONS)
    ]
    fig = go.Figure(data=[trace1, trace2], layout=layout, frames=frames)
    fig.show()


# def generate_video():
#     os.chdir(TIMEGEN)
#     os.system(f"ffmpeg -framerate 10 -i %d.png output.mp4")

if __name__ == "__main__":
    main()
    # generate_video()
