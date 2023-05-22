from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
from dash import dcc
from dash import html
import pandas as pd
import numpy as np

from tqdm import tqdm

from environment import Simple2DContinuousEnvironment
from organism import Organism

width = 1
height = 1

env = Simple2DContinuousEnvironment(width, height)
ORGANISMS_NUM = 15

# for i in tqdm(range(ORGANISMS_NUM)):
for i in range(ORGANISMS_NUM):
    org = Organism()
    org.set_genome_size([env.organism_input_dimension, 4, 2])
    env.add_organism(org)

# env.propagate_organisms(orgs)
# # env.set_organism_constraint(np.array([.3, .3]))
# env.set_organism_size(0.05)
# env.set_environment_viscosity(0.05)

organisms_df = pd.DataFrame(columns=['x', 'y', 'name', 'iteration'])
food_df = pd.DataFrame(columns=['x', 'y', 'name', 'iteration'])
iterations = 300
for i in tqdm(range(iterations)):
    env.tick()
    for org in env.organisms:
        # for (k, v) in env.organisms_coordinates.items():
        organisms_df = pd.concat([
            organisms_df,
            pd.DataFrame(
                {
                    'x': org[0],
                    'y': org[1],
                    'name': '1',
                    'iteration': i
                },
                index=['name', 'iteration'])
        ],
            ignore_index=True)
    for food in env.food:
        # for (k, v) in env.organisms_coordinates.items():
        food_df = pd.concat([
            food_df,
            pd.DataFrame(
                {
                    'x': food[0],
                    'y': food[1],
                    'name': 'food',
                    'iteration': i
                },
                index=['name', 'iteration'])
        ],
            ignore_index=True)

print(organisms_df.shape)
print(food_df.shape)
# fig = px.scatter(df,
#                  x="x",
#                  y="y",
#                  animation_frame="iteration",
#                  animation_group="name",
#                  color="name",
#                  hover_name="name",
#                  range_x=[0, 1],
#                  range_y=[0, 1])
#
# fig.update_traces(marker=dict(size=5,
#                               line=dict(width=2, color='DarkSlateGrey')),
#                   selector=dict(mode='markers'))

# create scatter plot trace for organisms
trace1 = go.Scatter(
    x=organisms_df['x'],
    y=organisms_df['y'],
    # animation_frame="iteration",
    # animation_group="name",
    mode='markers',
    name='Organisms',
    # range_x=[-0.1, 1.1],
    # range_y=[-0.1, 1.1],
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
    # range_x=[-0.1, 1.1],
    # range_y=[-0.1, 1.1],
    marker=dict(size=5, line=dict(width=2, color='Red')),
)

# create figure layout
layout = go.Layout(title='Scatter Plot with Two Dataframes',
                   xaxis=dict(title='X-axis'),
                   yaxis=dict(title='Y-axis'),
                   updatemenus=[
                       dict(type='buttons',
                            showactive=False,
                            buttons=[
                                dict(label='Play',
                                     method='animate',
                                     args=[
                                         None, {
                                             'frame': {
                                                 'duration': 50,
                                                 # 'duration': 500,
                                                 'redraw': True
                                             },
                                             'fromcurrent': True,
                                             'transition': {
                                                 'duration': 0
                                             }
                                         }
                                     ]),
                                dict(label='Pause',
                                     method='animate',
                                     args=[[None], {
                                         'frame': {
                                             'duration': 0,
                                             'redraw': False
                                         },
                                         'mode': 'immediate',
                                         'transition': {
                                             'duration': 0
                                         }
                                     }])
                            ])
                   ])

# create animation frames
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
    ]) for i in range(iterations)
]

# add frames to figure
fig = go.Figure(data=[trace1, trace2], layout=layout, frames=frames)

app = Dash(__name__)
app.layout = html.Div(children=[
    html.H1(children='Live Organisms'),
    html.Div(id='current-time'),
    dcc.Graph(id='environment',
              figure=fig,
              style={
                  'width': '90vw',
                  'height': '90vh'
              }),
])

# @app.callback(Output('environment', 'figure'),
#               Input('interval-component', 'n_intervals'))
# def update_environment(n):
#     fig = px.scatter(df.loc[df['iteration'] == n],
#                      x="x",
#                      y="y",
#                      color="name",
#                      range_x=[0, 1],
#                      range_y=[0, 1])
#     return fig

if __name__ == '__main__':
    app.run_server(debug=True)
