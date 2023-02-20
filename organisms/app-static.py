from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np

from tqdm import tqdm

from environment import Simple2DContinuousEnvironment
from organism import Organism

env = Simple2DContinuousEnvironment()
ORGANISMS_NUM = 25

orgs = []
for i in tqdm(range(ORGANISMS_NUM)):
    org = Organism()
    # input is 9 as we devide our vision spectre into 9 subsectors
    # output is 2: x and y acceleration
    org.set_genome_size([9, 4, 2])
    orgs.append(org)

env.propagate_organisms(orgs)
# env.set_organism_constraint(np.array([.3, .3]))
env.set_organism_size(0.05)
env.set_environment_viscosity(0.1)

df = pd.DataFrame(columns=['x', 'y', 'name', 'iteration'])
iterations = 100
for i in tqdm(range(iterations)):
    print('i', i)
    env.tick()
    for (k, v) in env.organisms_coordinates.items():
        df = pd.concat([
            df,
            pd.DataFrame({
                'x': v[0],
                'y': v[1],
                'name': k,
                'iteration': i
            },
                         index=['name', 'iteration'])
        ],
                       ignore_index=True)

# fig = px.scatter(
#     df,
#     x="x",
#     y="y",
#     # size="population",
#     color="name",
#     hover_name="name",
#     size_max=4)
fig = px.scatter(
    df,
    x="x",
    y="y",
    animation_frame="iteration",
    animation_group="name",
    # size=,
    color="name",
    hover_name="name",
    # size_max=10,
    range_x=[0, 1],
    range_y=[0, 1])

fig.update_traces(marker=dict(size=8,
                              line=dict(width=2, color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

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
