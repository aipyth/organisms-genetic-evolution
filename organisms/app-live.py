from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

from environment import Simple2DContinuousEnvironment
from organism import Organism

UPDATE_INTERVAL = 100  # 100ms

app = Dash(__name__)
app.layout = html.Div(children=[
    html.H1(children='Live Organisms'),
    html.Div(id='current-time'),
    dcc.Graph(id='environment', ),
    dcc.Interval(
        id='interval-component', interval=UPDATE_INTERVAL, n_intervals=0)
])

env = Simple2DContinuousEnvironment(2)
ORGANISMS_NUM = 5

orgs = []
for i in range(ORGANISMS_NUM):
    org = Organism()
    # input is 9 as we devide our vision spectre into 9 subsectors
    # output is 2: x and y acceleration
    org.set_genome_size([9, 4, 2])
    orgs.append(org)

env.propagate_organisms(orgs)

df = pd.DataFrame(columns=['x', 'y', 'name', 'iteration'])
iterations = 1000
for i in range(iterations):
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


@app.callback(Output('environment', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_environment(n):
    fig = px.scatter(df.loc[df['iteration'] == n],
                     x="x",
                     y="y",
                     color="name",
                     range_x=[0, 1],
                     range_y=[0, 1])
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
