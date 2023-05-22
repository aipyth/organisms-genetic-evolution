import os
import glob
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

import traces as trs

RESULT_DIR_TEMPLATE = 'results/result_*'

expms = {}

for record_dir in glob.glob(RESULT_DIR_TEMPLATE):
    org_loc = pq.read_table(os.path.join(record_dir,
                                         'organisms_locations')).to_pandas()
    org_loc.reset_index(drop=True, inplace=True)
    with open(os.path.join(record_dir, 'metadata.json'),
              mode='r',
              encoding='utf-8') as fp:
        metadata = json.load(fp)
    expms[record_dir] = {
        'metadata': metadata,
        'org_loc': org_loc,
        'visible': True  # Add a 'visible' key to track dataset visibility
    }

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

metadata_keys = set()
for dataset in expms.values():
    metadata_keys.update(dataset['metadata'].keys())

metadata_keys = sorted(metadata_keys)

metadata_value_options = {}
for key in metadata_keys:
    values_options = []
    for dataset in expms.values():
        if key not in dataset['metadata']:
            continue
        values_options.append({
            'label': str(dataset['metadata'].get(key)),
            'value': str(dataset['metadata'].get(key))
        })
    metadata_value_options[key] = [
        dict(s) for s in set(frozenset(d.items()) for d in values_options)
    ]

app.layout = html.Div(children=[
    html.Div(
        children=[
            # Filter data by warious params
            html.Form(
                children=[
                    html.Div(children=[
                        dbc.Card([
                            html.Label(
                                f'Filter Datasets by {str(key).upper()}',
                                className='card-title'),
                            dcc.Dropdown(id=f'dropdown-{key}',
                                         options=metadata_value_options[key],
                                         multi=True,
                                         className='dropdown')
                        ],
                                 className='m-1 p-2',
                                 style={'min-width': '400px'}),
                    ]) for key in metadata_keys
                ],
                style={
                    'display': 'flex',
                    'flex-direction': 'row',
                    'flex-wrap': 'wrap',
                    'width': '90vw',
                    #   'height': '20vh',
                    # 'overflow': 'scroll',
                }),

            # Plots
            html.Div(children=[
                html.Div(children=[
                    html.Div(children=[
                        html.Label('MA Window:'),
                        dcc.Input(id='evolution-by-time-window',
                                  value=80,
                                  type='number'),
                    ],
                             style={'display': 'flex'}),
                    dcc.Graph(id='evolution-by-time-graph',
                              style={
                                  'display': 'flex',
                                  'width': '100%',
                                  'height': '600px',
                              }),
                ]),
                html.Div(children=[
                    html.Div(children=[
                        html.Label('MA Window:'),
                        dcc.Input(id='evolution-by-generation-window',
                                  value=1,
                                  type='number'),
                    ],
                             style={'display': 'flex'}),
                    dcc.Graph(id='evolution-by-generation-graph',
                              style={
                                  'display': 'flex',
                                  'width': '100%',
                                  'height': '600px',
                              }),
                ]),
            ],
                     style={
                         'display': 'flex',
                         'flex-direction': 'column'
                     })
        ],
        style={
            'display': 'flex',
            'flex-direction': 'column',
        }),
])

KEYS_IN_HOVERTEMPLATE = [
    'genome_size', 'remove_dead_organisms', 'elitism', 'generation_time'
]


@app.callback(Output('evolution-by-time-graph', 'figure'),
              Input('evolution-by-time-window', 'value'),
              [Input(f'dropdown-{key}', 'value') for key in metadata_keys])
def update_evolution_by_time_graph(window, *filter_values):
    data = []
    for dataset in expms.values():
        org_loc = dataset['org_loc']
        visible = dataset['visible']
        metadata = dataset['metadata']
        hovertemplate = '(%{x} %{y})<br>'
        match = True
        for key, values in zip(metadata_keys, filter_values):
            metadata_value = metadata.get(key)
            if values is not None and str(metadata_value) not in values:
                match = False
                break
            if (values and str(metadata_value)
                    in values) or (key in KEYS_IN_HOVERTEMPLATE):
                hovertemplate += f'<br>{key}: {metadata_value}'
        if visible and match:
            traces = trs.get_energy_traces_over_time(
                org_loc,
                window=80 if not window or window <= 0 else window,
                legendgroup=str(metadata),
                hovertemplate=hovertemplate)
            data.extend(traces)

    layout = go.Layout(title='Evolution Statistics by Time',
                       xaxis=dict(title='Time'))
    return go.Figure(data=data, layout=layout)


@app.callback(Output('evolution-by-generation-graph', 'figure'),
              Input('evolution-by-generation-window', 'value'),
              [Input(f'dropdown-{key}', 'value') for key in metadata_keys])
def update_evolution_by_generation_graph(window, *filter_values):
    data = []
    for result_dir, dataset in expms.items():
        if not dataset['visible']:
            continue
        org_loc = dataset['org_loc']
        metadata = dataset['metadata']
        hovertemplate = '(%{x} %{y})<br>'
        match = True
        for key, values in zip(metadata_keys, filter_values):
            metadata_value = metadata.get(key)
            if values is not None and str(metadata_value) not in values:
                match = False
                break
            if (values and str(metadata_value)
                    in values) or (key in KEYS_IN_HOVERTEMPLATE):
                hovertemplate += f'<br>{key}: {metadata_value}'
        if match:
            traces = trs.get_energy_traces_over_generation(
                org_loc,
                window=max(window or 0, 0),
                legendgroup=result_dir,
                hovertemplate=hovertemplate)
            data.extend(traces)

    layout = go.Layout(title='Evolution Statistics by Generation',
                       xaxis=dict(title='Generation'))
    return go.Figure(data=data, layout=layout)


if __name__ == '__main__':
    app.run_server(debug=True)
