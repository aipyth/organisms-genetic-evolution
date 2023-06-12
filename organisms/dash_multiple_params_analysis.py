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

ITERATIONS = 8000

ORG_LOC_DROP_COLUMNS = []
EATEN_STATS_DROP_COLUMNS = []
DROP_NETWORK_WEIGHTS_COLUMNS = True


def read_results_dir(template):
    expms = {}
    for record_dir in glob.glob(template):
        org_loc = pq.read_table(os.path.join(
            record_dir, 'organisms_locations')).to_pandas()
        eaten_stats = pq.read_table(os.path.join(record_dir,
                                                 'eaten_food')).to_pandas()
        # organisms_stats = pq.read_table(
        #     os.path.join(record_dir, 'organisms_stats')).to_pandas()
        org_loc.reset_index(drop=True, inplace=True)
        eaten_stats.reset_index(drop=True, inplace=True)

        org_loc = org_loc[org_loc['iteration'] < ITERATIONS]
        eaten_stats = eaten_stats[eaten_stats['iteration'] < ITERATIONS]

        # org_loc = org_loc.drop(columns=ORG_LOC_DROP_COLUMNS)
        # eaten_stats = eaten_stats.drop(columns=EATEN_STATS_DROP_COLUMNS)

        if DROP_NETWORK_WEIGHTS_COLUMNS:
            columns_to_drop = []
            genome_length = len(org_loc.iloc[0].loc['genome_shape'])
            print(org_loc.iloc[0].loc['genome_shape'])
            print(genome_length)
            for i in range(genome_length - 1):
                columns_to_drop.extend(
                    (f'genome_{i}_weights', f'genome_{i}_bias'))
            print(org_loc.columns)
            org_loc = org_loc.drop(columns=columns_to_drop)
        # organisms_stats.reset_index(drop=True, inplace=True)
        # organisms_genome = pd.DataFrame()
        # for genome_file_name in glob.glob(
        #         os.path.join(record_dir, 'organisms_genome_*.csv')):
        #     organisms_genome = pd.concat(
        #         [organisms_genome,
        #          pd.read_csv(genome_file_name)])
        with open(os.path.join(record_dir, 'metadata.json'),
                  mode='r',
                  encoding='utf-8') as fp:
            metadata = json.load(fp)
        expms[record_dir] = {
            'metadata': metadata,
            'org_loc': org_loc,
            'eaten_stats': eaten_stats,
            # 'organisms_stats': organisms_stats,
            # 'organisms_genome': organisms_genome,
            'visible': True  # Add a 'visible' key to track dataset visibility
        }
    return expms


expms = read_results_dir(RESULT_DIR_TEMPLATE)


def wrap_metadata_value(value):
    if type(value) != dict:
        return str(value)
    ret = ''
    if 'name' in value:
        ret += value['name'] + '_'
    if 'object' in value:
        ret += '('
        for k, v in value['object'].items():
            if k == '_encoding':
                continue
            ret += k + '_'
            ret += str(v) if type(v) != dict else v['name']
        ret += ')'
    return ret


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# Get unique dataset names (keys in expms)
dataset_names = list(expms.keys())

# Generate options for checkboxes
checkbox_options = [{'label': name, 'value': name} for name in dataset_names]

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
            'label':
            str(wrap_metadata_value(dataset['metadata'].get(key))),
            'value':
            str(wrap_metadata_value(dataset['metadata'].get(key)))
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
                        dbc.Card(
                            [
                                html.Label(
                                    f'Filter Datasets by {str(key).upper()}',
                                    className='card-title'),
                                dcc.Checklist(
                                    id=f'dropdown-{key}',
                                    options=metadata_value_options[key],
                                )
                                # multi=True,
                                # className='dropdown')
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
            # Dataset checkboxes
            html.Div(id='dataset-checkboxes'),
            # Plots
            html.Div(
                children=[
                    # Energy by Time
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
                    # Energy by Generation
                    html.Div(children=[
                        html.Div(children=[
                            html.Label('MA Window:'),
                            dcc.Input(id='evolution-by-generation-window',
                                      value=5,
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
                    # Highest Lowest Energy Graph
                    html.Div(children=[
                        dcc.Graph(
                            id='highest-lowest-energy-by-generation-graph',
                            style={
                                'display': 'flex',
                                'width': '100%',
                                'height': '600px',
                            }),
                    ]),
                    # Food-to-Movement Ratio Graph
                    html.Div(children=[
                        html.Div(children=[
                            html.Label('MA Window:'),
                            dcc.Input(id='food-to-movement-ratio-window',
                                      value=360,
                                      type='number'),
                        ],
                            style={'display': 'flex'}),
                        dcc.Graph(id='food-to-movement-ratio-graph',
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
    # 'genome_size', 'remove_dead_organisms', 'elitism', 'generation_time'
]


@app.callback(Output('dataset-checkboxes', 'children'),
              Output('evolution-by-time-graph', 'figure'),
              Output('evolution-by-generation-graph', 'figure'),
              Output('highest-lowest-energy-by-generation-graph', 'figure'),
              Output('food-to-movement-ratio-graph', 'figure'),
              Input('evolution-by-time-window', 'value'),
              Input('evolution-by-generation-window', 'value'),
              Input('food-to-movement-ratio-window', 'value'),
              [Input(f'dropdown-{key}', 'value') for key in metadata_keys])
def update_dataset(ev_time_window, ev_generation_window, ftm_window,
                   *filter_values):
    valid_datasets = {}
    for dataset_name, dataset in expms.items():
        metadata = dataset['metadata']
        hovertemplate = '(%{x} %{y})<br>'
        hovertemplate += f'{dataset_name}<br>'
        match = True
        for key, values in zip(metadata_keys, filter_values):
            # metadata_value = metadata.get(key)
            metadata_value = str(wrap_metadata_value(metadata.get(key)))
            if values is not None and str(metadata_value) not in values:
                match = False
                expms[dataset_name]['visible'] = False
                break
            if (values and str(metadata_value)
                    in values) or (key in KEYS_IN_HOVERTEMPLATE):
                hovertemplate += f'<br> {key}: {wrap_metadata_value(metadata_value)}'
        if match:
            expms[dataset_name]['visible'] = True
            expms[dataset_name]['hovertemplate'] = hovertemplate
            valid_datasets[dataset_name] = expms[dataset_name]

    # Generate checkboxes
    checkboxes = [
        dbc.Form([
            dbc.Checkbox(id=f'checkbox-{name}',
                         className='form-check-input',
                         value=True),
            dbc.Label(name,
                      html_for=f'checkbox-{name}',
                      className='form-check-label')
        ]) for name in valid_datasets.keys()
    ]
    # print(checkboxes)
    return (
        checkboxes,
        update_evolution_by_time_graph(valid_datasets, ev_time_window),
        update_evolution_by_generation_graph(valid_datasets,
                                             ev_generation_window),
        update_highest_lowest_energy_by_generation_graph(valid_datasets),
        update_food_to_movement_ratio_graph(valid_datasets, ftm_window),
    )


# @app.callback(
#     Output('evolution-by-time-graph', 'figure'),
#     Input('evolution-by-time-window', 'value'),
#     Input('dataset-checkboxes', 'children'),
# )
# *[Input(f'dropdown-{key}', 'value') for key in metadata_keys],
# *[Input(f'checkbox-{name}', 'checked') for name in dataset_names])
def update_evolution_by_time_graph(datasets, window):
    data = []
    # checkbox_values = filter_values[-len(dataset_names):]
    # filter_values = filter_values[:-len(dataset_names)]
    for dataset_name, dataset in datasets.items():
        # checkbox_id = f'checkbox-{dataset_name}'
        # try:
        #     checkbox_value = dash.callback_context.inputs[checkbox_id]
        # except KeyError:
        #     continue
        # if checkbox_value is None or not checkbox_value:
        #     continue
        org_loc = dataset['org_loc']
        # visible = dataset['visible']
        metadata = dataset['metadata']
        hovertemplate = dataset['hovertemplate']
        # match = True
        # for key, values in zip(metadata_keys, filter_values):
        #     metadata_value = metadata.get(key)
        #     # if values is not None and str(metadata_value) not in values:
        #     #     match = False
        #     #     break
        #     if (values and str(metadata_value)
        #             in values) or (key in KEYS_IN_HOVERTEMPLATE):
        #         hovertemplate += f'<br>{key}: {metadata_value}'
        # if visible:
        legendgroup_items = hovertemplate.split('<br>')[3:]
        legendgroup = '<br>'.join(legendgroup_items)
        # print(legendgroup)

        traces = trs.get_energy_traces_over_time(
            org_loc,
            window=80 if not window or window <= 0 else window,
            legendgroup=legendgroup,
            hovertemplate=hovertemplate)
        data.extend(traces)

    layout = go.Layout(title='Evolution Energy Levels Statistics by Time',
                       xaxis=dict(title='Time'))
    return go.Figure(data=data, layout=layout)


# @app.callback(Output('evolution-by-generation-graph', 'figure'),
#               Input('evolution-by-generation-window', 'value'),
#               [Input(f'dropdown-{key}', 'value') for key in metadata_keys],
#               [Input(f'checkbox-{name}', 'checked') for name in dataset_names])
def update_evolution_by_generation_graph(datasets, window):
    data = []
    # checkbox_values = filter_values[-len(dataset_names):]
    # filter_values = filter_values[:-len(dataset_names)]
    for dataset_name, dataset in datasets.items():
        # checkbox_id = f'checkbox-{dataset_name}'
        # try:
        #     checkbox_value = dash.callback_context.inputs[checkbox_id]
        # except KeyError:
        #     continue
        # if checkbox_value is None or not checkbox_value:
        #     continue
        # if not dataset['visible']:
        #     continue
        org_loc = dataset['org_loc']
        metadata = dataset['metadata']
        hovertemplate = dataset['hovertemplate']
        # match = True
        # for key, values in zip(metadata_keys, filter_values):
        #     metadata_value = metadata.get(key)
        #     if values is not None and str(metadata_value) not in values:
        #         match = False
        #         break
        #     if (values and str(metadata_value)
        #             in values) or (key in KEYS_IN_HOVERTEMPLATE):
        #         hovertemplate += f'<br>{key}: {metadata_value}'
        # if match:
        legendgroup_items = hovertemplate.split('<br>')[3:]
        legendgroup = '<br>'.join(legendgroup_items)
        traces = trs.get_energy_traces_over_generation(
            org_loc,
            window=max(window or 0, 0),
            legendgroup=legendgroup,
            hovertemplate=hovertemplate)
        data.extend(traces)

    layout = go.Layout(
        title='Evolution Energy Levels Statistics by Generation',
        xaxis=dict(title='Generation'))
    return go.Figure(data=data, layout=layout)


# @app.callback(Output('highest-lowest-energy-by-generation-graph', 'figure'),
#               [Input(f'dropdown-{key}', 'value') for key in metadata_keys],
#               [Input(f'checkbox-{name}', 'checked') for name in dataset_names])
def update_highest_lowest_energy_by_generation_graph(datasets):
    data = []
    # checkbox_values = filter_values[-len(dataset_names):]
    # filter_values = filter_values[:-len(dataset_names)]
    for dataset_name, dataset in datasets.items():
        # checkbox_id = f'checkbox-{dataset_name}'
        # try:
        #     checkbox_value = dash.callback_context.inputs[checkbox_id]
        # except KeyError:
        #     continue
        # if checkbox_value is None or not checkbox_value:
        #     continue
        # if not dataset['visible']:
        #     continue
        org_loc = dataset['org_loc']
        metadata = dataset['metadata']
        hovertemplate = dataset['hovertemplate']
        # match = True
        # for key, values in zip(metadata_keys, filter_values):
        #     metadata_value = metadata.get(key)
        #     if values is not None and str(metadata_value) not in values:
        #         match = False
        #         break
        #     if (values and str(metadata_value)
        #             in values) or (key in KEYS_IN_HOVERTEMPLATE):
        #         hovertemplate += f'<br>{key}: {metadata_value}'
        # if match:
        legendgroup_items = hovertemplate.split('<br>')[3:]
        legendgroup = '<br>'.join(legendgroup_items)
        traces = trs.get_highest_lowest_energy_traces_over_generation(
            org_loc, legendgroup=legendgroup, hovertemplate=hovertemplate)
        data.extend(traces)

    layout = go.Layout(title_text='Evolution of Organisms with Highest and ' +
                       'Lowest Mean Energy Over Time',
                       xaxis_title='Generation',
                       yaxis_title='Mean Energy')
    return go.Figure(data=data, layout=layout)


# @app.callback(Output('food-to-movement-ratio-graph', 'figure'),
#               Input('food-to-movement-ratio-window', 'value'),
#               [Input(f'dropdown-{key}', 'value') for key in metadata_keys],
#               [Input(f'checkbox-{name}', 'checked') for name in dataset_names])
def update_food_to_movement_ratio_graph(datasets, window):
    data = []
    # checkbox_values = filter_values[-len(dataset_names):]
    # filter_values = filter_values[:-len(dataset_names)]
    for dataset_name, dataset in datasets.items():
        # checkbox_id = f'checkbox-{dataset_name}'
        # try:
        #     checkbox_value = dash.callback_context.inputs[checkbox_id]
        # except KeyError:
        #     continue
        # if checkbox_value is None or not checkbox_value:
        #     continue
        # if not dataset['visible']:
        #     continue
        metadata = dataset['metadata']
        hovertemplate = dataset['hovertemplate']
        # match = True
        # for key, values in zip(metadata_keys, filter_values):
        #     metadata_value = metadata.get(key)
        #     if values is not None and str(metadata_value) not in values:
        #         match = False
        #         break
        #     if (values and str(metadata_value)
        #             in values) or (key in KEYS_IN_HOVERTEMPLATE):
        #         hovertemplate += f'<br>{key}: {metadata_value}'
        # if match:
        legendgroup_items = hovertemplate.split('<br>')[3:]
        legendgroup = '<br>'.join(legendgroup_items)
        org_loc = dataset['org_loc']
        eaten_stats = dataset['eaten_stats']
        traces = trs.get_food_to_movement_ratio_traces(
            org_loc,
            eaten_stats,
            window=max(window or 0, 0),
            legendgroup=legendgroup,
            hovertemplate=hovertemplate)
        data.extend(traces)

    layout = go.Layout(title_text='Worst and Best Food-to-Movement' +
                       'Ratio Over Time',
                       xaxis_title='Iteration',
                       yaxis_title='Food-to-Movement Ratio')
    return go.Figure(data=data, layout=layout)


if __name__ == '__main__':
    app.run_server(debug=True)
