from plotly import express as px
import plotly.graph_objs as go
import numpy as np


def get_energy_traces_over_time(org_loc,
                                window=80,
                                legendgroup='',
                                hovertemplate='x: %{x}; y: %{y}'):
    moving_age_mean = org_loc[[
        'iteration', 'age'
    ]].groupby('iteration').mean().rolling(window).mean()
    moving_age_max = org_loc[[
        'iteration', 'age'
    ]].groupby('iteration').max().rolling(window).mean()
    moving_energy_mean = org_loc[[
        'iteration', 'energy'
    ]].groupby('iteration').mean().rolling(window).mean()
    moving_energy_max = org_loc[[
        'iteration', 'energy'
    ]].groupby('iteration').max().rolling(window).mean()
    x = np.arange(len(moving_age_max))

    ma_age = go.Scatter(
        x=x,
        y=moving_age_mean['age'],
        name=f'Moving Age Average {window=}',
        mode='lines',
        legendgroup=legendgroup,
        hovertemplate=hovertemplate,
    )
    mm_age = go.Scatter(
        x=x,
        y=moving_age_max['age'],
        name=f'Moving Age Max {window=}',
        mode='lines',
        legendgroup=legendgroup,
        hovertemplate=hovertemplate,
    )
    ma_energy = go.Scatter(
        x=x,
        y=moving_energy_mean['energy'],
        name=f'Moving Energy Average {window=}',
        mode='lines',
        legendgroup=legendgroup,
        hovertemplate=hovertemplate,
    )
    mm_energy = go.Scatter(
        x=x,
        y=moving_energy_max['energy'],
        name=f'Moving Energy Max {window=}',
        mode='lines',
        legendgroup=legendgroup,
        hovertemplate=hovertemplate,
    )
    return mm_energy, ma_energy


def get_energy_traces_over_generation(org_loc,
                                      window=1,
                                      legendgroup='',
                                      hovertemplate='x: %{x}; y: %{y}'):
    age_mean = org_loc[['generation', 'age']].groupby('generation').mean()

    age_max = org_loc[['generation', 'age']].groupby('generation').max()

    energy_mean = org_loc[['generation',
                           'energy']].groupby('generation').mean()

    energy_max = org_loc[['generation', 'energy']].groupby('generation').max()

    energy_max_ma = energy_max.rolling(window).mean()

    x = np.arange(len(age_max))

    age_avg = go.Scatter(
        x=x,
        y=age_mean['age'],
        name='Age Average',
        mode='lines',
        legendgroup=legendgroup,
        hovertemplate=hovertemplate,
    )

    age_max = go.Scatter(
        x=x,
        y=age_max['age'],
        name='Age Max',
        mode='lines',
        legendgroup=legendgroup,
        hovertemplate=hovertemplate,
    )

    energy_mean = go.Scatter(
        x=x,
        y=energy_mean['energy'],
        name='Energy Average',
        mode='lines',
        legendgroup=legendgroup,
        hovertemplate=hovertemplate,
    )

    energy_max = go.Scatter(
        x=x,
        y=energy_max['energy'],
        name='Energy Max',
        mode='lines',
        # marker_color='rgba(255, 180, 0, 100)',
        legendgroup=legendgroup,
        hovertemplate=hovertemplate,
        line=dict(width=1),
    )

    energy_max_ma = go.Scatter(
        x=x,
        y=energy_max_ma['energy'],
        name='MA Energy Max',
        mode='lines',
        #  marker_color='rgba(255, 50, 0, 100)',
        legendgroup=legendgroup,
        hovertemplate=hovertemplate,
        line=dict(width=4),
    )

    return energy_mean, energy_max, energy_max_ma


def get_worst_best_energy_traces_over_time(org_loc, legendgroup=''):
    pass
