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
        line=dict(width=3),
    )

    return energy_mean, energy_max, energy_max_ma


def get_highest_lowest_energy_traces_over_generation(
        org_loc, legendgroup='', hovertemplate='x: %{x}; y: %{y}'):
    org_loc_sorted = org_loc.sort_values(['index', 'iteration'])
    org_loc_sorted['energy_diff'] = org_loc_sorted.groupby(
        'index')['energy'].diff()

    org_energy_mean = org_loc_sorted.groupby(['index',
                                              'generation'])['energy'].mean()
    organism_with_highest_energy_per_gen = org_energy_mean.groupby(
        'generation').idxmax()
    organism_with_lowest_energy_per_gen = org_energy_mean.groupby(
        'generation').idxmin()
    # Get the energy levels of the organisms with the highest and lowest mean energy
    highest_energy_levels = org_energy_mean[
        organism_with_highest_energy_per_gen].groupby('generation').mean()
    lowest_energy_levels = org_energy_mean[
        organism_with_lowest_energy_per_gen].groupby('generation').mean()

    lowest_energy_level_trace = go.Scatter(x=lowest_energy_levels.index,
                                           y=lowest_energy_levels.values,
                                           mode='lines',
                                           name='Lowest Energy',
                                           line=dict(color='firebrick'),
                                           legendgroup=legendgroup,
                                           hovertemplate=hovertemplate,
                                           fill='tozeroy')

    highest_energy_level_trace = go.Scatter(x=highest_energy_levels.index,
                                            y=highest_energy_levels.values,
                                            mode='lines',
                                            name='Highest Energy',
                                            line=dict(color='royalblue'),
                                            legendgroup=legendgroup,
                                            hovertemplate=hovertemplate,
                                            fill='tonexty')
    return lowest_energy_level_trace, highest_energy_level_trace


def get_food_to_movement_ratio_traces(
    org_loc,
    eaten_stats,
    window=240,
    legendgroup='',
    hovertemplate='x: %{x}; y: %{y}',
):
    # Calculate the total movements and food acquisitions at each time step
    movements_per_time = org_loc.groupby(['index', 'iteration']).size()
    food_acquisitions_per_time = eaten_stats.groupby(
        ['organism_id', 'iteration']).size()

    movements_per_time.index = movements_per_time.index.set_names(
        food_acquisitions_per_time.index.names)

    # Calculate the Food-to-Movement ratio at each time step
    ratio_per_time = movements_per_time / food_acquisitions_per_time

    # Fill in any missing values
    ratio_per_time = ratio_per_time.fillna(ratio_per_time.max() * 2)

    # Find the organisms with the worst and best Food-to-Movement ratio at each time step
    worst_ratio = ratio_per_time.groupby('iteration').max().rolling(
        window).mean()
    best_ratio = ratio_per_time.groupby('iteration').min().rolling(
        window).mean()

    worst_ratio_trace = go.Scatter(x=worst_ratio.index,
                                   y=worst_ratio.values,
                                   mode='lines',
                                   name='Worst Ratio',
                                   legendgroup=legendgroup,
                                   hovertemplate=hovertemplate,
                                   line=dict(color='firebrick'))

    best_ratio_trace = go.Scatter(x=best_ratio.index,
                                  y=best_ratio.values,
                                  mode='lines',
                                  name='Best Ratio',
                                  legendgroup=legendgroup,
                                  hovertemplate=hovertemplate,
                                  line=dict(color='royalblue'))

    return worst_ratio_trace, best_ratio_trace
