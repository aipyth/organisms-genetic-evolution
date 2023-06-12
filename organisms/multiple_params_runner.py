from datetime import datetime
import itertools
import os
import shutil
import contextlib
import pathlib
import json
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import run
import environment
import evolve
import encode
import plot
from utils import ClassEncoder


def generate_argument_variations(arguments):
    keys = arguments.keys()
    values = arguments.values()
    variations = list(itertools.product(*values))
    argument_variations = []

    for variation in variations:
        argument_variation = dict(zip(keys, variation))

        results_dir = generate_results_dir(argument_variation)
        argument_variation['results_dir'] = results_dir

        argument_variations.append(argument_variation)

    return argument_variations


def generate_results_dir(arguments):
    selected_keys = [
        'iterations', 'genome_size', 'elitism', 'remove_dead_organisms',
        'generation_time', 'food_energy', 'mutation', 'crossover'
    ]
    dir_name = "result_"
    for key, value in arguments.items():
        if key not in selected_keys:
            continue
        if isinstance(value, str):
            dir_name += f"{key}_{value.replace(',', '-')}_"
        elif isinstance(value, (int, float)):
            dir_name += f"{key}_{str(value).replace(',', '-')}_"
        elif isinstance(value, list):
            dir_name += f"{key}_{'-'.join(str(v) for v in value)}_"
        elif isinstance(value,
                        (evolve.Selection, evolve.Crossover, evolve.Mutation)):
            dir_name += f"{value.__class__.__name__}_"
        else:
            dir_name += f"{key}_"

    return os.path.join('results',
                        dir_name[:-1])  # Remove the trailing underscore


def run_organisms_environment(variation):
    if os.path.exists(variation['results_dir']):
        # variation['results_dir'] += '-' + str(time.time())[-3:]
        variation['results_dir'] += f'-{str(datetime.now())}'

    with contextlib.suppress(Exception):
        os.mkdir(variation['results_dir'])

    metadata_filename = os.path.join(variation['results_dir'], 'metadata.json')
    with open(metadata_filename, 'w') as file:
        json.dump(variation, file, cls=ClassEncoder)

    run_tool = run.OrganismsSimpleEnvironmentRunTool(**variation)
    run_tool.run()


organism_vision_range = 2

vision = environment.SectorVision(
    distance=organism_vision_range,
    distance_sectors=4,
    angle_sectors=4,
)
genome_shapes = [
    # [vision.organism_input_shape, 3, 2],
    [vision.organism_input_shape, 6, 2],
    # [vision.organism_input_shape, 9, 2],
    # [vision.organism_input_shape, 9, 9, 2],
    # [vision.organism_input_shape, 9, 4, 2],
    # [vision.organism_input_shape, 18, 9, 2],
    [vision.organism_input_shape, 18, 9, 9, 2],
    [vision.organism_input_shape, 12, 6, 2],
    # [vision.organism_input_shape, 24, 12, 2],
]

fitness = evolve.FoodConsumnptionToDistanceFitness()
# fitness = evolve.EnergyFitness()

# generation_time = 80
generation_time = 60
generations = 100
# generations = 60

arguments = {
    'start_organism_number': [140],
    'width': [20],
    'height': [20],
    'iterations': [generation_time * generations],
    'generation_time': [generation_time],
    'organism_size': [0.16],
    'food_size': [0.05],
    'organism_vision_range': [organism_vision_range],
    'vision': [vision],
    'food_energy': [2],
    'food_appearance_number_rate': [1.4],
    'energy_decrease_rate': [0.7],
    'encoding': [encode.RealValued()],
    'selection': [
        # evolve.TruncationSelection(fitness=fitness, n=10),
        # evolve.TruncationSelection(fitness=fitness, n=15),
        evolve.TruncationSelection(fitness=fitness, n=30),
        # evolve.TruncationSelection(fitness=fitness, n=40),
        # evolve.TruncationSelection(fitness=fitness, n=50),
    ],
    'crossover': [
        # evolve.SASBXCrossover(alpha=1.1),
        # evolve.SimpleAdaptiveSBXCrossover(alpha=1.1),
        # evolve.SBXCrossover(n=1),
        # evolve.SBXCrossover(n=2),
        # evolve.SBXCrossover(n=8),
        evolve.ArithmeticCrossover(),
        # evolve.BLXCrossover(alpha=0.5),
        # evolve.SBXCrossover(n=12),
    ],
    'mutation': [
        # evolve.GaussianMutation(mu=0, sigma=0.05, p=0.05),
        # evolve.NonUniformMutation(b=6, p=0.1, T=generation_time * generations),
        evolve.UniformMutation(low=-0.1, high=0.1, p=0.1),
    ],
    'elitism': [10],
    'genome_size':
    genome_shapes,
    'food_particles_at_start': [80],
    'remove_dead_organisms': [True],
    'init_organism_genome_distibution_mu': [0],
    'init_organism_genome_distibution_sigma': [0.2]
}

# as we are using steady-state GA in this implementation we need to set a small number of generation_time
generation_time_steady = 2
# iterations_steady = 8000
iterations_steady = 3500
# iterations_steady = generation_time * generations
# print('Running evolution with generative GA 2')
# variations = generate_argument_variations(arguments_2)
#
# print(f'Total number of configurations (variations): {len(variations)}')
#
# for i, var in enumerate(variations):
#     print(f'Running sample №{i+1}/{len(variations)}')
#     run_organisms_environment(var)

arguments_steady = {
    'start_organism_number': [80],
    'width': [20],
    'height': [20],
    'iterations': [iterations_steady],
    'generation_time': [generation_time_steady],
    'organism_size': [0.12],
    'food_size': [0.05],
    'organism_vision_range': [organism_vision_range],
    'vision': [vision],
    'food_energy': [2.5],
    'food_appearance_number_rate': [0.8],
    'energy_decrease_rate': [0.06],
    'encoding': [encode.RealValued()],
    'selection': [
        # evolve.TruncationSelection(fitness=fitness, n=10),
        # evolve.TruncationSelection(fitness=fitness, n=15),
        # evolve.TruncationSelection(fitness=fitness, n=20),
        evolve.TruncationSelection(fitness=fitness, n=2),
        # evolve.TruncationSelection(fitness=fitness, n=50),
    ],
    'crossover': [
        # evolve.SASBXCrossover(alpha=1.1),
        # evolve.SimpleAdaptiveSBXCrossover(alpha=1.1),
        # evolve.SBXCrossover(n=2),
        evolve.ArithmeticCrossover(),
    ],
    'mutation': [
        evolve.NonUniformMutation(b=5, p=0.05,
                                  T=generation_time * generations),
    ],
    'elitism': [80],
    'genome_size':
    genome_shapes,
    'food_particles_at_start': [30],
    'remove_dead_organisms': [True],
    'init_organism_genome_distibution_mu': [0],
    'init_organism_genome_distibution_sigma': [0.1]
}

if __name__ == '__main__':
    print('Running evolution with generative GA')
    variations = generate_argument_variations(arguments)

    print(f'Total number of configurations (variations): {len(variations)}')

    for i, var in enumerate(variations):
        print(f'Running sample №{i+1}/{len(variations)}')
        run_organisms_environment(var)

    # print('Running evolution with steady-state genetic algorithm')
    # variations_steady = generate_argument_variations(arguments_steady)

    # print(
    #     f'Total number of configurations (variations): {len(variations_steady)}'
    # )
    # for i, var in enumerate(variations_steady):
    #     print(f'Running sample №{i+1}/{len(variations_steady)}')
    #     run_organisms_environment(var)
