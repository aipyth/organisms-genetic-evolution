import json
from tqdm import tqdm
import os
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from environment import Simple2DContinuousEnvironment, Vision
from organism import Organism
import evolve
import encode


class OrganismsSimpleEnvironmentRunTool:
    ORGANISMS_LOCATIONS = 'organisms_locations'
    FOOD_LOCATIONS = 'food_locations'
    EATEN_FOOD = 'eaten_food'
    ORGANISMS_STATS = 'organisms_stats'
    ORGANISMS_GENOME = 'organisms_genome_'

    def __init__(self, start_organism_number: int, width: float, height: float,
                 iterations: int, generation_time: int, organism_size: float,
                 food_size: float, organism_vision_range: float,
                 results_dir: str, vision: Vision, food_energy: float,
                 food_appearance_number_rate: float, encoding: encode.Encoding,
                 selection: evolve.Selection, crossover: evolve.Crossover,
                 mutation: evolve.Mutation, elitism: int,
                 genome_size: list[int], food_particles_at_start: int,
                 remove_dead_organisms: bool,
                 energy_decrease_rate: float) -> None:
        self._start_organisms_num = start_organism_number
        self._width = width
        self._height = height
        self._iterations = iterations
        self._generation_time = generation_time
        self._organism_size = organism_size
        self._food_size = food_size
        self._organism_vision_range = organism_vision_range
        self._results_dir = results_dir
        self._vision = vision
        self._food_energy = food_energy
        self._food_appearance_number_rate = food_appearance_number_rate
        self._encoding = encoding
        self._selection = selection
        self._crossover = crossover
        self._mutation = mutation
        self._elitism = elitism
        self._genome_size = genome_size
        self._food_particles_at_start = food_particles_at_start
        self._remove_dead_organisms = remove_dead_organisms
        self._energy_decrease_rate = energy_decrease_rate
        self._organisms_genome_file_count = 0

    def run(self):
        self.env = Simple2DContinuousEnvironment(
            width=self._width,
            height=self._height,
            vision=self._vision,
            vision_range=self._organism_vision_range,
            organism_size=self._organism_size,
            food_size=self._food_size,
            food_energy=self._food_energy,
            food_appearance_number_rate=self._food_appearance_number_rate,
            remove_dead_organisms=self._remove_dead_organisms)
        self.ev = evolve.Evolve(
            encoding=self._encoding,
            selection=self._selection,
            crossover=self._crossover,
            mutation=self._mutation,
            elitism=self._elitism,
        )
        generation = 0

        for _ in range(self._start_organisms_num):
            org = Organism()
            org.set_genome_size(self._genome_size)
            org.set_energy_decrease_rate(self._energy_decrease_rate)
            self.env.add_organism(org)

        for _ in range(self._food_particles_at_start):
            self.env.add_food()

        INTERMEDIATE_WRITE_THRESHOLD = self._generation_time * 2

        # organisms_loc_df = pd.DataFrame(
        #     [], columns=['index', "x", "y", "name", "iteration"])
        organisms_loc_df = pd.DataFrame()
        food_loc_df = pd.DataFrame(
            [], columns=['index', "x", "y", "name", "iteration"])
        eaten_stats = pd.DataFrame([],
                                   columns=[
                                       'organism_id', 'food_location',
                                       'energy_taken', 'iteration'
                                   ])
        # organisms_stats = pd.DataFrame([], columns=[
        #                                 'id', 'name',
        #                                 'age',
        #                                 'parents',
        #                                 'iteration', 'energy'
        #                             ])
        # organisms_stats.set_index(['id', 'iteration'])
        # organisms_stats = pd.DataFrame()
        # organisms_genome = pd.DataFrame()

        with tqdm(total=self._iterations, desc="Simulating organisms") as pbar:
            for i in range(self._iterations):

                def on_organism_death(organism: Organism):
                    x = organism.x[0]
                    y = organism.x[1]
                    theta = organism.r
                    record = {
                        "index": i,
                        "x": x,
                        "y": y,
                        "theta": theta,
                        "name": r.name,
                        "iteration": i,
                        "generation": generation,
                    }
                    record["v"] = organism.v
                    record["a"] = organism.a
                    record["energy"] = organism.energy
                    record['age'] = organism.age
                    record['distance_traveled'] = organism.distance_traveled
                    record['genome_shape'] = organism._size
                    record['is_alive'] = organism.is_alive
                    record['parent'] = [
                        parent.id for parent in organism._parents
                    ]
                    for num, allele in enumerate(organism.genome):
                        record[f'genome_{num}_weights'] = json.dumps(
                            allele[0].tolist())
                        record[f'genome_{num}_bias'] = json.dumps(
                            allele[1].tolist())
                    # records.append(record)
                    # d = {
                    #     'id': [organism.id],
                    #     'name': [organism.name],
                    #     'genome_shape': [organism._size],
                    #     # 'genome': [organism.genome],
                    #     'age': [organism.age],
                    #     'iteration': [i],
                    #     'energy': [organism.energy],
                    #     'parent':
                    #     [[parent.id for parent in organism._parents]],
                    #     'distance_traveled': [organism.distance_traveled],
                    # }
                    # for num, allele in enumerate(organism.genome):
                    #     d[f'genome_{num}_weights'] = json.dumps(
                    #         allele[0].tolist())
                    #     d[f'genome_{num}_bias'] = json.dumps(
                    #         allele[1].tolist())
                    organisms_loc_df.append(record)

                self.env.on_organism_death(on_organism_death)

                # perform one step in time
                eaten_on_step = self.env.tick()

                # evolution
                if i % self._generation_time == 0 and i != 0:
                    next_organisms = self.ev(list(self.env.organisms.values()),
                                             i)
                    # remove organisms that did not pass
                    for _, org in list(self.env.organisms.items()):
                        if org not in next_organisms:
                            self.env.remove_organism(org)
                    for org in next_organisms:
                        if org not in self.env.organisms.values():
                            org.set_energy_decrease_rate(
                                self._energy_decrease_rate)
                            self.env.add_organism(org)
                    generation += 1

                # store organisms and food locations
                organisms_loc_df = add_locations_record(self.env.organisms,
                                                        organisms_loc_df,
                                                        i,
                                                        generation=generation,
                                                        concatenate=True)
                food_loc_df = add_locations_record(self.env.food,
                                                   food_loc_df,
                                                   i,
                                                   name="food",
                                                   concatenate=True)
                eaten_on_step['iteration'] = i
                eaten_stats = pd.concat([eaten_stats, eaten_on_step],
                                        ignore_index=True)
                # pull_organisms_stats(self.env.organisms, iteration=i)
                # organisms_stats = pd.concat([
                #     organisms_stats,
                #     pull_organisms_stats(self.env.organisms, iteration=i)
                # ],
                #                             ignore_index=True)
                # if i % self._generation_time == 0 and i != 0:
                #     organisms_genome = pd.concat([
                #         organisms_genome,
                #         pull_organisms_genomes(self.env.organisms, iteration=i)
                #     ])
                pbar.set_postfix({
                    "Number of organisms": len(self.env.organisms),
                    "Gen": generation,
                })
                pbar.update(1)

                if i % max(INTERMEDIATE_WRITE_THRESHOLD, 50) == 0 and i != 0:
                    # if i % max(self._generation_time, 50) == 0 and i != 0:
                    organisms_loc = pa.Table.from_pandas(organisms_loc_df)
                    food_loc = pa.Table.from_pandas(food_loc_df)
                    eaten = pa.Table.from_pandas(eaten_stats)
                    # organisms_stats_pa = pa.Table.from_pandas(organisms_stats)

                    pq.write_to_dataset(table=organisms_loc,
                                        root_path=os.path.join(
                                            self._results_dir,
                                            self.ORGANISMS_LOCATIONS),
                                        use_legacy_dataset=False)
                    pq.write_to_dataset(table=food_loc,
                                        root_path=os.path.join(
                                            self._results_dir,
                                            self.FOOD_LOCATIONS),
                                        use_legacy_dataset=False)
                    pq.write_to_dataset(table=eaten,
                                        root_path=os.path.join(
                                            self._results_dir,
                                            self.EATEN_FOOD),
                                        use_legacy_dataset=False)
                    # pq.write_to_dataset(table=organisms_stats_pa,
                    #                     root_path=os.path.join(
                    #                         self._results_dir,
                    #                         self.ORGANISMS_STATS),
                    #                     use_legacy_dataset=False)
                    # organisms_genome.to_csv(os.path.join(
                    #     self._results_dir, self.ORGANISMS_GENOME +
                    #     str(self._organisms_genome_file_count) + '.csv'),
                    #                         index=False)
                    # self._organisms_genome_file_count += 1

                    organisms_loc_df = pd.DataFrame()
                    food_loc_df = pd.DataFrame()
                    eaten_stats = pd.DataFrame()
                    # organisms_stats = pd.DataFrame()
                    # organisms_genome = pd.DataFrame()

        organisms_loc = pa.Table.from_pandas(organisms_loc_df)
        food_loc = pa.Table.from_pandas(food_loc_df)
        eaten = pa.Table.from_pandas(eaten_stats)
        # organisms_stats_pa = pa.Table.from_pandas(organisms_stats)
        pq.write_to_dataset(table=organisms_loc,
                            root_path=os.path.join(self._results_dir,
                                                   self.ORGANISMS_LOCATIONS),
                            use_legacy_dataset=False)
        pq.write_to_dataset(table=food_loc,
                            root_path=os.path.join(self._results_dir,
                                                   self.FOOD_LOCATIONS),
                            use_legacy_dataset=False)
        pq.write_to_dataset(table=eaten,
                            root_path=os.path.join(self._results_dir,
                                                   self.EATEN_FOOD),
                            use_legacy_dataset=False)
        # pq.write_to_dataset(table=organisms_stats_pa,
        #                     root_path=os.path.join(self._results_dir,
        #                                            self.ORGANISMS_STATS),
        #                     use_legacy_dataset=False)

        # organisms_genome = pd.concat([
        #     organisms_genome,
        #     pull_organisms_genomes(self.env.organisms, iteration=i)
        # ])
        # organisms_genome.to_csv(os.path.join(
        #     self._results_dir, self.ORGANISMS_GENOME +
        #     str(self._organisms_genome_file_count) + '.csv'),
        #                         index=False)
        # self._organisms_genome_file_count += 1


def pull_organisms_stats(organisms: dict[int, Organism], iteration: int):
    df = pd.DataFrame(
        [], columns=['id', 'name', 'age', 'parents', 'iteration', 'energy'])
    for i, o in organisms.items():
        organism_data = {
            'id': [i],
            'name': [o.name],
            # 'genome': o.genome,
            'age': [o.age],
            # 'parent': o._parents,
            # 'genes_num': len(o.genome),
            'iteration': [iteration],
            'energy': [o.energy],
            'distance_traveled': [o.distance_traveled],
        }
        df = pd.concat([df, pd.DataFrame(organism_data)], ignore_index=True)
    return df


def pull_organisms_genomes(organisms: dict[int, Organism], iteration: int):
    df = pd.DataFrame(
        [], columns=['id', 'name', 'age', 'parents', 'iteration', 'energy'])
    for organism in organisms.values():
        d = {
            'id': [organism.id],
            'name': [organism.name],
            'genome_shape': [organism._size],
            # 'genome': [organism.genome],
            'age': [organism.age],
            'iteration': [iteration],
            'energy': [organism.energy],
            'parent': [[parent.id for parent in organism._parents]],
            'distance_traveled': [organism.distance_traveled],
        }
        for num, allele in enumerate(organism.genome):
            d[f'genome_{num}_weights'] = json.dumps(allele[0].tolist())
            d[f'genome_{num}_bias'] = json.dumps(allele[1].tolist())
        # organism_data = {
        #     'id': [i],
        #     'name': [o.name],
        #     'genome': [o.genome],
        #     'age': [o.age],
        #     'parent': [[parent.id for parent in o._parents]],
        #     # 'genes_num': len(o.genome),
        #     'iteration': [iteration],
        #     'energy': [o.energy],
        #     # 'distance_traveled'
        # }
        df = pd.concat([df, pd.DataFrame(d)], ignore_index=True)
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
            record['distance_traveled'] = r.distance_traveled
            record['genome_shape'] = r._size
            record['is_alive'] = r.is_alive
            record['parent'] = [parent.id for parent in r._parents]
            for num, allele in enumerate(r.genome):
                record[f'genome_{num}_weights'] = json.dumps(
                    allele[0].tolist())
                record[f'genome_{num}_bias'] = json.dumps(allele[1].tolist())
        records.append(record)

    new_df = pd.DataFrame(records)
    return pd.concat([df, new_df],
                     ignore_index=False) if concatenate else new_df
