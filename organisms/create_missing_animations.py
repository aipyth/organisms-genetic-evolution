import plot
import glob
import pyarrow.parquet as pq
import os
import json

RESULTS_DIR = 'results/result_*'
FRAMES_DIR_NAME = 'frames/'
ORGANISMS_LOCATIONS_DIR_NAME = 'organisms_locations'
FOOD_LOCATIONS_DIR_NAME = 'food_locations'
VIDEO_FILE_NAME = 'evolution_in_the_environment_video.mp4'
METADATA_FILE_NAME = 'metadata.json'

for result_dir in glob.glob(RESULTS_DIR):
    print(f'Checking {result_dir}')
    org_loc_path = os.path.join(result_dir, ORGANISMS_LOCATIONS_DIR_NAME)
    food_loc_path = os.path.join(result_dir, FOOD_LOCATIONS_DIR_NAME)
    frames_dir_path = os.path.join(result_dir, FRAMES_DIR_NAME)
    video_path = os.path.join(result_dir, VIDEO_FILE_NAME)

    if os.path.exists(frames_dir_path):
        print(f'Skipping due to existing frames directory: {result_dir}')
        continue
    if os.path.exists(video_path):
        print(f'Skipping due to existing video file: {result_dir}')
        continue

    org_loc = pq.read_table(org_loc_path).to_pandas()
    food_loc = pq.read_table(food_loc_path).to_pandas()

    org_loc.reset_index(drop=True, inplace=True)
    food_loc.reset_index(drop=True, inplace=True)

    with open(os.path.join(result_dir, METADATA_FILE_NAME),
              mode='r',
              encoding='utf-8') as fp:
        metadata = json.load(fp)
    environment_width = metadata['width']
    environment_height = metadata['height']
    organism_size = metadata['organism_size']
    food_size = metadata['food_size']

    os.mkdir(frames_dir_path)

    print(f'Creating frames for {result_dir}')
    plot.create_frames(org_loc, food_loc, environment_width,
                       environment_height, organism_size, food_size,
                       frames_dir_path)

    if not os.path.exists(frames_dir_path):
        print(
            f'Skipping video generation due to unexisting frames directory: {result_dir}'
        )
        continue
    print('Generating video...')
    plot.generate_video(
        frames_dir_path,
        framerate=24,
        output=os.path.join(result_dir, 'evolution.mp4'),
    )
