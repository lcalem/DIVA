import argparse
import joblib
import numpy as np
import os
import torch
import sys

# print(sys.path)
# sys.path.pop(0)

from PIL import Image
from skimage import io

import matplotlib.pyplot as plt

from nuscenes import nuscenes as ns

from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper

from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer

from dataset.nuscenes_static_raster import StaticLayerRasterizerCustom, InputRepresentationCustom

INTER_TERMS = ['intersection', 'turn']      # intersection terms in the scene description
SCENE_BLACKLIST = [499, 515, 517]           # some scenes have bad localization (source: trajectron++)


def preprocess_nuscenes(data_root,
                        dataset_name='v1.0-mini',
                        past_history=6,
                        future_horizon=3,
                        output_dir=None,
                        complete='full',
                        reverse=False):
    '''
    from the original NuScenes interface (from nuscenes.nuscenes import NuScenes)
    to a format consumable by the Dataset

    data_root:      folder where the original nuscene data is
    dataset_name:   either 'v1.0-mini' or 'v1.0-trainval'
    past_history:   past window in seconds
    future_horizon: future window in seconds
    output_dir:     for putting the preprocessed dataset somewhere in particular. If left None, will create a subfolder 'preprocessed' in the data_root folder
    complete:       'full': only output sequences that have a full past of data
                    'fill': pad missing past data with zeros
    reverse:        whether

    Output:
    - in a subfolder of data_root/preprocessed, one csv (gt_trajectories.joblib) with each line containing:
        - past: list of [[x, y]] past trajectory
        - future: same for future trajectory
        - image_name: instance_sample.png
    - in the images subsubfolder, images for each sample
    '''

    assert dataset_name in ['v1.0-mini', 'v1.0-trainval']
    assert complete in ['full', 'fill']

    prefix = 'mini_' if 'mini' in dataset_name else ''

    nuscenes = ns.NuScenes(dataset_name, dataroot=data_root)
    helper = PredictHelper(nuscenes)

    # for images
    static_layer_rasterizer = StaticLayerRasterizerCustom(helper)
    agent_rasterizer = AgentBoxesWithFadedHistory(helper, seconds_of_history=past_history)
    input_representation = InputRepresentationCustom(static_layer_rasterizer, agent_rasterizer, Rasterizer())

    r = '_reverse' if reverse is True else ''
    output_dir = os.path.join(data_root, 'preprocessed', '%s_%s_%s_%s%s' % (dataset_name, past_history, future_horizon, complete, r))
    img_dir = os.path.join(output_dir, 'images')
    da_dir = os.path.join(output_dir, 'drivable_area_mask')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(da_dir, exist_ok=True)

    for split_name in ['train', 'val']:
        instances = set()
        samples = set()
        count_images = 0
        fatal_count = 0

        challenge_split = get_prediction_challenge_split("%s%s" % (prefix, split_name), dataroot=data_root)

        to_persist = list()
        count = 0
        count_weird = 0
        # iterate over all instance + samples couples
        for elt in challenge_split:
            if count % 1000 == 0:
                print(f'[{split_name}] done {count} ({count_weird})')
            count += 1

            instance_token, sample_token = elt.split('_')

            instances.add(instance_token)
            samples.add(sample_token)

            sample = nuscenes.get('sample', sample_token)
            scene = nuscenes.get('scene', sample['scene_token'])
            scene_id = int(scene['name'].replace('scene-', ''))
            if scene_id in SCENE_BLACKLIST:
                continue

            # check if sample is from an intersection
            description = scene['description'].lower()
            is_intersection = any([term in description for term in INTER_TERMS])

            # x,y trajectories for past and future
            past_xy = helper.get_past_for_agent(instance_token, sample_token, seconds=past_history, in_agent_frame=True)
            future_xy = helper.get_future_for_agent(instance_token, sample_token, seconds=future_horizon, in_agent_frame=True)

            # padding or exclusion (see 'complete' option)
            nb_ticks_past = past_history * 2
            if past_xy.shape != (nb_ticks_past, 2):

                if past_xy.shape[0] > nb_ticks_past:
                    print(f'shouldnt happen bug happened for i {instance_token}, s {sample_token}, shape was {past_xy}, past seconds {past_history}, skipping')
                    fatal_count += 1
                    continue

                if complete == 'fill':
                    # pad with zeros
                    to_pad = nb_ticks_past - past_xy.shape[0]
                    # assert to_pad > 0, f'something went wrong with padding i {instance_token} s {sample_token}, got shape {past_xy.shape}, {past_history}'
                    past_xy = np.pad(past_xy, ((to_pad, 0), (0, 0)), 'constant', constant_values=0)

                elif complete == 'full':
                    # just exclude non full sequences
                    continue

            nb_ticks_future = future_horizon * 2
            if future_xy.shape != (nb_ticks_future, 2):
                to_pad = nb_ticks_future - future_xy.shape[0]
                future_xy = np.pad(future_xy, ((0, to_pad), (0, 0)), 'constant', constant_values=0)

            # images
            img_name = '%s_%s.png' % (instance_token, sample_token)
            img_path = os.path.join(img_dir, img_name)

            da_name = 'da_%s_%s.png' % (instance_token, sample_token)
            da_path = os.path.join(da_dir, da_name)

            if (not os.path.exists(img_path)) or (not os.path.exists(da_path)):

                img, da_mask = input_representation.make_input_representation(instance_token, sample_token)

                # put the DA mask between 0 and 255
                if reverse:
                    da_mask = 1 - da_mask

                da_mask = da_mask * 255

                # whole BEV representation
                if not os.path.exists(img_path):
                    im = Image.fromarray(img)
                    im.save(img_path)
                    count_images += 1

                # drivable_area
                if not os.path.exists(da_path):
                    da_im = Image.fromarray(da_mask)
                    da_im.save(da_path)

            # save line
            assert isinstance(past_xy, np.ndarray)
            assert isinstance(future_xy, np.ndarray)
            assert isinstance(img_name, str)
            assert isinstance(is_intersection, bool)
            assert past_xy.shape == (nb_ticks_past, 2)
            assert future_xy.shape == (nb_ticks_future, 2)
            to_persist.append((past_xy, future_xy, img_name, is_intersection))

            count_weird += 1

        # writing
        traj_path = os.path.join(output_dir, 'gt_trajectories_%s.joblib' % split_name)
        print('writing GT trajectories in %s' % traj_path)

        with open(traj_path, 'wb+') as traj_f:
            joblib.dump(to_persist, traj_f)

        print('%s instances %s samples %s images' % (len(instances), len(samples), count_images))
        print('preprocessed dataset %s written in:' % split_name)
        print(output_dir)

        print('got %s fatal errors' % fatal_count)
        with open(os.path.join(output_dir, 'errors.txt'), 'w+') as err_f:
            err_f.write(str(fatal_count))

        print(f'total count {count}, weird {count_weird}')

    return output_dir


# check preprocessed data
def check_preprocessed(folder):
    '''
    - read csv to check number of lines
    - read first two lines to:
        - load and display the past and future trajectories
        - display the associated image

    for mini nuscenes:
    68 instances 205 samples 742 images

    for full dataset:
    1779 instances / 7197 samples / 16548 images
    (3529 instances 13898 samples 16548 images)

    for fill dataset:
    3529 instances / 13898 samples / 41205 images


    TODO: check coord type?
    '''
    sample_hz = 2       # how the dataset is sampled (2 img / s)

    count_data_line = 0

    split = folder.split('/')[-1].split('_')
    dataset_name = split[0]

    csv_path = os.path.join(folder, 'gt_trajectories_train.joblib')
    with open(csv_path, 'rb') as gt_f:

        all_file = joblib.load(gt_f)

        for line in all_file:

            count_data_line += 1

            past_np, future_np, img_name, is_intersection = line
            # print(past_np)
            # print(type(past_np))

            assert type(past_np) == np.ndarray
            assert past_np.shape == (past_h * sample_hz, 2), "expected %s, got %s" % (str(past_h * sample_hz), str(past_np.shape))

            assert type(future_np) == np.ndarray
            # commented for now because we keep incomplete futures as well
            # assert future_np.shape == (future_h * sample_hz, 2), "expected %s, got %s" % (str(future_h * sample_hz), str(future_np.shape))

            if count_data_line in [1, 2]:

                # trajectories
                print('Past trajectory')
                print(past_np)

                print('Future trajectory')
                print(future_np)

                # image
                img_path = os.path.join(folder, 'images', img_name)
                img = Image.open(img_path)
                plt.imshow(img)

    if dataset_name == 'v1.0-mini':
        # 274 is for when we take only complete past sequences (complete='full'), 742 is when we pad the past sequence with 0
        assert count_data_line in [275, 742], 'expected %s data lines, got %s' % ('742', str(count_data_line))
    else:
        assert count_data_line == 12876, 'expected %s data lines, got %s' % ('12876', str(count_data_line))   # for full


# TODO: proper argparse
# python3 preprocess_nuscenes.py -p /share/homes/lcalem/nuscenes -c fill
# python3 preprocess_nuscenes.py -p /share/homes/lcalem/nuscenes -c full -r
# python3 preprocess_nuscenes.py -p /share/homes/lcalem/nuscenes -c full -r -fs 6
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', '-p', required=True, help='actual path of the nuscenes dataset')
    parser.add_argument('--dataset', '-d', default='v1.0-trainval', help='v1.0-trainval | v1.0-mini')
    parser.add_argument('--complete', '-c', default='fill', help='fill: pad incomplete past sequences with 0 | full: discard examples with incomplete past')
    parser.add_argument('--reverse_da', '-r', default=False, action='store_true', help='False: DA mask is 1.0 in the drivable area and 0.0 outside | True: the other way around')
    parser.add_argument('--future_s', '-fs', type=int, default=3, help='Future seconds')

    args = parser.parse_args()

    output_dir = preprocess_nuscenes(args.dataroot, future_horizon=args.future_s, dataset_name=args.dataset, complete=args.complete, reverse=args.reverse_da)
    check_preprocessed(output_dir)
