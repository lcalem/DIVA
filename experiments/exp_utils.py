import datetime
import os
import shutil
import yaml

import numpy as np

from config.config import cfg
from models import log

from pprint import pprint


def exp_init(cmd, exps_folder=None, exp_name=''):
    '''
    common actions for setuping an experiment:
    - create experiment folder
    - dump config in it
    - create cmd file
    - dump current model code in it (because for now we only save weights)
    '''
    if exps_folder is None:
        exps_folder = os.path.join(os.environ.get('EXP_FOLDER', os.environ['HOME']), 'phy_experiments')

    # model folder
    model_folder = '%s/exp_%s_%s' % (exps_folder, datetime.datetime.now().strftime("%Y%m%d_%H%M"), exp_name)
    os.makedirs(model_folder)

    n_epochs = cfg.NB_EPOCHS
    log.printcn(log.OKBLUE, "Conducting experiment for %s epochs in folder %s" % (n_epochs, model_folder))

    # dump config
    config_path = os.path.join(model_folder, 'config.yaml')
    with open(config_path, 'w+') as f_conf:
        yaml.dump(cfg, f_conf, default_flow_style=False)

    # cmd
    cmd_path = os.path.join(model_folder, 'cmd.txt')
    with open(cmd_path, 'w+') as f_cmd:
        f_cmd.write(cmd + '\n')

    # model
    src_folder = os.path.dirname(os.path.realpath(__file__)) + '/..'
    dst_folder = os.path.join(model_folder, 'model_src/')

    shutil.copytree(src_folder, dst_folder, ignore=shutil.ignore_patterns('.*'))

    return model_folder


def find_best_checkpoint(exps_folder, load_weights_opts, mode='cvae'):
    '''
    load_weights_opts have 3 modes (by order of precedence):

    - path: the checkpoint path is given, so use that
    - epoch: if no path, find the checkpoint for specified epoch
    - metric: if no epoch, find the checkpoint that gives the best value for specified metric

    note: purposefully we don't check the validity/existence of the checkpoint, we let the weights loading fail downstream
    '''
    print('finding checkpoint with options:')
    pprint(load_weights_opts)

    # logline = ','.join([str(epoch), str(dpp_acc / i), str(loc_acc / i), str(end_time - t0), ADE_for_epoch, FDE_for_epoch, minADE_for_epoch, minFDE_for_epoch, rF_for_epoch, DAO_for_epoch, DAC_for_epoch, ASD_for_epoch, FSD_for_epoch, minASD_for_epoch, minFSD_for_epoch]) + '\n'
    # for CVAE metrics start at index 5, for DSF at 4
    metrics_indices = {
        'ade': 5,
        'fde': 6,
        'minade': 7,
        'minfde': 8,
        'rf': 9,
        'dao': 10,
        'dac': 11,
        'asd': 12,
        'fsd': 13,
        'minasd': 14,
        'minfsd': 15
    }

    if load_weights_opts['path'] is not None:
        return load_weights_opts['path']

    ckpt_by_epoch = dict()
    for filename in os.listdir(exps_folder):
        if filename.startswith('checkpoint_'):
            epoch = int(filename.strip().split('ep')[1])
            ckpt_by_epoch[epoch] = os.path.join(exps_folder, filename)

    epoch_keys = list(ckpt_by_epoch.keys())

    # epoch given, find the checkpoint for the closest epoch (rounding: floor)
    if load_weights_opts['epoch'] is not None:
        epoch_distances = [abs(int(load_weights_opts['epoch']) - e) for e in epoch_keys]
        closest_index = np.argmin(epoch_distances)
        return ckpt_by_epoch[epoch_keys[closest_index]]

    # metric given, find the epoch where the metric is best
    elif load_weights_opts['metric'] is not None:
        metric = load_weights_opts['metric'].lower()
        assert metric in metrics_indices, f'unknown metric criterion {metric}'
        metric_index = metrics_indices[metric]
        if mode == 'dsf':
            metric_index -= 1    # i know it's horrible but the loglines are not the same for cvae and dsf models

        print(f'[ckpt] metric {metric} should be at index {metric_index}')

        logsuffix = '_dsf' if mode == 'dsf' else ''
        results_path = os.path.join(exps_folder, 'train%s_logs.csv' % logsuffix)
        metric_by_epoch = dict()
        with open(results_path, 'r') as f_logs:
            for line in f_logs:
                logline = line.strip().split(',')
                metric_val = logline[metric_index]
                if metric_val == 'N/A':
                    continue
                metric_by_epoch[int(logline[0])] = float(metric_val)

        # ADE and FDE: min // other: max
        if metric in ['ade', 'fde', 'minade', 'minfde']:
            best_epoch = min(metric_by_epoch, key=metric_by_epoch.get)
        else:
            best_epoch = max(metric_by_epoch, key=metric_by_epoch.get)
        print(f'[ckpt] found epoch {best_epoch} with metric {metric}={metric_by_epoch[best_epoch]}')
        return ckpt_by_epoch[best_epoch]

    # otherwise just give off the latest epoch
    else:
        log.printcn(log.WARNING, 'No directive was passed to load weights, loading from last epoch')
        return ckpt_by_epoch[max(epoch_keys)]
