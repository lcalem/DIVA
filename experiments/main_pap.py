import argparse
import os
import shutil
import sys
import time

from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

import properscoring as ps

from config import config_utils

from dataset.nuscenes_dataset import NuScenesDataset
from nuscenes.eval.prediction import metrics

from experiments import exp_utils

from models import diversity_metrics
from models import log
from models import model_utils as mutils
from models.losses import loss_kullback_leibler, diversity_loss, layout_loss, old_layout_loss, reconstruction_loss
from models.cvae import cVAE
from models.cvae_loc import cVAE_loc
from models.cvae_conv import cVAE_conv
from models.dsf import DSF, DSF_loc, DSF_layout

from config.config import cfg


class Launcher():

    def __init__(self, exp_folder, debug=False, init_weights_opts=None):

        self.exp_folder = exp_folder
        self.data_dir = cfg.DATASET.PATH
        self.init_weights_opts = init_weights_opts

        self.debug = debug

        if not torch.cuda.is_available():
            raise Exception('GPU not found')

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_datasets(self, include_da_in_train=False):
        # dataset
        dataset_train = NuScenesDataset(self.data_dir, split='train', include_da=include_da_in_train, include_diff=include_da_in_train)
        self.dataset_train = DataLoader(dataset_train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
        print('len dataset train %s' % len(self.dataset_train))

        dataset_test = NuScenesDataset(self.data_dir, split='val', include_da=True, include_diff=False)
        self.dataset_test = DataLoader(dataset_test, batch_size=cfg.TEST_BATCH_SIZE, shuffle=False, num_workers=0)

    def train_cvae(self, archi):
        '''
        training results are logged in the 'train_logs.csv' file which contains one line per epoch with the following columns:
        - epoch number
        - total loss
        - reconstruction loss
        - KL loss
        - execution time (in seconds)
        - for epochs where an eval occurs: minADE value
        '''

        # dataset
        self.load_datasets()

        # model
        model_options = {
            'input_size': cfg.MODEL.INPUT_SIZE,
            'rnn_units': cfg.MODEL.RNN_UNITS,
            'nlayers': cfg.MODEL.NLAYERS,
            'bidirectional': cfg.MODEL.BIDIRECTIONAL,
            'batch_size': cfg.BATCH_SIZE,
            'latent_dim': cfg.MODEL.LATENT_DIM,
            'local_dim': cfg.MODEL.LOCAL_DIM,
            'fc_units': cfg.MODEL.FC_UNITS,
            'target_length': cfg.MODEL.N_OUTPUT,
            'pretrained_enc': cfg.MODEL.ENCODER.PRETRAINED,
            'device': self.device
        }
        self.build_genmodel(archi, model_options)

        logs_file = os.path.join(self.exp_folder, 'train_logs.csv')

        # train
        optimizer = torch.optim.Adam(self.genmodel.parameters(), lr=cfg.TRAINING.LR)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)
        criterion = torch.nn.MSELoss()
        nsamples = cfg.NSAMPLES    # number of future trajectories to generate for each past
        gamma = cfg.GAMMA   # TODO

        best = {
            'best_ADE': float('inf'),
            'best_FDE': float('inf'),
            'best_minADE': float('inf'),
            'best_minFDE': float('inf'),
            'best_rF': 0.0,
            'best_DAO': 0.0,
            'best_DAC': 0.0,
            'best_ASD': 0.0,
            'best_FSD': 0.0,
            'best_minASD': 0.0,
            'best_minFSD': 0.0
        }

        with open(logs_file, 'w+') as f_log:

            for epoch in range(cfg.NB_EPOCHS):
                self.genmodel.train()
                t0 = time.time()

                # TODO: trouver ce qu'enumerate appelle
                for i, data in enumerate(self.dataset_train, 0):
                    inputs = data['past_xy']
                    targets = data['future_xy']
                    layouts = data['image']

                    rec_acc = 0.0
                    kl_acc = 0.0
                    loss_acc = 0.0

                    # print('input shape %s' % str(inputs.shape))
                    # print('target shape %s' % str(targets.shape))
                    # print(layouts.shape)

                    inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
                    targets = torch.tensor(targets, dtype=torch.float32).to(self.device)
                    layouts = layouts.to(self.device)   # should already be a tensor because of applied transformation
                    batch_size, N_output = targets.shape[0:2]

                    outputs, z_mu, z_logvar = self.genmodel(inputs, targets, layouts)           # outputs [batch, seq_len, nfeatures]
                    loss_reconstruction, loss_kl = 0, 0

                    loss_mse = criterion(targets, outputs)       # can be dilate loss later
                    loss_reconstruction = loss_mse
                    loss_kl = cfg.TRAINING.CVAE.BETA * loss_kullback_leibler(z_mu, z_logvar)         # Kullback-Leibler (we put the beta here to better track the loss_kl value in logs and graphs)

                    loss = loss_reconstruction + loss_kl

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    rec_acc += loss_reconstruction.item()
                    kl_acc += loss_kl.item()
                    loss_acc += loss.item()

                end_time = time.time()
                print(f'ep {epoch}, loss {loss_acc}, rec={rec_acc}, KL={kl_acc}, time {end_time - t0}')

                metrics = self.eval_and_print(epoch, self.genmodel, nsamples, best, mode='cvae')
                minADE = metrics[2]

                if minADE != 'N/A':
                    scheduler.step(minADE)     # use step(dilate) if using dilate loss?

                logline = ','.join([str(epoch), str(loss_acc / i), str(rec_acc / i), str(kl_acc / i), str(end_time - t0)] + metrics) + '\n'
                print(logline)
                f_log.write(logline)

    def train_dsf(self, cvae_folder):
        # dataset
        self.load_datasets(include_da_in_train=True)

        # dsf
        self.load_cvae(cvae_folder, self.init_weights_opts)
        if cfg.DSF.ARCHI == 'past_only':
            self.dsf = DSF(self.genmodel, cfg.NSAMPLES, cfg.MODEL.N_OUTPUT)
            params = self.dsf.MLP.parameters()
        elif cfg.DSF.ARCHI == 'layout_only':
            self.dsf = DSF_layout(self.genmodel, cfg.NSAMPLES, cfg.MODEL.N_OUTPUT)   # Layout DSF
            params = self.dsf.MLP_loc.parameters()
        elif cfg.DSF.ARCHI == 'both':
            self.dsf = DSF_loc(self.genmodel, cfg.NSAMPLES, cfg.MODEL.N_OUTPUT, cfg.DSF.FUSION)
            params = list(self.dsf.MLP.parameters()) + list(self.dsf.MLP_loc.parameters())
        else:
            raise Exception('Unsupported DSF architecture (choose from past_only, layout_only or both)')

        self.dsf.to(self.device)
        print(self.dsf)

        logs_file = os.path.join(self.exp_folder, 'train_dsf_logs.csv')

        # train (we only pass the DSF parameters to the optimizer so that the original CVAE is not updated)
        if cfg.DSF.FINETUNE_CVAE is True:
            log.printcn(log.OKGREEN, '[DSF training] finetuning cvae')
            params += list(self.genmodel.parameters())
        optimizer = torch.optim.Adam(params, lr=cfg.DSF.TRAINING_LR)

        # loss balancing
        lambda_dpp = 1.0
        lambda_loc = 1.0
        if cfg.DSF.LAMBDA is not None:
            assert 0.0 <= cfg.DSF.LAMBDA <= 1.0
            lambda_dpp = cfg.DSF.LAMBDA
            lambda_loc = 1.0 - lambda_dpp

        best = {
            'best_ADE': float('inf'),
            'best_FDE': float('inf'),
            'best_minADE': float('inf'),
            'best_minFDE': float('inf'),
            'best_rF': 0.0,
            'best_DAO': 0.0,
            'best_DAC': 0.0,
            'best_ASD': 0.0,
            'best_FSD': 0.0,
            'best_minASD': 0.0,
            'best_minFSD': 0.0
        }

        with open(logs_file, 'w+') as f_log:

            for epoch in range(cfg.DSF.NB_EPOCHS):
                self.dsf.train()
                t0 = time.time()

                dpp_acc = 0.0
                loc_acc = 0.0
                layout_loss_time = 0.0

                for i, data in enumerate(self.dataset_train, 0):

                    inputs = data['past_xy']
                    targets = data['future_xy']
                    layouts = data['image']
                    da_mask = data['da_mask']
                    d_mask = data['d_mask']
                    dx_mask = data['dx_mask']
                    dy_mask = data['dy_mask']

                    inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
                    targets = torch.tensor(targets, dtype=torch.float32).to(self.device)
                    layouts = layouts.to(self.device)   # should already be a tensor because of applied transformation
                    batch_size, N_output = targets.shape[0:2]

                    d_mask = torch.tensor(d_mask, dtype=torch.float32).to(self.device)
                    dx_mask = torch.tensor(dx_mask, dtype=torch.float32).to(self.device)
                    dy_mask = torch.tensor(dy_mask, dtype=torch.float32).to(self.device)

                    dsf_outputs, _ = self.dsf(inputs, layouts) # outputs [batch, nsamples, seq_len, nfeatures]
                    # dsf_outputs = torch.tensor(dsf_outputs, requires_grad=True).to(device)

                    loss = torch.tensor(0.0)
                    dpp_loss = torch.tensor(0.0)
                    dpp_loss = Variable(dpp_loss, requires_grad=True)

                    # DPP loss
                    if cfg.DSF.USE_DPP_LOSS:
                        dpp_loss = diversity_loss(dsf_outputs, targets, cfg.DSF.KERNEL, self.device)
                        dpp_acc += dpp_loss.item()

                        if cfg.DSF.OFFSET is True:
                            dpp_loss = 10 + dpp_loss

                    # print("dpp loss")
                    # print(dpp_loss)

                    # Layout loss
                    if cfg.DSF.USE_LAYOUT_LOSS:
                        # print(f'dsf_outputs shape {dsf_outputs.shape}')
                        t0loss = time.time()

                        # loc_loss = old_layout_loss(dsf_outputs, da_mask, self.device)
                        loc_loss = layout_loss(dsf_outputs, d_mask, dx_mask, dy_mask, self.device)

                        # metrics
                        layout_loss_time += time.time() - t0loss
                        loc_acc += loc_loss.item()

                    # combining the losses
                    if cfg.DSF.USE_LAYOUT_LOSS and cfg.DSF.USE_DPP_LOSS:
                        loss = lambda_dpp * dpp_loss + lambda_loc * loc_loss
                    elif cfg.DSF.USE_LAYOUT_LOSS:
                        loss = loc_loss
                    else:
                        loss = dpp_loss

                    # print("total loss")
                    # print(loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                end_time = time.time()
                print(f'ep {epoch}, dpp_loss {dpp_acc / i}, loc_loss {loc_acc / i}, time {end_time - t0}, loc_loss_time {layout_loss_time}')

                metrics = self.eval_and_print(epoch, self.dsf, cfg.NSAMPLES, best, mode='dsf')
                logline = ','.join([str(epoch), str(dpp_acc / i), str(loc_acc / i), str(end_time - t0)] + metrics) + '\n'
                print(logline)
                f_log.write(logline)

    def train_jointly(self, cvae_archi, k, rec_select):
        '''
        jointly train the cVAE and the DSF

        k: out of NSAMPLES predicted trajectories, in how many is the reconstruction loss backpropped
        rec_select: how do we select those k trajectories (random | closest [to GT])
        '''
        self.load_datasets(include_da_in_train=True)
        writer = SummaryWriter(self.exp_folder)

        nsamples = cfg.NSAMPLES    # number of future trajectories to generate for each past
        assert k <= nsamples       # we can't backprop the reconstruction loss in more than the total number of trajectories
        log.printcn(log.OKBLUE, f'Reconstruction loss for {k} trajectories out of {nsamples}')

        # model
        model_options = {
            'input_size': cfg.MODEL.INPUT_SIZE,
            'rnn_units': cfg.MODEL.RNN_UNITS,
            'nlayers': cfg.MODEL.NLAYERS,
            'bidirectional': cfg.MODEL.BIDIRECTIONAL,
            'batch_size': cfg.BATCH_SIZE,
            'latent_dim': cfg.MODEL.LATENT_DIM,
            'local_dim': cfg.MODEL.LOCAL_DIM,
            'fc_units': cfg.MODEL.FC_UNITS,
            'target_length': cfg.MODEL.N_OUTPUT,
            'pretrained_enc': cfg.MODEL.ENCODER.PRETRAINED,
            'device': self.device
        }
        logs_file = os.path.join(self.exp_folder, 'train_logs.csv')

        # generative model
        self.build_genmodel(cvae_archi, model_options)
        gen_params = list(self.genmodel.parameters())

        # DSF
        self.dsf = DSF_loc(self.genmodel, cfg.NSAMPLES, cfg.MODEL.N_OUTPUT, cfg.DSF.FUSION)
        dsf_params = list(self.dsf.MLP.parameters()) + list(self.dsf.MLP_loc.parameters())
        self.dsf.to(self.device)
        print(self.dsf)

        params = gen_params + dsf_params
        optimizer = torch.optim.Adam(params, lr=cfg.TRAINING.JOINT.LR)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)
        criterion = torch.nn.MSELoss()

        best = {
            'best_ADE': float('inf'),
            'best_FDE': float('inf'),
            'best_minADE': float('inf'),
            'best_minFDE': float('inf'),
            'best_rF': 0.0,
            'best_DAO': 0.0,
            'best_DAC': 0.0,
            'best_ASD': 0.0,
            'best_FSD': 0.0,
            'best_minASD': 0.0,
            'best_minFSD': 0.0
        }

        with open(logs_file, 'w+') as f_log:

            for epoch in range(cfg.NB_EPOCHS):
                self.genmodel.train()
                self.dsf.train()

                t0 = time.time()

                rec_acc = 0.0
                kl_acc = 0.0
                loss_acc = 0.0
                dpp_acc = 0.0
                loc_acc = 0.0
                layout_loss_time = 0.0
                k_dist = list()

                for i, data in enumerate(self.dataset_train, 0):
                    inputs = data['past_xy']
                    targets_np = data['future_xy']
                    layouts = data['image']
                    da_mask = data['da_mask']
                    d_mask = data['d_mask']
                    dx_mask = data['dx_mask']
                    dy_mask = data['dy_mask']

                    # print('input shape %s' % str(inputs.shape))
                    # print('target shape %s' % str(targets.shape))
                    # print(layouts.shape)

                    inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
                    targets = torch.tensor(targets_np, dtype=torch.float32).to(self.device)
                    layouts = layouts.to(self.device)   # should already be a tensor because of applied transformation
                    batch_size, h_future, n_dim = targets.shape

                    d_mask = torch.tensor(d_mask, dtype=torch.float32).to(self.device)
                    dx_mask = torch.tensor(dx_mask, dtype=torch.float32).to(self.device)
                    dy_mask = torch.tensor(dy_mask, dtype=torch.float32).to(self.device)

                    dsf_outputs, _ = self.dsf(inputs, layouts) # outputs [batch, nsamples, seq_len, nfeatures]

                    # losses
                    loss = torch.tensor(0.0)
                    dpp_loss = torch.tensor(0.0)
                    dpp_loss = Variable(dpp_loss, requires_grad=True)

                    # TODO: if we use DPP loss
                    if True:
                        dpp_loss = diversity_loss(dsf_outputs, targets, cfg.DSF.KERNEL, self.device)
                        dpp_acc += dpp_loss.item()

                    # TODO: if we use layout loss
                    if True:
                        t0loss = time.time()
                        loc_loss = layout_loss(dsf_outputs, d_mask, dx_mask, dy_mask, self.device)
                        layout_loss_time += time.time() - t0loss
                        loc_acc += loc_loss.item()

                    # TODO if we use reconstruction loss
                    if True:
                        # modular way
                        rec_loss, k_dist_step = reconstruction_loss(dsf_outputs, targets_np, k, self.device, rec_select)
                        k_dist.extend(k_dist_step)

                        # normal way
                        # stacked_targets = np.repeat(targets_np, cfg.NSAMPLES, axis=0).to(self.device)
                        # stacked_outputs = dsf_outputs.view(-1, n_future, n_dim).to(self.device)
                        # rec_loss = criterion(stacked_targets, stacked_outputs)

                        rec_acc += rec_loss.item()

                    # TODO: adapt this line when the ifs are not True anymore
                    loss = cfg.TRAINING.JOINT.LAMBDA_REC * rec_loss + cfg.TRAINING.JOINT.LAMBDA_DPP * dpp_loss + cfg.TRAINING.JOINT.LAMBDA_LOC * loc_loss
                    loss_acc += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # log losses for every batch for epoch 1
                    # TODO: add checks if we don't use all the losses
                    if epoch == 0:
                        writer.add_scalar('firstepoch/dpp', dpp_loss, i)
                        writer.add_scalar('firstepoch/loc', loc_loss, i)
                        writer.add_scalar('firstepoch/rec', rec_loss, i)
                        writer.add_histogram('firstepoch/k', k_dist_step, i, bins=12)

                # logging
                writer.add_scalar('loss/dpp', dpp_acc / i, epoch)
                writer.add_scalar('loss/loc', loc_acc / i, epoch)
                writer.add_scalar('loss/rec', rec_acc / i, epoch)
                writer.add_histogram('k', k_dist, epoch, bins=12)

                end_time = time.time()
                print(f'ep {epoch}, dpp_loss {dpp_acc / i}, loc_loss {loc_acc / i}, rec_loss {rec_acc / i}, time {end_time - t0}, loc_loss_time {layout_loss_time}')

                metrics = self.eval_and_print(epoch, self.dsf, cfg.NSAMPLES, best, mode='dsf', writer=writer)
                logline = ','.join([str(epoch), str(dpp_acc / i), str(loc_acc / i), str(rec_acc / i), str(end_time - t0)] + metrics) + '\n'
                print(logline)
                f_log.write(logline)

        writer.close()

    def eval_and_print(self, epoch, model, nsamples, best, mode, writer=None):
        '''
        wrapper to avoid repeating logging between eval for cVAE and DSF
        '''

        ADE_for_epoch, FDE_for_epoch, minADE_for_epoch, minFDE_for_epoch, rF_for_epoch, DAO_for_epoch, DAC_for_epoch, ASD_for_epoch, FSD_for_epoch, minASD_for_epoch, minFSD_for_epoch = ['N/A'] * 11

        if (epoch % cfg.EVAL_EVERY == 0) or (epoch == cfg.NB_EPOCHS - 1):
            model.eval()
            ADE, FDE, minADE, minFDE, ASD, FSD, minASD, minFSD, rF, DAO, DAC = self.eval_model(model, nsamples, mode=mode)

            if ADE < best['best_ADE']:
                best['best_ADE'] = ADE
                print(f'---------------- new best ADE (ep {epoch}): {ADE}')

            if FDE < best['best_FDE']:
                best['best_FDE'] = FDE
                print(f'---------------- new best FDE (ep {epoch}): {FDE}')

            if minADE < best['best_minADE']:
                best['best_minADE'] = minADE
                print(f'---------------- new best minADE (ep {epoch}): {minADE}')

            if minFDE < best['best_minFDE']:
                best['best_minFDE'] = minFDE
                print(f'---------------- new best minFDE (ep {epoch}): {minFDE}')

            # diversity metrics
            if rF > best['best_rF']:
                best['best_rF'] = rF
                print(f'---------------- new best rF (ep {epoch}): {rF}')

            if DAO > best['best_DAO']:
                best['best_DAO'] = DAO
                print(f'---------------- new best DAO (ep {epoch}): {DAO}')

            if DAC > best['best_DAC']:
                best['best_DAC'] = DAC
                print(f'---------------- new best DAC (ep {epoch}): {DAC}')

            if ASD > best['best_ASD']:
                best['best_ASD'] = ASD
                print(f'---------------- new best ASD (ep {epoch}): {ASD}')

            if FSD > best['best_FSD']:
                best['best_FSD'] = FSD
                print(f'---------------- new best FSD (ep {epoch}): {FSD}')

            # save the model/dsf for each evaluated epoch
            ckpt_suffix = '_dsf' if mode == 'dsf' else ''
            save_name = os.path.join(self.exp_folder, 'checkpoint%s_ep%s' % (ckpt_suffix, str(epoch)))
            torch.save(model.state_dict(), save_name)

            if writer is not None:
                writer.add_scalar('metrics/ADE', ADE, epoch)
                writer.add_scalar('metrics/FDE', FDE, epoch)
                writer.add_scalar('metrics/minADE', minADE, epoch)
                writer.add_scalar('metrics/minFDE', minFDE, epoch)
                writer.add_scalar('metrics/rF', rF, epoch)
                writer.add_scalar('metrics/DAO', DAO, epoch)
                writer.add_scalar('metrics/DAC', DAC, epoch)
                writer.add_scalar('metrics/ASD', ASD, epoch)
                writer.add_scalar('metrics/FSD', FSD, epoch)
                writer.add_scalar('metrics/minASD', minASD, epoch)
                writer.add_scalar('metrics/minFSD', minFSD, epoch)

            ADE_for_epoch = str(ADE)
            FDE_for_epoch = str(FDE)
            minADE_for_epoch = str(minADE)
            minFDE_for_epoch = str(minFDE)
            rF_for_epoch = str(rF)
            DAO_for_epoch = str(DAO)
            DAC_for_epoch = str(DAC)
            ASD_for_epoch = str(ASD)
            FSD_for_epoch = str(FSD)
            minASD_for_epoch = str(minASD)
            minFSD_for_epoch = str(minFSD)

        metrics = [ADE_for_epoch, FDE_for_epoch, minADE_for_epoch, minFDE_for_epoch, rF_for_epoch, DAO_for_epoch, DAC_for_epoch, ASD_for_epoch, FSD_for_epoch, minASD_for_epoch, minFSD_for_epoch]
        return metrics

    def eval_model(self, model, nsamples, mode):
        '''
        eval computes minADE over the nsamples provided
        nsamples: number of trajectories we sample

        /!\ ASSUMING TEST DATA COMES IN BATCHES OF 1

        Eval metrics:
        - minADE (min Average Displacement Error)
        - minFDE (min Final Displacement Error)
        - rF (ratio of avgFDE and minFDE)
        - DAC (Drivable Area Occupancy, see Park et al ECCV 2020)
        - DAC (Drivable Area Count, see Chang et al Argoverse CVPR 2019)
        '''

        all_ADEs = list()
        all_FDEs = list()
        all_mADEs = list()
        all_mFDEs = list()
        all_rFs = list()
        all_DAOs = list()
        all_DACs = list()
        all_ASDs = list()
        all_FSDs = list()
        all_mASDs = list()
        all_mFSDs = list()

        eval_t0 = time.time()

        for i, data in enumerate(self.dataset_test):
            inputs = data['past_xy']
            targets = data['future_xy']
            layouts = data['image']

            da_mask = data['da_mask']

            inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)  # [bs, seq_len, nfeatures]
            # targets = torch.tensor(targets, dtype=torch.float32).to(self.device)  # [bs, seq_len, nfeatures]
            layouts = layouts.to(self.device)

            if (mode == 'dsf'):    # sampling everything at the same time
                outputs, _ = model(inputs, layouts) # outputs [nfutures, nsamples, seq_len, nfeatures]
                outputs = outputs[0,:,:,:].detach().cpu().numpy()  # we can do that because evaluation is batch size 1

            elif (mode == 'cvae'): # mode cVAE, sample several predictions
                outputs = np.empty((nsamples, targets.shape[1], targets.shape[2]))
                for k in range(nsamples):
                    output_k = model.sample(inputs, layouts).detach().cpu().numpy()
                    # print('output_k shape %s' % str(output_k.shape))
                    outputs[k] = output_k

            stacked_targets = np.repeat(targets, nsamples, axis=0)
            probs = np.array([[1 / nsamples] * nsamples])

            if self.debug:
                print('shape outputs min ADE %s' % str(outputs.shape))              # (12, 6, 2)
                print('shape targets min ADE %s' % str(stacked_targets.shape))      # torch.Size([12, 6, 2])
                print('shape probs min ADE %s' % str(probs.shape))                  # (1, 12)
                print('da_mask of shape %s' % str(da_mask.shape))                   # torch.Size([1, 1, 224, 224])

            ADEs = metrics.mean_distances(outputs, stacked_targets.detach().cpu().numpy())
            FDEs = metrics.final_distances(outputs, stacked_targets.detach().cpu().numpy())

            ADE = np.mean(ADEs)
            FDE = np.mean(FDEs)
            minADE = np.min(ADEs)
            minFDE = np.min(FDEs)

            # ASD and FSD (self distances)
            stacked_outputs_1, stacked_outputs_2 = stack_for_comparison(outputs)
            mdist = metrics.mean_distances(stacked_outputs_1, stacked_outputs_2)
            fdist = metrics.final_distances(stacked_outputs_1, stacked_outputs_2)
            minASD = np.min(mdist)
            minFSD = np.min(fdist)
            ASD = np.mean(mdist)
            FSD = np.mean(fdist)

            # diversity metrics
            rF = np.mean(FDEs) / minFDE

            DAO = diversity_metrics.dao(outputs, da_mask[0][0])   # [0][0] because we only need the (224, 224) binary mask from the (1, 1, 224, 224) dataloader shape
            DAC = diversity_metrics.dac(outputs, da_mask[0][0])

            all_ADEs.append(ADE)
            all_FDEs.append(FDE)
            all_mADEs.append(minADE)
            all_mFDEs.append(minFDE)
            all_ASDs.append(ASD)
            all_FSDs.append(FSD)
            all_mASDs.append(minASD)
            all_mFSDs.append(minFSD)
            all_rFs.append(rF)
            all_DAOs.append(DAO)
            all_DACs.append(DAC)

        mean_ADE = np.array(all_ADEs).mean()
        mean_FDE = np.array(all_FDEs).mean()
        mean_minADE = np.array(all_mADEs).mean()
        mean_minFDE = np.array(all_mFDEs).mean()
        mean_ASD = np.array(all_ASDs).mean()
        mean_FSD = np.array(all_FSDs).mean()
        mean_minASD = np.array(all_mASDs).mean()
        mean_minFSD = np.array(all_mFSDs).mean()
        mean_rF = np.array(all_rFs).mean()
        mean_DAO = np.array(all_DAOs).mean()
        mean_DAC = np.array(all_DACs).mean()

        eval_t1 = time.time()

        print(f'--- Eval ADE= {mean_ADE}')
        print(f'--- Eval FDE= {mean_FDE}')
        print(f'--- Eval min ADE= {mean_minADE}')
        print(f'--- Eval min FDE= {mean_minFDE}')
        print(f'--- Eval ASD= {mean_ASD}')
        print(f'--- Eval FSD= {mean_FSD}')
        print(f'--- Eval min ASD= {mean_minASD}')
        print(f'--- Eval min FSD= {mean_minFSD}')
        print(f'--- Eval rF= {mean_rF}')
        print(f'--- Eval DAO= {mean_DAO}')
        print(f'--- Eval DAC= {mean_DAC}')

        print(f'Done eval in {eval_t1 - eval_t0}')

        return mean_ADE, mean_FDE, mean_minADE, mean_minFDE, mean_ASD, mean_FSD, mean_minASD, mean_minFSD, mean_rF, mean_DAO, mean_DAC

    def build_genmodel(self, archi, options_dict):
        '''
        Underlying generative model
        '''
        if archi == 'cvae_loc':
            self.genmodel = cVAE_loc(**options_dict)

        elif archi == 'cvae':
            # TODO: probably will break because of local_dim option (maybe add **kwargs like a douche)
            self.genmodel = cVAE(**options_dict)

        elif archi == 'cvae_conv':
            self.genmodel = cVAE_conv(**options_dict)

        elif archi == 'cvae_cut':
            self.genmodel = cVAE_cut(**options_dict)

        else:
            raise Exception('Unsupported archi name %s. (use cvae, cvae_loc or cvae_conv)' % archi)

        self.genmodel.to(self.device)

        # freeze stem parameters except the FC
        if cfg.MODEL.ENCODER.FREEZE is True:
            for name, param in self.genmodel.named_parameters():
                if name.startswith('encoder.local_convnet.') and not name.startswith('encoder.local_convnet.fc.'):
                    param.requires_grad = False

        print('trainable convnet parameters: %s' % mutils.count_params(self.genmodel.encoder.local_convnet))
        print('trainable parameters: %s' % mutils.count_params(self.genmodel))
        print(self.genmodel)

    def load_cvae(self, cvae_folder, init_weights_opts):
        '''
        - loads options from the original CVAE training
        - builds the CVAE with the original options
        - finds the best weights checkpoint
        - loads the weights
        '''

        # options
        cvae_options_path = os.path.join(cvae_folder, 'config.yaml')
        cvae_config = config_utils.parse_options_file(cvae_options_path)

        cvae_options = {
            'input_size': cvae_config['MODEL']['INPUT_SIZE'],
            'rnn_units': cvae_config['MODEL']['RNN_UNITS'],
            'nlayers': cvae_config['MODEL']['NLAYERS'],
            'bidirectional': cvae_config['MODEL']['BIDIRECTIONAL'],
            'batch_size': cfg.BATCH_SIZE,
            'latent_dim': cvae_config['MODEL']['LATENT_DIM'],
            'local_dim': cvae_config['MODEL']['LOCAL_DIM'],
            'fc_units': cvae_config['MODEL']['FC_UNITS'],
            'target_length': cvae_config['MODEL']['N_OUTPUT'],
            'pretrained_enc': cvae_config['MODEL']['ENCODER']['PRETRAINED'],
            'device': self.device
        }

        # archi
        cmd_file = os.path.join(cvae_folder, 'cmd.txt')
        with open(cmd_file, 'r') as f_cmd:
            cmd = f_cmd.read()
        cvae_archi = cmd.split('-a ')[1].split(' ')[0]
        print(f'loading CVAE with archi {cvae_archi}')

        # CVAE
        self.build_genmodel(cvae_archi, cvae_options)

        ckpt = exp_utils.find_best_checkpoint(cvae_folder, init_weights_opts)
        log.printcn(log.OKBLUE, f'Loading CVAE checkpoint {ckpt}')
        self.genmodel.load_state_dict(torch.load(ckpt))

    def load_dsf(self, dsf_folder, dsf_init_opts):

        self.dsf = DSF_loc(self.genmodel, cfg.NSAMPLES, cfg.MODEL.N_OUTPUT, cfg.DSF.FUSION)

        self.dsf.to(self.device)
        print(self.dsf)

        ckpt = exp_utils.find_best_checkpoint(dsf_folder, dsf_init_opts)
        log.printcn(log.OKBLUE, f'Loading CVAE checkpoint {ckpt}')
        self.genmodel.load_state_dict(torch.load(ckpt))

    def eval_dsf(self, dsf_folder, dsf_init_opts):

        # load cVAE
        dsf_cmd_file = os.path.join(dsf_folder, 'cmd.txt')
        with open(dsf_cmd_file, 'r') as f_cmd:
            dsf_cmd = f_cmd.read()
        cvae_folder = dsf_cmd.split('-cf ')[1].split(' ')[0].strip()   # TODO long versions
        # -wm minADE -cf /share/homes/lcalem/phy_experiments/exp_20210419_1216_baseline_pf_full
        cvae_init_opts = {
            'path': dsf_cmd.split('-wp ')[1].split(' ')[0] if '-wp ' in dsf_cmd else None,
            'epoch': dsf_cmd.split('-we ')[1].split(' ')[0] if '-we ' in dsf_cmd else None,
            'metric': dsf_cmd.split('-wm ')[1].split(' ')[0] if '-wm ' in dsf_cmd else None
        }

        print(f'cvae folder {cvae_folder} and cvae_init_opts')
        pprint(cvae_init_opts)
        self.load_cvae(cvae_folder, cvae_init_opts)

        # load DSF
        self.load_dsf(dsf_folder, dsf_init_opts)

        # eval
        # self.dsf.eval()
        # ADE, FDE, minADE, minFDE, ASD, FSD, minASD, minFSD, rF, dao, dac = self.eval_model(self.dsf, cfg.NSAMPLES, mode='dsf')

        # print(f'ADE {ADE}, FDE {FDE}, minADE {minADE}, minFDE {minFDE}, ASD {ASD}, FSD {FSD}, minASD {minASD}, minFSD {minFSD}, rF {rF}, DAO {dao}, DAC {dac}')


def stack_for_comparison(pred_traj):
    '''
    outputs: predicted targets (npred, npoints, 2)    ex: [12, 6, 2]

    gives 2 stacks of size (npred * npred-1, npoints, 2):
    stack_1: repeated predicted trajectories
    stack_2: trajectories for comparison

    It is better explained with an example so if outputs has 3 trajectories t0, t1, t2:
    stack1 = t0 t0 t1 t1 t2 t2
    stack2 = t1 t2 t0 t2 t0 t1
    '''
    pred_traj = torch.Tensor(pred_traj)
    n, npoints, _ = pred_traj.shape
    size_stacked = n * (n - 1)

    stacked_1 = torch.zeros((size_stacked, npoints, 2))
    stacked_2 = torch.zeros((size_stacked, npoints, 2))

    for i in range(n):
        traj_idx = 0

        indices_1 = [(n-1) * i + j for j in range(n - 1)]
        stacked_1[indices_1,:,:] = pred_traj[i]

        for j in range(n-1):
            # increment index to skip same trajectory
            if i == traj_idx:
                traj_idx += 1

            stacked_2[indices_1[j]] = pred_traj[traj_idx]
            traj_idx += 1

    return stacked_1, stacked_2


# python3 main_pap.py -m train_jointly -a cvae_loc -o joint_trainval_full -g 2 -n jointly_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', required=True, help='train_cvae | train_dsf | eval_dsf | train_jointly')
    parser.add_argument('--gpu', '-g', required=True, help='# of the gpu device')
    parser.add_argument('--options', '-o', help='options yaml file')
    parser.add_argument('--exp_name', '-n', help='optional experiment name')
    parser.add_argument('--debug', '-d', action='store_true', default=False)

    # CVAE specific parameters
    parser.add_argument('--archi', '-a', help='cVAE | cVAE_loc | cVAE_conv')
    parser.add_argument('--lambda_rec', '-lrec', default=1.0, type=float, help='weight for the rec loss')
    parser.add_argument('--beta', '-b', default=1.0, type=float, help='beta like in beta-vae (aka weight on the KL term)')

    # CVAE loading parameters
    parser.add_argument('--load_weights_path', '-wp', help='optional path to the weights that should be loaded')
    parser.add_argument('--load_weights_epoch', '-we', help='load a specific epoch')
    parser.add_argument('--load_weights_metric', '-wm', help='find the checkpoint that has the highest value for given metric')

    # DSF specific parameters
    parser.add_argument('--dsf_archi', '-da', default='both', help='past_only | layout_only | both')
    parser.add_argument('--dsf_loss', '-dl', default='both', help='dpp_only | layout_only | both')
    parser.add_argument('--lambda_loss', '-l', type=float, help='loss weighting parameter')
    parser.add_argument('--cvae_folder', '-cf', help='base cvae to train the DSF on')
    parser.add_argument('--sim_kernel', '-ker', default='mse', help='mse|weighted_l2|final|azimuth')
    parser.add_argument('--fusion', '-f', default='ew', help='ew|plus|concat')
    parser.add_argument('--finetune_cvae', '-fc', action='store_true', default=False, help='finetuning of cvae')
    parser.add_argument('--offset', '-off', action='store_true', default=False, help='offset the dpp loss by 10 to put it in positive range')

    # DSF loading parameters
    parser.add_argument('--dsf_folder', '-df', help='exp folder of the DSF you want to eval')
    parser.add_argument('--dsf_load_weights_path', '-dwp', help='optional path to the weights that should be loaded')
    parser.add_argument('--dsf_load_weights_epoch', '-dwe', help='load a specific epoch')
    parser.add_argument('--dsf_load_weights_metric', '-dwm', help='find the checkpoint that has the highest value for given metric')

    # Joint training parameters
    parser.add_argument('--k', '-k', help='number of trajectories for which to backprop the reconstruction_loss')
    parser.add_argument('--rec_select', '-rec', default='random', help='how those k are selected (random|closest)')

    args = parser.parse_args()

    # eval: fast track
    if args.mode == 'eval_dsf':
        if args.dsf_folder is None:
            raise Exception('Need --dsf_folder option to eval the DSF on')

        # load dsf options
        dsf_options_path = os.path.join(args.dsf_folder, 'config.yaml')
        dsf_options = config_utils.parse_options_file(dsf_options_path)
        config_utils.update_config(dsf_options)

        dsf_init_opts = {
            'path': args.dsf_load_weights_path,
            'epoch': args.dsf_load_weights_epoch,
            'metric': args.dsf_load_weights_metric
        }
        launcher = Launcher('')
        launcher.eval_dsf(args.dsf_folder, dsf_init_opts)

    # options management
    options = config_utils.parse_options_file(args.options)
    config_utils.update_config(options)

    # DSF layout
    dsf_opts = {'dsf': {
        'archi': args.dsf_archi,
        'use_layout_loss': args.dsf_loss in ['both', 'layout_only'],
        'use_dpp_loss': args.dsf_loss in ['both', 'past_only'],
        'lambda': args.lambda_loss,
        'kernel': args.sim_kernel,
        'fusion': args.fusion,
        'finetune_cvae': args.finetune_cvae,
        'offset': args.offset
    }}

    log.printcn(log.OKBLUE, f'dsf opts {dsf_opts}')
    config_utils.update_config(dsf_opts)
    pprint(cfg)

    # init
    exp_folder = exp_utils.exp_init(' '.join(sys.argv), exp_name=(args.exp_name or args.options))

    # CUDA status
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print('CUDA STATUS: \n\tDevice count: %s\n\tIs available: %s\n\tCUDA_VISIBLE_DEVICES: %s' % (torch.cuda.device_count(), torch.cuda.is_available(), os.environ["CUDA_VISIBLE_DEVICES"]))

    init_weights_opts = {
        'path': args.load_weights_path,
        'epoch': args.load_weights_epoch,
        'metric': args.load_weights_metric
    }

    launcher = Launcher(exp_folder, debug=args.debug, init_weights_opts=init_weights_opts)
    if args.mode == 'train_cvae':
        print("training CVAE")
        launcher.train_cvae(args.archi)
    # elif args.mode == 'eval':
    #     launcher.test()
    elif args.mode == 'train_dsf':
        print("training DSF")
        if args.cvae_folder is None:
            raise Exception('Need --cvae_folder option to train the DSF on')
        launcher.train_dsf(args.cvae_folder)
    elif args.mode == 'train_jointly':
        print("jointly training cVAE + DSF")
        launcher.train_jointly(cvae_archi=args.archi, k=int(args.k), rec_select=args.rec_select)

