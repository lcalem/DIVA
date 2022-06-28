
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.resnet as resnet

        self.E_poly = None  # evaluation of the
class VAEloc_encoder_cut(nn.Module):
    def __init__(self, input_size, rnn_units, nlayers, bidirectional, batch_size, latent_dim, local_dim, pretrained, device):
        '''
        loc: l
        convnet: cut (half of a resnet18) or fc (full resnet18 with fc mapping to local_dim) [both have the same output size]
        '''
        super(VAEloc_encoder_cut, self).__init__()
        # input_shape: 1 for univariate time series
        # rnn_units: nb of units of the RNN (e.g. 128)
        # nlayers: nb of layers of the RNN
        # latent_dim: dimension of latent vector z

        self.latent_dim = latent_dim
        self.n_directions = 2 if bidirectional is True else 1
        self.device = device

        self.rnn_past = nn.GRU(input_size=input_size, hidden_size=rnn_units, num_layers=nlayers, bidirectional=bidirectional, batch_first=True)
        # self.rnn_future = nn.GRU(input_size=input_size, hidden_size=rnn_units, num_layers=nlayers, bidirectional=bidirectional, batch_first=True)

        self.local_convnet = resnet.resnet18_fc(local_dim, pretrained=pretrained)

        # self.flat_dim = self.n_directions * rnn_units * 2 + local_dim
        self.flat_dim = self.n_directions * rnn_units

        self.encoder_mu = nn.Sequential(
            # nn.BatchNorm1d(self.flat_dim),
            nn.Linear(self.flat_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, latent_dim)
        )
        self.encoder_logvar = nn.Sequential(
            # nn.BatchNorm1d(self.flat_dim),
            nn.Linear(self.flat_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, past, future, layout):
        '''
        past:       (BS, past_horizon, 2)
        future:     (BS, future_horizon, 2)
        layout:     (BS, 224, 224, 3)           # IMG
        '''
        l = self.local_convnet(layout)
        # print('local shape %s' % str(l.shape))

        output_past, hidden_past = self.rnn_past(past)  # output_past [batch,seq_len,rnn_units*n_directions]
        # print(f'c la shape du hidden past bidirectional qui devrait etre 256 {hidden_past.shape}')
        # print('hidden_past shape %s' % str(hidden_past.shape))
        # output_future, hidden_future = self.rnn_future(future)
        # print('hidden_future shape %s' % str(hidden_future.shape))

        # concat everything
        features = hidden_past[0]
        # features = torch.cat((hidden_past[0], hidden_future[0]), dim=1)   # [batch, 2*rnn_units * n_directions + local_embedding_size]
        # print(f'la shape des cat features {features.shape}')
        # print('mu & logvar features shape %s' % str(features.shape))
        z_mu = self.encoder_mu(features)
        z_logvar = self.encoder_logvar(features)
        return z_mu, z_logvar, hidden_past, l   # hidden_past: penultimate latent state

    def encode_past(self, past, layout):
        '''
        sampling only requires past data
        '''
        _, hidden_init = self.rnn_past(past)
        l = self.local_convnet(layout)

        z_mu = self.encoder_mu(hidden_init[0])
        z_logvar = self.encoder_logvar(hidden_init[0])
        return hidden_init, l, z_mu, z_logvar


class VAEloc_decoder_cut(nn.Module):
    def __init__(self, input_size, rnn_units, fc_units, nlayers, latent_dim, local_dim, target_length, device):
        '''
        input_shape: 1 for univariate time series, 2 for
        rnn_units: nb of units of the RNN (e.g. 128)
        nlayers: nb of layers of the RNN
        local_dim: embedding size of the local features (output of the convnet)
        '''
        super(VAEloc_decoder_cut, self).__init__()
        self.target_length = target_length
        self.device = device

        self.rnn_prediction = nn.GRU(input_size=input_size, hidden_size=local_dim + latent_dim, num_layers=nlayers, bidirectional=False, batch_first=True)

        self.fc = nn.Linear(rnn_units + latent_dim + local_dim, fc_units)
        self.output_layer = nn.Linear(fc_units, input_size)  # output_size = input_size

    def forward(self, z, past, l):
        '''
        past    [BS, seq_len, input_dim]    past embedding
        z       [BS, latent_dim]            latent code
        l       [BS, local_dim]             local layout embedding
        hidden_init: avant dernier hidden state de l'encoder pour init le decoder
        '''

        decoder_input = past[:, -1, :].unsqueeze(1)  # first decoder input = last element of input sequence
        # print('hidden_init dim %s' % str(hidden_init.shape))
        # print('z unsqueezed dim %s' % str(z.unsqueeze(0).shape))
        # print('z dim %s' % str(z.shape))
        # print('l dim %s' % str(l.shape))
        decoder_hidden = torch.cat((l.unsqueeze(0), z.unsqueeze(0)), dim=2)
        # print('decoder_hidden size %s' % str(decoder_hidden.shape))

        outputs = torch.zeros([past.shape[0], self.target_length, past.shape[2]]).to(self.device)
        for di in range(self.target_length):
            decoder_output, decoder_hidden = self.rnn_prediction(decoder_input, decoder_hidden)
            output = F.relu(self.fc(decoder_output))
            output = self.output_layer(output)
            decoder_input = output      # no teacher forcing
            outputs[:, di:di + 1, :] = output

        return outputs


class cVAE_cut(nn.Module):
    '''
    cVAE but with a resnet module to take care of the layout representation (loc = local features)
    It also uses the HD maps like cVAE_loc but doesn't use the future
    uses p(z|past) and not p(z) as a prior
    '''
    def __init__(self,
                 input_size,
                 rnn_units,
                 nlayers,
                 bidirectional,
                 batch_size,
                 latent_dim,
                 local_dim,
                 fc_units,
                 target_length,
                 pretrained_enc,
                 device):

        super(cVAE_loc_cut, self).__init__()
        self.input_size = input_size  # 1 for univariate time series, 2 for 2D trajectories
        self.rnn_units = rnn_units
        self.latent_dim = latent_dim  # z dimension
        self.local_dim = local_dim    # local embedding dimension

        self.device = device
        self.encoder = VAEloc_encoder_cut(input_size, rnn_units, nlayers, bidirectional, batch_size, latent_dim, local_dim, pretrained_enc, device)
        self.decoder = VAEloc_decoder_cut(input_size, rnn_units, fc_units, nlayers, latent_dim, local_dim, target_length, device)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, past, future, layout):
        z_mu, z_logvar, hidden_init, l = self.encoder(past, future, layout)  # x [batch_size, target_length, input_size]
        z = self.reparameterize(z_mu, z_logvar)
        # print('training logvar shape %s' % str(z_logvar.shape))
        # print('training z shape %s' % str(z.shape))
        x_mu = self.decoder(z, past, l)
        return x_mu, z_mu, z_logvar

    def sample(self, past, layout):
        '''
        /!\ ASSUMING A TEST BATCH SIZE OF 1
        '''
        with torch.no_grad():
            # print('sampling past shape %s' % str(past.shape))
            hidden_init, l, z_mu, z_logvar = self.encoder.encode_past(past, layout)
            z = self.reparameterize(z_mu, z_logvar)
            # z = torch.randn(1, self.latent_dim, device=self.device)     # one by one in test mode
            x_mu = self.decoder(z, past, l)
            return x_mu
