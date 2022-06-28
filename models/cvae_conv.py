import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.resnet as resnet


class VAEconv_encoder(nn.Module):
    def __init__(self, input_size, rnn_units, nlayers, bidirectional, batch_size, latent_dim, local_dim, pretrained, device):
        '''
        loc: l
        convnet: cut (half of a resnet18) or fc (full resnet18 with fc mapping to local_dim) [both have the same output size]
        '''
        super(VAEconv_encoder, self).__init__()
        # input_shape: 1 for univariate time series
        # rnn_units: nb of units of the RNN (e.g. 128)
        # nlayers: nb of layers of the RNN
        # latent_dim: dimension of latent vector z

        self.n_directions = 2 if bidirectional is True else 1
        self.device = device

        self.local_convnet = resnet.resnet18_fc(local_dim, pretrained=pretrained)

        #self.flat_dim = self.n_directions * rnn_units * 2 + local_dim
        self.flat_dim = local_dim

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

    def forward(self, layout):
        '''
        past:       (BS, past_horizon, 2)
        layout:     (BS, 224, 224, 3)           # IMG

        For this encoder the future & past are not used at all
        '''
        l = self.local_convnet(layout)
        # print('local shape %s' % str(l.shape))

        z_mu = self.encoder_mu(l)
        z_logvar = self.encoder_logvar(l)
        return z_mu, z_logvar, l   # hidden_past: penultimate latent state

    def encode_past(self, layout):
        '''
        sampling only requires layout data
        '''
        l = self.local_convnet(layout)
        return l


class VAEloc_decoder(nn.Module):
    def __init__(self, input_size, rnn_units, fc_units, nlayers, latent_dim, local_dim, target_length, device):
        '''
        input_shape: 1 for univariate time series, 2 for
        rnn_units: nb of units of the RNN (e.g. 128)
        nlayers: nb of layers of the RNN
        local_dim: embedding size of the local features (output of the convnet)
        '''
        super(VAEloc_decoder, self).__init__()
        self.target_length = target_length
        self.device = device

        self.rnn_prediction = nn.GRU(input_size=input_size, hidden_size=local_dim + latent_dim, num_layers=nlayers, bidirectional=False, batch_first=True)

        self.fc = nn.Linear(latent_dim + local_dim, fc_units)
        self.output_layer = nn.Linear(fc_units, input_size)  # output_size = input_size

    def forward(self, z, past, l):
        '''
        past    [BS, seq_len, input_dim]    past embedding
        z       [BS, latent_dim]            latent code
        l       [BS, local_dim]             local layout embedding
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


class cVAE_conv(nn.Module):
    '''
    cVAE with the encoder as a resnet (no RNN encoding part)
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

        super(cVAE_conv, self).__init__()
        self.input_size = input_size  # 1 for univariate time series, 2 for 2D trajectories
        self.rnn_units = rnn_units
        self.latent_dim = latent_dim  # z dimension
        self.local_dim = local_dim    # local embedding dimension

        self.device = device
        self.encoder = VAEconv_encoder(input_size, rnn_units, nlayers, bidirectional, batch_size, latent_dim, local_dim, pretrained_enc, device)
        self.decoder = VAEloc_decoder(input_size, rnn_units, fc_units, nlayers, latent_dim, local_dim, target_length, device)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, past, future, layout):
        z_mu, z_logvar, l = self.encoder(layout)  # x [batch_size, target_length, input_size]
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
            l = self.encoder.encode_past(layout)
            z = torch.randn(1, self.latent_dim, device=self.device)     # one by one in test mode
            x_mu = self.decoder(z, past, l)
            return x_mu
