import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.resnet as resnet


class VAE_encoder(nn.Module):
    def __init__(self, input_size, rnn_units, nlayers, bidirectional, batch_size, latent_dim, device):
        super(VAE_encoder, self).__init__()
        # input_shape: 1 for univariate time series
        # rnn_units: nb of units of the RNN (e.g. 128)
        # nlayers: nb of layers of the RNN
        # latent_dim: dimension of latent vector z
        self.n_directions = 2 if bidirectional is True else 1
        self.device = device

        self.rnn_phi = nn.GRU(input_size=input_size, hidden_size=rnn_units, num_layers=nlayers, bidirectional=bidirectional, batch_first=True)
        self.rnn_x = nn.GRU(input_size=input_size, hidden_size=rnn_units, num_layers=nlayers, bidirectional=bidirectional, batch_first=True)

        self.flat_dim = self.n_directions * rnn_units * 2
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

    def forward(self, past, future):  # x [batch, seq_len, input_dim]
        '''
        Old notation:
            phi = past
            x = future
        '''
        output_past, hidden_past = self.rnn_phi(past)  # output_phi [batch,seq_len,rnn_units*n_directions]
        output_future, hidden_future = self.rnn_x(future)
        features = torch.cat((hidden_past[0], hidden_future[0]), dim=1)   # [batch,2*rnn_units*n_directions]
        z_mu = self.encoder_mu(features)
        z_logvar = self.encoder_logvar(features)
        return z_mu, z_logvar, hidden_past   # avant dernier etat latent

    def encode_past(self, past):
        '''
        sampling only requires past data
        '''
        return self.rnn_phi(past)


class VAE_decoder(nn.Module):
    def __init__(self, input_size, rnn_units, fc_units, nlayers, latent_dim, target_length, device):
        super(VAE_decoder, self).__init__()
        # input_shape: 1 for univariate time series
        # rnn_units: nb of units of the RNN (e.g. 128)
        # nlayers: nb of layers of the RNN
        self.target_length = target_length
        self.device = device

        self.rnn_prediction = nn.GRU(input_size=input_size, hidden_size=rnn_units + latent_dim, num_layers=nlayers, bidirectional=False, batch_first=True)

        self.fc = nn.Linear(rnn_units + latent_dim, fc_units)
        self.output_layer = nn.Linear(fc_units, input_size)  # output_size = input_size

    def forward(self, z, past, hidden_init):  # phi [batch, seq_len, input_dim]    z [batch_size, latent_dim]
        # hidden_init: avant dernier hidden state de l'encoder pour init le decoder
        decoder_input = past[:, -1, :].unsqueeze(1)  # first decoder input = last element of input sequence
        # print('hidden_init dim %s' % str(hidden_init.shape))
        # print('z unsqueezed dim %s' % str(z.unsqueeze(0).shape))
        # print('z dim %s' % str(z.shape))
        decoder_hidden = torch.cat((hidden_init, z.unsqueeze(0)), dim=2)

        outputs = torch.zeros([past.shape[0], self.target_length, past.shape[2]]).to(self.device)
        for di in range(self.target_length):
            decoder_output, decoder_hidden = self.rnn_prediction(decoder_input, decoder_hidden)
            output = F.relu(self.fc(decoder_output))
            output = self.output_layer(output)
            decoder_input = output      # no teacher forcing
            outputs[:, di:di + 1, :] = output
        return outputs


class cVAE(nn.Module):
    def __init__(self,
                 input_size,
                 rnn_units,
                 nlayers,
                 bidirectional,
                 batch_size,
                 latent_dim,
                 fc_units,
                 target_length,
                 device):

        super(cVAE, self).__init__()
        self.input_size = input_size # 1 for univariate time series
        self.rnn_units = rnn_units
        self.latent_dim = latent_dim # z dimension
        self.device = device
        self.encoder = VAE_encoder(input_size, rnn_units, nlayers, bidirectional, batch_size, latent_dim, device)
        self.decoder = VAE_decoder(input_size, rnn_units, fc_units, nlayers, latent_dim, target_length, device)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, past, future):
        z_mu, z_logvar, hidden_init = self.encoder(past, future)  # x [batch_size, target_length, input_size]
        z = self.reparameterize(z_mu, z_logvar)
        # print('training logvar shape %s' % str(z_logvar.shape))
        # print('training z shape %s' % str(z.shape))
        x_mu = self.decoder(z, past, hidden_init)
        return x_mu, z_mu, z_logvar

    def sample(self, past):
        '''
        /!\ ASSUMING A TEST BATCH SIZE OF 1
        '''
        with torch.no_grad():
            # print('sampling past shape %s' % str(past.shape))
            _, hidden_init = self.encoder.encode_past(past)
            z = torch.randn(1, self.latent_dim, device=self.device)     # one by one in test mode
            x_mu = self.decoder(z, past, hidden_init)
            return x_mu