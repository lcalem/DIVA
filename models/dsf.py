import numpy as np
import torch
import torch.nn as nn


class DSF(nn.Module):
    def __init__(self, cvae, nsamples, target_length):
        super(DSF, self).__init__()

        self.cVAE = cvae          # trained cVAE model
        self.nsamples = nsamples
        self.target_length = target_length
        self.latent_dim = cvae.latent_dim

        self.MLP = nn.Sequential(
            nn.BatchNorm1d(self.cVAE.rnn_units),
            nn.Linear(self.cVAE.rnn_units, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, self.latent_dim * self.nsamples) # generate the latent state
        )

    def forward(self, past, layout):
        '''
        # past [batch_size,seq_length,input_size]
        tanh * 4 because prior is N(0,1) with a support around [-4,4]
        '''
        batch_size, seq_len, nfeatures = past.shape
        output_past, hidden_past = self.cVAE.encoder.rnn_past(past)  # hidden_past [nlayers * ndirections, batch_size, rnn_units]
        l = self.cVAE.encoder.local_convnet(layout)

        # print(f'hidden past shape {hidden_past.shape}')

        sampled_z = self.MLP(hidden_past[0,:,:]) # sampled_z : [batch_size, latent_dim * nsamples]
        sampled_z = torch.tanh(sampled_z) * 4

        outputs = torch.zeros([batch_size, self.nsamples, self.target_length, nfeatures]).to(self.cVAE.device)

        for k in range(0, self.nsamples):
            z_dsf = sampled_z[:, self.latent_dim * k:self.latent_dim * (k+1)] # [batch_size, latent_dim]

            x_mu = self.cVAE.decoder(z_dsf, past, hidden_past, l) # [batch_size, target_length, nfeatures]
            outputs[:,k,:,:] = x_mu

        return outputs, sampled_z  # outputs [batch_size, nsamples, target_len, nfeatures]


class DSF_layout(nn.Module):

    def __init__(self, cvae_loc, nsamples, target_length):
        '''
        target_length = N_OUTPUT
        layout branch only // only for ablation study
        '''
        print(type(cvae_loc))
        print(type(self))
        super().__init__()

        self.cVAE = cvae_loc          # trained cVAE model
        self.nsamples = nsamples
        self.target_length = target_length
        self.latent_dim = cvae_loc.latent_dim
        self.local_dim = cvae_loc.local_dim

        self.output_dim = self.latent_dim * self.nsamples

        print(f'DSF output dim {self.output_dim}')

        self.MLP_loc = nn.Sequential(
            nn.BatchNorm1d(self.local_dim),
            nn.Linear(self.local_dim, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, self.output_dim)
        )

    def forward(self, past, layout):
        '''
        Uses the past MLP and the local MLP to sample latent codes
        The final sampled z is the element wise product of the two elements

        tanh * 4 because prior is N(0,1) with a support around [-4,4]
        '''

        batch_size, seq_len, nfeatures = past.shape
        output_past, hidden_past = self.cVAE.encoder.rnn_past(past)  # hidden_past [nlayers * ndirections, batch_size, rnn_units]
        l = self.cVAE.encoder.local_convnet(layout)

        sampled_z = self.MLP_loc(l)
        sampled_z = torch.tanh(sampled_z) * 4

        outputs = torch.zeros([batch_size, self.nsamples, self.target_length, nfeatures]).to(self.cVAE.device)

        for k in range(0, self.nsamples):
            z_dsf = sampled_z[:, self.latent_dim * k:self.latent_dim * (k+1)] # [batch_size, latent_dim]

            future = self.cVAE.decoder(z_dsf, past, hidden_past, l) # [batch_size, target_length, nfeatures]
            outputs[:,k,:,:] = future

        return outputs, sampled_z  # outputs [batch_size, nsamples, target_len, nfeatures]


class DSF_loc(nn.Module):

    def __init__(self, cvae_loc, nsamples, target_length, fusion):
        '''
        target_length = N_OUTPUT
        fusion = how to combine z_l and z_p (element-wise product or sum, concat)
        '''
        print(type(cvae_loc))
        print(type(self))
        super().__init__()

        self.cVAE = cvae_loc          # trained cVAE model
        self.nsamples = nsamples
        self.target_length = target_length
        self.latent_dim = cvae_loc.latent_dim
        self.local_dim = cvae_loc.local_dim
        self.fusion = fusion
        assert fusion in ('plus', 'ew', 'concat')

        self.output_dim = self.latent_dim * self.nsamples

        # concat = we concat 2 half dims
        if self.fusion == 'concat':
            self.output_dim = int(self.output_dim / 2)

        print(f'DSF output dim {self.output_dim}')

        self.MLP = nn.Sequential(
            nn.BatchNorm1d(self.cVAE.rnn_units),
            nn.Linear(self.cVAE.rnn_units, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, self.output_dim)
        )

        self.MLP_loc = nn.Sequential(
            nn.BatchNorm1d(self.local_dim),
            nn.Linear(self.local_dim, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, self.output_dim)
        )

    def forward(self, past, layout):
        '''
        Uses the past MLP and the local MLP to sample latent codes
        The final sampled z is the element wise product of the two elements

        tanh * 4 because prior is N(0,1) with a support around [-4,4]
        '''
        batch_size, seq_len, nfeatures = past.shape
        output_past, hidden_past = self.cVAE.encoder.rnn_past(past)  # hidden_past [nlayers * ndirections, batch_size, rnn_units]
        l = self.cVAE.encoder.local_convnet(layout)

        # print(f'loc shape {l.shape}')
        # print(f'hidden past shape {hidden_past.shape}')

        sampled_z_p = self.MLP(hidden_past[0,:,:])
        sampled_z_p = torch.tanh(sampled_z_p) * 4

        sampled_z_l = self.MLP_loc(l)
        sampled_z_l = torch.tanh(sampled_z_l) * 4

        if self.fusion == 'ew':
            sampled_z = sampled_z_p * sampled_z_l   # sampled_z : [batch_size, latent_dim * nsamples]
        elif self.fusion == 'plus':
            sampled_z = sampled_z_p + sampled_z_l   # sampled_z : [batch_size, latent_dim * nsamples]
        elif self.fusion == 'concat':
            sampled_z = torch.cat((sampled_z_p, sampled_z_l), 1) # sampled_z : [batch_size, latent_dim * nsamples]

        assert sampled_z.shape == (batch_size, self.latent_dim * self.nsamples), f'{sampled_z.shape}'

        outputs = torch.zeros([batch_size, self.nsamples, self.target_length, nfeatures]).to(self.cVAE.device)

        for k in range(0, self.nsamples):
            z_dsf = sampled_z[:, self.latent_dim * k:self.latent_dim * (k+1)] # [batch_size, latent_dim]

            future = self.cVAE.decoder(z_dsf, past, hidden_past, l) # [batch_size, target_length, nfeatures]
            outputs[:,k,:,:] = future

        return outputs, sampled_z  # outputs [batch_size, nsamples, target_len, nfeatures]
