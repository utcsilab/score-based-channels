#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 13:51:16 2021

@author: yanni
"""

import torch, hdf5storage
from torch.utils.data import Dataset
import numpy as np

class Channels(Dataset):
    """MIMO Channels"""

    def __init__(self, seed, config, norm=None):
        # Get spacings
        target_spacings = config.data.spacing_list
        target_channel  = config.data.channel
        # Output channels
        self.channels  = []
        self.spacings  = np.copy(target_spacings)
        self.filenames = []
        
        # For each spacing
        for spacing in target_spacings:
            # Get local filename
            if target_channel == 'CDL-D':
                filename = './data/%s_Nt64_Nr16_ULA%.1f_seed%d.mat' % (
                    target_channel, spacing, seed)
            else:
                filename = './data/%s_Nt64_Nr16_ULA%.2f_seed%d.mat' % (
                    target_channel, spacing, seed)
            # Log
            self.filenames.append(filename)
        
            # Preload file and serialize
            contents = hdf5storage.loadmat(filename)
            channels = np.asarray(contents['output_h'], dtype=np.complex64)
        
            # For now, only pick first subcarrier/symbol content
            if config.data.mixed_channels:
                self.channels.append(channels.reshape(
                    -1, channels.shape[-2], channels.shape[-1]))
            else:
                self.channels.append(channels[:, 0])
                
        # Convert to array
        self.channels = np.asarray(self.channels)
        self.channels = np.reshape(self.channels,
               (-1, self.channels.shape[-2], self.channels.shape[-1]))
            
        # Normalize
        if type(norm) == list:
            self.mean = norm[0]
            self.std  = norm[1]
        elif norm == 'entrywise':
            self.mean = np.mean(self.channels, axis=0)
            self.std  = np.std(self.channels, axis=0)
        elif norm == 'global':
            self.mean = 0.
            self.std  = np.std(self.channels)
            
        # Generate random QPSK pilots
        self.pilots = 1/np.sqrt(2) * (2 * np.random.binomial(1, 0.5, size=(
            self.channels.shape[0], config.data.image_size[1], config.data.num_pilots)) - 1 + \
                1j * (2 * np.random.binomial(1, 0.5, size=(
            self.channels.shape[0], config.data.image_size[1], config.data.num_pilots)) - 1))
            
        # Complex noise power
        self.noise_power = 1/np.sqrt(2) * config.data.noise_std
        
    def __len__(self):
        return len(self.channels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Normalize
        H_cplx = self.channels[idx]
        H_cplx_norm = (H_cplx - self.mean) / self.std
        
        # Convert to reals
        H_real_norm = \
            np.stack((np.real(H_cplx_norm), np.imag(H_cplx_norm)), axis=0)
        
        # Get complex pilots and create noisy Y
        P = self.pilots[idx]
        Y = np.matmul(H_cplx, P)
        N = self.noise_power * (np.random.normal(size=Y.shape) + \
                                1j * np.random.normal(size=Y.shape))
        Y = Y + N

        # Also get Hermitian H, real-viewed
        H_herm_norm = np.conj(np.transpose(H_cplx_norm))
        H_real_herm_norm = \
            np.stack((np.real(H_herm_norm), np.imag(H_herm_norm)), axis=0)

        sample = {'H': H_real_norm.astype(np.float32),
                  'H_herm': H_real_herm_norm.astype(np.float32),
                  'P': self.pilots[idx].astype(np.complex64),
                  'Y': Y.astype(np.complex64),
                  'sigma_n': self.noise_power.astype(np.float32)}

        return sample