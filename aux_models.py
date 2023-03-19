#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch, copy

from torch import nn
import basicmodels as B
from aux_unet import NormUnet, FlippedNormUnet

class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class DnCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=17, 
                 kernel_size=3, act_mode='BR'):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(DnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        m_head = B.conv(in_nc, nc, kernel_size=kernel_size,
                        mode='C'+act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, kernel_size=kernel_size,
                         mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = B.conv(nc, out_nc, 
                        kernel_size=kernel_size, mode='C', bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        n = self.model(x)
        return x-n

# Unrolled model
class LDAMP(nn.Module):
    def __init__(self, hparams):
        super(LDAMP, self).__init__()
        
        # Extract parameters
        in_channels      = hparams.in_channels
        hidden_channels  = hparams.hidden_channels
        kernel_size      = hparams.kernel_size
        self.shared_nets = hparams.shared_nets
        self.logging     = hparams.model.logging
        
        # Weight init sub-routine
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.trunc_normal_(m.weight, mean=0, std=0.1, a=-0.2, b=0.2)
            
        # Instantiate backbone networks
        if hparams.shared_nets:
            if hparams.backbone == 'DnCNN':
                update_nets = [DnCNN(in_channels, in_channels,
                                hidden_channels, 
                                kernel_size=kernel_size)]
                # Initialize all weights
                update_nets[0].apply(init_weights)
            elif hparams.backbone == 'UNet':
                update_nets = [NormUnet(chans=16,
                                num_pools=3)]
            
        else:
            update_nets = []
            for idx in range(hparams.max_unrolls):
                if hparams.backbone == 'DnCNN':
                    local_net = DnCNN(in_channels, in_channels,
                                    hidden_channels, 
                                    kernel_size=kernel_size)
                    # Initialize all weights
                    local_net.apply(init_weights)
                elif hparams.backbone == 'UNet':
                    local_net = NormUnet(chans=16, num_pools=3)
                elif hparams.backbone == 'FlippedUNet':
                    local_net = FlippedNormUnet(chans=16, num_pools=3)
                update_nets.append(copy.deepcopy(local_net))
            
        # Convert to modules
        self.update_nets = torch.nn.ModuleList(update_nets)
        
        # Safety
        self.safety_min = torch.tensor(0.00001).cuda()
        
    def forward(self, sample, num_unrolls):
        # Unwrap
        y   = sample['Y_herm']
        P   = sample['P_herm']
        eig = sample['eig1']
        
        # Initial values
        h = 0. * torch.randn(y.shape[0], P.shape[-1], y.shape[-1], dtype=y.dtype,
                             device='cuda')
        z = y
        
        # Logging
        if self.logging:
            h_log, z_log, div_log = [], [], []
            r_log, eps_log = [], []
            r_perturbed_log, h_perturbed_log = [], []
        
        # For each unroll
        for unroll_idx in range(num_unrolls):
            # Which net to use
            if self.shared_nets:
                net_idx = 0
            else:
                net_idx = unroll_idx
            
            # Previous iterate
            r = h + 1 / eig[:, None, None] * torch.matmul(P.transpose(-1, -2).conj(), z)
            if self.logging:
                r_log.append(copy.deepcopy(r.cpu().detach().numpy()))
            
            # Denoised signal
            r_real = torch.view_as_real(r)
            h_real = self.update_nets[net_idx](r_real[:, None])[:, 0]
            h      = torch.view_as_complex(h_real)
            
            ## Estimate divergence using Monte Carlo
            ## !!! No grads !!!
            with torch.no_grad():
                # Sample noise in a random direction
                random_dir = torch.randn(r_real.shape, dtype=r_real.dtype,
                                          device='cuda')
                
                # Travel a small amount
                eps = torch.maximum(torch.amax(torch.abs(r), dim=(-1, -2),
                                  keepdim=False) * 1e-3, self.safety_min)
                if self.logging:
                    eps_log.append(copy.deepcopy(eps.cpu().detach().numpy()))
                
                # Add epsilon noise in the random direction
                r_real_perturbed = r_real + eps[:, None, None, None] * random_dir
                
                # Pass noisy version through divergence network by converting to reals
                h_real_perturbed = self.update_nets[net_idx](r_real_perturbed[:, None])[:, 0]
                
                # Estimate divergence - scalar value
                div = 1/eps * \
                    torch.mean(random_dir *
                              (h_real_perturbed - h_real),
                              dim=(-1, -2, -3))
                if self.logging:
                    div_log.append(copy.deepcopy(div.cpu().detach().numpy()))
            
            # Update z
            z = y - torch.matmul(P, h) + z * div[:, None, None]
            
            if self.logging:
                r_perturbed_log.append(
                    copy.deepcopy((r_real_perturbed[:, 0] + 1j *
                                   r_real_perturbed[:, 1]).cpu().detach().numpy()))
                h_perturbed_log.append(
                    copy.deepcopy((h_real_perturbed[:, 0] + 1j *
                                   h_real_perturbed[:, 1]).cpu().detach().numpy()))
                h_log.append(copy.deepcopy(h.cpu().detach().numpy()))
                z_log.append(copy.deepcopy(z.cpu().detach().numpy()))
            
        if self.logging:
            return h, \
                h_log, h_perturbed_log, z_log, div_log, r_log, r_perturbed_log, eps_log
        else:
            return h