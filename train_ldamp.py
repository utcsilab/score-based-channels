#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 12:28:29 2022

@author: marius
"""

import sys, os
sys.path.append('..')
import numpy as np

import torch
from torch.optim.lr_scheduler import StepLR

from aux_models import LDAMP
from dotmap import DotMap
from tqdm import tqdm

from loaders          import Channels
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--alpha', type=float, default=0.6)
parser.add_argument('--channel', type=str, default='CDL-B')
parser.add_argument('--snr_values', nargs='+', type=float, 
                    # default=np.arange(-10, 17.5, 2.5))
                    default=np.arange(-30, 17.5, 2.5))
args = parser.parse_args()

# Always !!!
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False
torch.backends.cudnn.benchmark        = True

# GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# For each noise level
for noise_idx, train_snr in enumerate(args.snr_values):
    # Model config
    config          = DotMap()
    config.device   = 'cuda:0'
    config.purpose  = 'train'
    # Inner model
    config.model.in_channels     = 2
    config.model.hidden_channels = 32
    config.model.backbone        = 'FlippedUNet'
    config.model.kernel_size     = 3
    config.model.max_unrolls     = 10
    config.model.shared_nets     = False
    config.model.logging         = False
    
    # Optimizer
    config.optim.lr              = 1e-3
    # Training
    config.training.batch_size   = 128
    config.training.num_workers  = 2
    config.training.n_epochs     = 24
    config.training.decay_epochs = 16
    config.training.decay_gamma  = 0.1
    config.training.eval_freq    = 20 # In epochs
    
    # Data
    config.data.channels       = 2
    config.data.image_size     = [16, 64]
    config.data.array          = 'ULA'
    config.data.num_pilots     = int(config.data.image_size[1] * args.alpha)
    config.data.train_snr      = np.asarray([train_snr])
    config.data.noise_std      = 10 ** (-config.data.train_snr / 20.)
    config.model.multi_snr     = len(config.data.train_noise) > 1
    config.data.mixed_channels = False
    config.data.norm_channels  = 'none'
    config.data.channel        = args.channel
    config.data.spacing_list   = [0.5]
    # Unclear if this is needed
    config.input_dim           = \
        np.prod(config.data.image_size) * config.data.channels
        
    # Universal seeds
    train_seed, val_seed = 1234, 4321
    
    # Get datasets and loaders for channels
    dataset     = Channels(train_seed, config, norm=config.data.norm_channels)
    dataloader  = DataLoader(dataset, batch_size=config.training.batch_size, 
             shuffle=True, num_workers=config.training.num_workers,
             drop_last=True, pin_memory=True)
    
    # Get a model
    model = LDAMP(config.model)
    model = model.cuda()
    model.train()
    
    # Number of parameters
    num_params = np.sum([np.prod(p.shape) for p in model.parameters() if
                         p.requires_grad])
    print('Model has %d trainable parameters!' % num_params)
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), config.optim.lr)
    scheduler = StepLR(optimizer, config.training.decay_epochs,
                       gamma=config.training.decay_gamma)
    
    # Result directory
    if config.model.multi_snr is True:
        assert False, 'Deprecated!'
        local_dir = 'models_ldamp_%s/multi_snr_alpha%.1f' % (
            config.model.backbone, args.alpha)
    else:
        local_dir = 'models_ldamp_%s_correctSNR/%s_snr%.1f_alpha%.1f' % (
            config.model.backbone, args.channel,
            config.data.train_snr[0], args.alpha)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    # Logs
    loss_log, nmse_log = [], []
    # For each epoch
    for epoch_idx in tqdm(range(config.training.n_epochs)):
        # For each batch
        for batch_idx, sample in tqdm(enumerate(dataloader)):
            # Move samples to GPU
            for key in sample.keys():
                sample[key] = sample[key].cuda()
            
            # Inference
            H_est = model(sample, config.model.max_unrolls)
            
            # End-to-end training
            loss = torch.mean(
                torch.sum(torch.square(torch.abs(H_est - sample['H_herm_cplx'])), 
                          dim=(-1, -2)))
            
            # Logging
            with torch.no_grad():
                nmse_loss = torch.mean(
                    torch.sum(torch.square(torch.abs(
                        H_est - sample['H_herm_cplx'])), 
                              dim=(-1, -2))/
                    torch.sum(torch.square(torch.abs(sample['H_herm_cplx'])), 
                              dim=(-1, -2)))
                
                loss_log.append(loss.item())
                nmse_log.append(nmse_loss.item())
            
            # Verbose
            print('Epoch %d, Step %d, Train Loss %.3f, Train NMSE %.3f [dB]' % (
                epoch_idx, batch_idx, loss.item(), 
                10 * np.log10(nmse_loss.item())))
            
            # Backpop
            optimizer.zero_grad()
            loss.backward()
            # Clip gradients
            # torch.nn.utils.clip_grad_norm(model.parameters(), 1.)
            optimizer.step()
            
        # Periodic eval and saving
        if np.mod(epoch_idx+1, config.training.eval_freq) == 0:
            torch.save({'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'config': config,
                        'loss_log': loss_log,
                        'nmse_log': nmse_log}, local_dir + '/model_epoch%d.pt' % (
                            epoch_idx))
        # Scheduler
        scheduler.step()
        
    # Final save
    torch.save({'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'config': config,
                'loss_log': loss_log,
                'nmse_log': nmse_log}, local_dir + '/final_model.pt')