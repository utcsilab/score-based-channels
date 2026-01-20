#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
sys.path.append('..')
import numpy as np

import torch
from torch.optim.lr_scheduler import StepLR

from .aux_models import LDAMP
from dotmap import DotMap
from tqdm import tqdm

from .loaders          import Channels
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--alpha', type=float, default=0.6)
parser.add_argument('--train', type=str, default='CDL-C')
parser.add_argument('--snr_range', nargs='+', type=float, 
                    default=np.arange(-10, 35, 5))
args = parser.parse_args()

# Disable TF32 due to potential precision issues
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False
torch.backends.cudnn.benchmark        = True
# GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# For each SNR level instantiate and train a new model
for noise_idx, train_snr in enumerate(args.snr_range):
    # Configuration
    config          = DotMap()
    config.device   = 'cuda:0'
    # Inner model architecture
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
    config.training.num_workers  = 2 if os.name == "posix" else 0
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
    # TODO: For LDAMP, this is currently amplitude not power
    config.data.noise_std      = 10 ** (-config.data.train_snr / 20.) * np.sqrt(config.data.image_size[1])
    config.model.multi_snr     = False
    config.data.mixed_channels = False
    config.data.norm_channels  = 'global'
    config.data.channel        = args.train
    config.data.spacing_list   = [0.5]
    config.input_dim           = \
        np.prod(config.data.image_size) * config.data.channels
        
    # Train and test seeds
    train_seed, val_seed = 1234, 4321
    
    # Get datasets and loaders for channels
    dataset     = Channels(train_seed, config, norm=config.data.norm_channels)
    dataloader  = DataLoader(dataset, batch_size=config.training.batch_size, 
             shuffle=True, num_workers=config.training.num_workers,
             drop_last=True, pin_memory=True)
    
    # Get model
    model = LDAMP(config.model)
    model = model.cuda()
    model.train()
    
    # Print number of parameters
    num_params = np.sum([np.prod(p.shape) for p in model.parameters() if
                         p.requires_grad])
    print('Model has %d trainable parameters' % num_params)
    
    # Get optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), config.optim.lr)
    scheduler = StepLR(optimizer, config.training.decay_epochs,
                       gamma=config.training.decay_gamma)
    
    # Result directory
    local_dir = './models/ldamp-%s/train-%s' % (
        config.model.backbone, args.train)
    os.makedirs(local_dir, exist_ok=True)
    
    # Logs
    loss_log, nmse_log = [], []
    # For each epoch
    for epoch_idx in tqdm(range(config.training.n_epochs)):
        # For each batch
        for batch_idx, sample in tqdm(enumerate(dataloader)):
            # Move samples to GPU
            for key in sample.keys():
                sample[key] = sample[key].cuda()
            
            # Get output of model
            H_est = model(sample, config.model.max_unrolls)
            
            # End-to-end training using MSE loss
            loss = torch.mean(
                torch.sum(torch.square(torch.abs(H_est - sample['H_herm_cplx'])), 
                          dim=(-1, -2)))
            
            # Log MSE and NMSE after every step
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
            print('Epoch %d, Step %d, Train Loss %.3f, Train SNR %.2f dB, Train NMSE %.3f dB' % (
                epoch_idx, batch_idx, loss.item(), train_snr, 10 * np.log10(nmse_loss.item())))
            
            # Backpop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Increment scheduler after every epoch
        scheduler.step()
        
    # Save model weights
    torch.save({'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'config': config, 'args': args,
                'loss_log': loss_log,
                'nmse_log': nmse_log
                }, 
               os.path.join(local_dir, 'model_snr%.2f_alpha%.2f.pt' % (
                   config.data.train_snr[0], args.alpha)))