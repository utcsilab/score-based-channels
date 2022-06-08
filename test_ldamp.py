#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 12:28:29 2022

@author: marius
"""

import sys, os, copy, itertools, hdf5storage
sys.path.append('..')
import numpy as np

import torch
from aux_models import LDAMP
from tqdm import tqdm

from loaders          import Channels
from torch.utils.data import DataLoader
from matplotlib       import pyplot as plt

# Always !!!
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False
torch.backends.cudnn.benchmark        = True

# GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Target channel model
train_channels  = 'CDL-B'
target_channels = 'CDL-D'
# Meta-configs
spacing_range     = [0.5]
pilot_alpha_range = [0.6]
snr_range         = np.arange(-30, 17.5, 2.5)

train_seed, test_seed = 1234, 4321

# Wrap spacing, sparsity and SNR
meta_params = itertools.product(spacing_range, pilot_alpha_range, snr_range)

# Global results
num_channels = 100 # For testing
oracle_log   = np.zeros((len(spacing_range), len(pilot_alpha_range),
                       len(snr_range), num_channels)) # Should match data
# Saved H
saved_H = np.zeros((len(snr_range), num_channels, 64, 16),
                   dtype=np.complex64)
result_dir = 'LDAMP_journal_final'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# For each hyper-combo
for meta_idx, (spacing, pilot_alpha, snr) in tqdm(enumerate(meta_params)):
    # Unwrap indices
    spacing_idx, pilot_alpha_idx, snr_idx = np.unravel_index(
        meta_idx, (len(spacing_range), len(pilot_alpha_range),
                   len(snr_range)))
    
    # Fetch config and load model
    if train_channels == 'CDL-D':
        target_file = 'models_ldamp_FlippedUNet/snr%.1f_alpha%.1f/final_model.pt' % (
            snr, pilot_alpha)
    else:
        target_file = 'models_ldamp_FlippedUNet_correctSNR/%s_snr%.1f_alpha%.1f/final_model.pt' % (
            train_channels, snr, pilot_alpha)
    contents = torch.load(target_file)
    config   = contents['config']
    
    # Create a model (just once) and load weights
    if meta_idx == 0:
        model = LDAMP(config.model)
        model = model.cuda()
    model.eval()
    model.load_state_dict(contents['model_state'])
    
    # Get a validation dataset and adjust parameters
    val_config = copy.deepcopy(config)
    val_config.purpose           = 'val'
    val_config.data.channel      = target_channels
    val_config.data.spacing_list = [spacing]
    val_config.data.train_pilots = config.data.train_pilots
    val_config.data.train_snr    = np.asarray([snr])
    val_config.data.noise_std    = 10 ** (-val_config.data.train_snr / 20.)
    
    # Get training dataset - just for normalization purposes, only once
    if np.isin(train_channels, ['CDL-D', 'CDL-C', 'CDL-B']):
        if meta_idx == 0:
            ref_dataset = Channels(
                train_seed, config, norm=config.data.norm_channels)
            norm        = [ref_dataset.mean, ref_dataset.std]
    else:
        norm = [0., 1.]
        
    # Get validation dataset
    dataset     = Channels(test_seed, val_config, norm=norm)
    dataloader  = DataLoader(dataset, batch_size=num_channels,
                             shuffle=False, num_workers=0, drop_last=True)
    print('There are %d validation channels!' % len(dataset))
    
    # For each batch
    with torch.no_grad():
        for batch_idx, sample in tqdm(enumerate(dataloader)):
            # Move samples to GPU
            for key in sample.keys():
                sample[key] = sample[key].cuda()
            
            # Get ground truth
            val_H_herm = sample['H_herm']
            val_H = val_H_herm[:, 0] + 1j * val_H_herm[:, 1]
            
            # Inference
            H_est = model(sample, config.model.max_unrolls)
    
            # Compute NMSE
            nmse_loss = \
                torch.sum(torch.square(torch.abs(H_est - val_H)),
                          dim=(-1, -2))/\
                torch.sum(torch.square(torch.abs(val_H)), 
                          dim=(-1, -2))
            
            # Store NMSE
            oracle_log[spacing_idx, pilot_alpha_idx, snr_idx] = \
                nmse_loss.cpu().detach().numpy()
                
    # Save channels
    saved_H[snr_idx] = H_est.cpu().numpy()

# Save estimated channels in Matlab format
ideal_H = val_H.cpu().numpy()
est_H   = saved_H
sanity_error = np.mean(np.sum(np.square(np.abs(est_H - ideal_H[None, :])),
                              axis=(-1, -2)) /\
                       np.sum(np.square(np.abs(ideal_H[None, :])),
                  axis=(-1, -2)), axis=-1)
    
save_dict  = {'ideal_H': ideal_H,
              'est_H': est_H}
hdf5storage.savemat(result_dir + '/saved_channels_LDAMP_%s_to_%s_alpha%.1f.mat' % (
    train_channels, target_channels, pilot_alpha_range[0]), 
    save_dict, truncate_existing=True)

# Save results
mean_oracle = np.mean(oracle_log[0], axis=-1)
for alpha_idx, local_alpha in enumerate(pilot_alpha_range):
    if spacing_range[0] == 0.1:
        torch.save({'ldamp': mean_oracle[alpha_idx]},
                   result_dir + '/ldamp_results_%s_to_%s_lam1p10_alpha%.1f.pt' % (
                       train_channels, target_channels, local_alpha))
    elif spacing_range[0] == 0.5:
        torch.save({'ldamp': mean_oracle[alpha_idx]},
                   result_dir + '/ldamp_results_%s_to_%s_lam1p2_alpha%.1f.pt' % (
                       train_channels, target_channels, local_alpha))
    
# Average and plot
# labels = ['alpha = %.2f' % pilot_alpha_range[idx] for idx in range(len(pilot_alpha_range))]
# plt.rcParams['font.size'] = 22
# plt.figure(figsize=(12, 12))
# plt.plot(snr_range, 10 * np.log10(mean_oracle).T, 
#          linewidth=4, label=labels)
# plt.xlabel('SNR [dB]'); plt.ylabel('NMSE [dB]')
# plt.title('Learned Denoising AMP')
# plt.legend(); plt.grid()
# if target_channels == 'CDL-C':
#     plt.ylim([-25, 6])
# plt.tight_layout()
# plt.savefig('ldamp_%s_to_%s_spacing%.1f.png' % (
#     train_channels, target_channels, spacing_range[0]))
# plt.close()