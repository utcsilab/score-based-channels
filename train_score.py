#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch, sys, os, copy, argparse
sys.path.append('./')

from tqdm import tqdm as tqdm

from ncsnv2.models        import get_sigmas
from ncsnv2.models.ema    import EMAHelper
from ncsnv2.models.ncsnv2 import NCSNv2Deepest
from ncsnv2.losses        import get_optimizer
from ncsnv2.losses.dsm    import anneal_dsm_score_estimation

from loaders          import Channels
from torch.utils.data import DataLoader
from dotmap           import DotMap

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train', type=str, default='CDL-C')
args = parser.parse_args()

# Disable TF32 due to potential precision issues
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False
torch.backends.cudnn.benchmark        = True
# GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# Model config
config          = DotMap()
config.device   = 'cuda:0'
# Inner model
config.model.ema           = True
config.model.ema_rate      = 0.999
config.model.normalization = 'InstanceNorm++'
config.model.nonlinearity  = 'elu'
config.model.sigma_dist    = 'geometric'
config.model.num_classes   = 2311 # Number of train sigmas and 'N'
config.model.ngf           = 32

# Optimizer
config.optim.weight_decay  = 0.000 # No weight decay
config.optim.optimizer     = 'Adam'
config.optim.lr            = 0.0001
config.optim.beta1         = 0.9
config.optim.amsgrad       = False
config.optim.eps           = 0.001

# Training
config.training.batch_size     = 32
config.training.num_workers    = 4
config.training.n_epochs       = 400
config.training.anneal_power   = 2
config.training.log_all_sigmas = False

# Data
config.data.channel        = args.train
config.data.channels       = 2 # {Re, Im}
config.data.noise_std      = 0
config.data.image_size     = [16, 64] # [Nt, Nr] for the transposed channel
config.data.num_pilots     = config.data.image_size[1]
config.data.norm_channels  = 'global'
config.data.spacing_list   = [0.5] # Training and validation

# Seeds for train and test datasets
train_seed, val_seed = 1234, 4321

# Get datasets and loaders for channels
dataset     = Channels(train_seed, config, norm=config.data.norm_channels)
dataloader  = DataLoader(dataset, batch_size=config.training.batch_size, 
         shuffle=True, num_workers=config.training.num_workers, drop_last=True)

# Validation data
val_datasets, val_loaders, val_iters = [], [], []
for idx in range(len(config.data.spacing_list)):
    # Validation config
    val_config = copy.deepcopy(config)
    val_config.data.spacing_list = [config.data.spacing_list[idx]]
    # Create locals
    val_datasets.append(Channels(val_seed, val_config, norm=[dataset.mean, dataset.std]))
    val_loaders.append(DataLoader(
        val_datasets[-1], batch_size=len(val_datasets[-1]),
        shuffle=False, num_workers=0, drop_last=True))
    val_iters.append(iter(val_loaders[-1])) # For validation

# Construct pairwise distances
if False: # Set to true to follow [Song '20] exactly
    dist_matrix   = np.zeros((len(dataset), len(dataset)))
    flat_channels = dataset.channels.reshape((len(dataset), -1))
    for idx in tqdm(range(len(dataset))):
        dist_matrix[idx] = np.linalg.norm(
            flat_channels[idx][None, :] - flat_channels, axis=-1)
# Pre-determined values from 'Mixed' setting
config.model.sigma_begin = 39.15
config.model.sigma_rate  = 0.995
config.model.sigma_end   = config.model.sigma_begin * \
    config.model.sigma_rate ** (config.model.num_classes - 1)

# Choose the inference step size (epsilon) according to [Song '20]
candidate_steps = np.logspace(-13, -8, 1000)
step_criterion  = np.zeros((len(candidate_steps)))
gamma_rate      = 1 / config.model.sigma_rate
for idx, step in enumerate(candidate_steps):
    step_criterion[idx] = (1 - step / config.model.sigma_end ** 2) \
        ** (2 * config.model.num_classes) * (gamma_rate ** 2 -
            2 * step / (config.model.sigma_end ** 2 - config.model.sigma_end ** 2 * (
                1 - step / config.model.sigma_end ** 2) ** 2)) + \
            2 * step / (config.model.sigma_end ** 2 - config.model.sigma_end ** 2 * (
                1 - step / config.model.sigma_end ** 2) ** 2)
best_idx = np.argmin(np.abs(step_criterion - 1.))
config.model.step_size = candidate_steps[best_idx]

# Instantiate model
diffuser = NCSNv2Deepest(config)
diffuser = diffuser.cuda()

# Instantiate optimizer
optimizer = get_optimizer(config, diffuser.parameters())

# Instantiate counters and EMA helper
start_epoch, step = 0, 0
if config.model.ema:
    ema_helper = EMAHelper(mu=config.model.ema_rate)
    ema_helper.register(diffuser)

# Get all sigma values for the discretized VE-SDE
sigmas = get_sigmas(config)

# Sample fixed validation data
val_H_list = []
for idx in range(len(config.data.spacing_list)):
    val_sample = next(val_iters[idx])
    val_H_list.append(val_sample['H_herm'].cuda())

# Logging
config.log_path = './models/score/%s' % args.train
os.makedirs(config.log_path, exist_ok=True)
train_loss, val_loss  = [], []

# For each epoch
for epoch in tqdm(range(start_epoch, config.training.n_epochs)):
    # For each batch
    for i, sample in tqdm(enumerate(dataloader)):
        diffuser.train()
        step += 1
        # Move data to device
        for key in sample:
            sample[key] = sample[key].cuda()

        # Compute DSM loss using Hermitian channels
        loss = anneal_dsm_score_estimation(
            diffuser, sample['H_herm'], sigmas, None, 
            config.training.anneal_power)
        
        # Logging
        if step == 1:
            running_loss = loss.item()
        else:
            running_loss = 0.99 * running_loss + 0.01 * loss.item()
        train_loss.append(loss.item())
        
        # Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # EMA update
        if config.model.ema:
            ema_helper.update(diffuser)
            
        # Verbose
        if step % 100 == 0:
            if config.model.ema:
                val_score = ema_helper.ema_copy(diffuser)
            else:
                val_score = diffuser
            
            # For each validation setup
            local_val_losses = []
            for idx in range(len(config.data.spacing_list)):
                with torch.no_grad():
                    val_dsm_loss = \
                        anneal_dsm_score_estimation(
                            val_score, val_H_list[idx],
                            sigmas, None,
                            config.training.anneal_power)
                # Store
                local_val_losses.append(val_dsm_loss.item())
            # Sanity delete
            del val_score
            # Log
            val_loss.append(local_val_losses)
                
            # Print
            if len(local_val_losses) == 1:
                print('Epoch %d, Step %d, Train Loss (EMA) %.3f, \
Val. Loss %.3f' % (
                    epoch, step, running_loss, 
                    local_val_losses[0]))
            elif len(local_val_losses) >= 2:
                print('Epoch %d, Step %d, Train Loss (EMA) %.3f, \
Val. Loss (Split) %.3f %.3f' % (
                    epoch, step, running_loss, 
                    local_val_losses[0], local_val_losses[1]))
        
# Save final weights
torch.save({'model_state': diffuser.state_dict(),
            'optim_state': optimizer.state_dict(),
            'config': config,
            'train_loss': train_loss,
            'val_loss': val_loss}, 
   os.path.join(config.log_path, 'final_model.pt'))