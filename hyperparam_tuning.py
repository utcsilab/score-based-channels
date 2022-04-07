#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch, sys, os, itertools, copy, argparse
sys.path.append('./')

from tqdm import tqdm as tqdm
from ncsnv2.models.ncsnv2 import NCSNv2Deepest

from loaders          import Channels
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from dotmap import DotMap

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=2)
parser.add_argument('--spacing', nargs='+', type=float, default=[0.1])
parser.add_argument('--pilot_alpha', nargs='+', type=float, default=[0.2])
args = parser.parse_args()

# Always !!!
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False
# Sometimes
torch.backends.cudnn.benchmark = True

# GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "0";

dc_boost    = 1.
score_boost = 1.
noise_boost = 0.1
# device   = torch.device('cpu')
device   = torch.device('cuda:0')

# Target weights
# target_weights = './\
# models_oct12_VarScaling/sigmaT39.1/intermediate_model.pt'
target_weights = './models_oct14/\
numLambdas2_lambdaMin0.1_lambdaMax0.5_sigmaT39.1/final_model.pt'
contents = torch.load(target_weights, map_location=device)
# Extract config
config = contents['config']
config.sampling.sigma = 0. # Nothing here
config.device = device
# Get a model
diffuser = NCSNv2Deepest(config)
diffuser = diffuser.to(device)
# !!! Load weights
diffuser.load_state_dict(contents['model_state']) 
diffuser.eval()

# Universal seeds
train_seed, val_seed = 1234, 4321
# Get training config
# config.data.spacing_list = [0.1]
dataset = Channels(train_seed, config, norm=config.data.norm_channels)

# Choose the core step size (epsilon)
config.sampling.steps_each = 3
candidate_steps = np.logspace(-11, -7, 10000)
step_criterion  = np.zeros((len(candidate_steps)))
gamma_rate      = 1 / config.model.sigma_rate
for idx, step in enumerate(candidate_steps):
    sigma_squared   = config.model.sigma_end ** 2
    one_minus_ratio = (1 - step / sigma_squared) ** 2
    big_ratio       = 2 * step /\
        (sigma_squared - sigma_squared * one_minus_ratio)
    
    # Criterion
    step_criterion[idx] = one_minus_ratio ** config.sampling.steps_each * \
        (gamma_rate ** 2 - big_ratio) + big_ratio
    
best_idx        = np.argmin(np.abs(step_criterion - 1.))
fixed_step_size = candidate_steps[best_idx]

# Range of SNR, test channels and hyper-parameters
snr_range          = np.arange(-10, 17.5, 2.5)
step_factor_range  = np.asarray([0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.]) # Multiplicative
spacing_range      = np.asarray(args.spacing) # From a pre-defined index
pilot_alpha_range  = np.asarray(args.pilot_alpha)
noise_range        = 10 ** (-snr_range / 10.)
assert len(pilot_alpha_range) == 1, 'Too many pilot alphas for files!'

# Global results
oracle_log = np.zeros((len(spacing_range), len(pilot_alpha_range),
                       len(step_factor_range), len(snr_range),
                       int(config.model.num_classes * \
                       config.sampling.steps_each), 100)) # Should match data
result_dir = 'results_tuning'
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
    
# Wrap sparsity, steps and spacings
meta_params = itertools.product(spacing_range, pilot_alpha_range, step_factor_range)

# For each hyper-combo
for meta_idx, (spacing, pilot_alpha, step_factor) in tqdm(enumerate(meta_params)):
    # Unwrap indices
    spacing_idx, pilot_alpha_idx, step_factor_idx = np.unravel_index(
        meta_idx, (len(spacing_range), len(pilot_alpha_range), len(step_factor_range)))
    
    # Get a validation dataset and adjust parameters
    val_config = copy.deepcopy(config)
    val_config.data.spacing_list = [spacing]
    val_config.mode.step_size    = fixed_step_size * step_factor
    val_config.data.num_pilots   = int(np.floor(config.data.num_pilots * pilot_alpha))
    
    # Create locals
    val_dataset = Channels(val_seed, val_config, norm=[dataset.mean, dataset.std])
    val_loader  = DataLoader(val_dataset, batch_size=len(val_dataset),
        shuffle=False, num_workers=0, drop_last=True)
    val_iter = iter(val_loader) # For validation
        
    # Always the same initial points and data for validation
    val_sample = next(val_iter)
    _, val_P, _ = \
        val_sample['H'].to(device), val_sample['P'].to(device), \
            val_sample['Y'].to(device)
    # Transpose pilots
    val_P = torch.conj(torch.transpose(val_P, -1, -2))
    val_H_herm = val_sample['H_herm'].to(device)
    val_H      = val_H_herm[:, 0] + 1j * val_H_herm[:, 1]
    # Initial value and measurements
    init_val_H = torch.randn_like(val_H)
    
    # For each SNR value
    for snr_idx, local_noise in tqdm(enumerate(noise_range)):
        val_Y     = torch.matmul(val_P, val_H)
        val_Y     = val_Y + \
            np.sqrt(local_noise) / np.sqrt(2.) * torch.randn_like(val_Y) 
        current   = init_val_H.clone()
        y         = val_Y
        forward   = val_P
        forward_h = torch.conj(torch.transpose(val_P, -1, -2))
        norm      = [0., 1.]
        oracle    = val_H
        
        # Stop the count!
        trailing_idx = 0
        # For each SNR point
        with torch.no_grad():
            for step_idx in tqdm(range(val_config.model.num_classes)):
                # Compute current step size and noise power
                current_sigma = diffuser.sigmas[step_idx].item()
                # Labels for diffusion model
                labels = torch.ones(init_val_H.shape[0]).to(device) * step_idx
                labels = labels.long()
                
                # For each step spent at that noise level
                for inner_idx in range(val_config.sampling.steps_each):
                    # Compute score
                    current_real = torch.view_as_real(current).permute(0, 3, 1, 2)
                    # Get score
                    score = diffuser(current_real, labels)
                    # View as complex
                    score = \
                        torch.view_as_complex(score.permute(0, 2, 3, 1).contiguous())
                        
                    # Get un-normalized version for measurements
                    current_unnorm = norm[1] * current
                    # Compute alpha
                    alpha = val_config.model.step_size * \
                        (current_sigma / val_config.model.sigma_end) ** 2
                    
                    # Compute gradient for measurements in un-normalized space
                    meas_grad = torch.matmul(forward_h, 
                                 torch.matmul(forward, current_unnorm) - y)
                    # Re-normalize gradient to match score model
                    meas_grad = meas_grad / norm[1]
                    
                    # Annealing noise
                    grad_noise = np.sqrt(2 * alpha * noise_boost) * torch.randn_like(current) 
                    
                    # Apply update
                    current = current + \
                        score_boost * alpha * score - \
                            dc_boost * alpha / current_sigma ** 2 * \
                        meas_grad + grad_noise
                    
                    # Store loss
                    oracle_log[
                        spacing_idx, pilot_alpha_idx, step_factor_idx, snr_idx, trailing_idx] = \
                        (torch.sum(torch.square(torch.abs(current - oracle)), dim=(-1, -2))/\
                        torch.sum(torch.square(torch.abs(oracle)), dim=(-1, -2))).cpu().numpy()
                            
                    # Increment count
                    trailing_idx = trailing_idx + 1
                    
    # Delete validation dataset
    del val_dataset, val_loader
    # torch.cuda.empty_cache()

# Squeeze
oracle_log = np.squeeze(oracle_log)
# Plot at best stopping point and step factor value
plt.figure(); plt.plot(10*np.log10(np.min(np.mean(np.squeeze(oracle_log), axis=-1), axis=(-1, -3)))); plt.show()
