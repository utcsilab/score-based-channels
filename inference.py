#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch, sys, os, itertools, copy, argparse
sys.path.append('./')

from tqdm import tqdm as tqdm
from ncsnv2.models.ncsnv2 import NCSNv2Deepest

from loaders          import Channels
from torch.utils.data import DataLoader

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--channel', type=str, default='CDL-D')
parser.add_argument('--save_channels', type=int, default=0)
parser.add_argument('--spacing', nargs='+', type=float, default=[0.5])
parser.add_argument('--pilot_alpha', nargs='+', type=float, default=[0.6])
parser.add_argument('--noise_boost', type=float, default=0.001)
args = parser.parse_args()

# Always !!!
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False
# Sometimes
torch.backends.cudnn.benchmark = True

# GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu);

# Target weights - replace with target model
target_weights = './models/\
numLambdas2_lambdaMin0.1_lambdaMax0.5_sigmaT39.1/final_model.pt'
contents = torch.load(target_weights)
# Extract config
config = contents['config']
config.sampling.sigma = 0. # Nothing here

# !!! 'Beta' in paper
noise_boost = args.noise_boost

# Get a model
diffuser = NCSNv2Deepest(config)
diffuser = diffuser.cuda()
# !!! Load weights
diffuser.load_state_dict(contents['model_state']) 
diffuser.eval()

# Universal seeds
train_seed, val_seed = 1234, 9999
# Get training config
config.data.channel = 'CDL-D'
dataset             = \
    Channels(train_seed, config, norm=config.data.norm_channels)

# Choose the core step size (epsilon) according to [Song '20]
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
snr_range          = np.arange(-10, 17.5, 2.5) # np.arange(-10, 17.5, 2.5)
step_factor_range  = np.asarray([1.]) # Multiplicative
spacing_range      = np.asarray(args.spacing) # From a pre-defined index
pilot_alpha_range  = np.asarray(args.pilot_alpha)
noise_range        = 10 ** (-snr_range / 10.)

# Save test results
if args.save_channels:
    num_channels = 200 # !!! More
    alpha_match = [0.6, 0.8, 1.0]
    step_match  = [1260, 1161, 1104]
    
    # For noise = 0.001 !!!
    step_snr_match = [
        [ 429,  583,  864,  885, 1213, 1353, 1541, 1652, 1870, 2216, 2328], # Alpha = 0.6, CDL-D
        [ 523,  612,  790,  995, 1122, 1437, 1538, 1843, 2028, 2246, 2437], # Alpha = 0.8, CDL-D
        [ 525,  687,  816, 1000, 1137, 1435, 1623, 1765, 1938, 2141, 2270], # Alpha = 1.0
        ]
    
    print('Saving test results!')
    saved_H = np.zeros((len(pilot_alpha_range), 
                        len(snr_range), num_channels, 64, 16),
                       dtype=np.complex64)
else:
    num_channels = 100 # Validation
# Global results
oracle_log = np.zeros((len(spacing_range), len(pilot_alpha_range),
                       len(step_factor_range), len(snr_range),
                       int(config.model.num_classes * \
                       config.sampling.steps_each), num_channels)) # Should match data
result_dir = 'results_seed%d' % val_seed
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

# Wrap sparsity, steps and spacings
meta_params = itertools.product(spacing_range, pilot_alpha_range, step_factor_range)

# For each hyper-combo
for meta_idx, (spacing, pilot_alpha, step_factor) in tqdm(enumerate(meta_params)):
    # Unwrap indices
    spacing_idx, pilot_alpha_idx, step_factor_idx = np.unravel_index(
        meta_idx, (len(spacing_range), len(pilot_alpha_range),
                   len(step_factor_range)))
    
    # Get a validation dataset and adjust parameters
    val_config = copy.deepcopy(config)
    val_config.data.channel      = args.channel
    val_config.data.spacing_list = [spacing]
    val_config.mode.step_size    = fixed_step_size * step_factor
    val_config.data.num_pilots   = int(np.floor(config.data.num_pilots * pilot_alpha))
    
    # Create locals
    val_dataset = Channels(val_seed, val_config, norm=[dataset.mean, dataset.std])
    val_loader  = DataLoader(val_dataset, batch_size=len(val_dataset),
        shuffle=False, num_workers=0, drop_last=True)
    val_iter = iter(val_loader) # For validation
    print('There are %d validation channels!' % len(val_dataset))
        
    # Always the same initial points and data for validation
    val_sample = next(val_iter)
    _, val_P, _ = \
        val_sample['H'].cuda(), val_sample['P'].cuda(), val_sample['Y'].cuda()
    # Transpose pilots
    val_P = torch.conj(torch.transpose(val_P, -1, -2))
    val_H_herm = val_sample['H_herm'].cuda()
    val_H = val_H_herm[:, 0] + 1j * val_H_herm[:, 1]
    # Initial value and measurements
    init_val_H = torch.randn_like(val_H)
    
    # Save oracle once
    if args.save_channels:
        oracle_H = val_H.cpu().numpy()
    
    # For each SNR value
    for snr_idx, local_noise in tqdm(enumerate(noise_range)):
        if args.save_channels:
            # Find exact stopping point
            target_stop = step_snr_match[int(np.where(pilot_alpha == alpha_match)[0])][snr_idx]
            print('For this SNR, stopping at %d!' % target_stop)
        
        val_Y     = torch.matmul(val_P, val_H)
        val_Y     = val_Y + \
            np.sqrt(local_noise) * torch.randn_like(val_Y) 
        current   = init_val_H.clone()
        y         = val_Y
        forward   = val_P
        forward_h = torch.conj(torch.transpose(val_P, -1, -2))
        norm      = [0., 1.]
        oracle    = val_H
        
        # Stop the count!
        trailing_idx = 0
        mark_break   = False
        # For each SNR point
        with torch.no_grad():
            for step_idx in tqdm(range(val_config.model.num_classes)):
                # Compute current step size and noise power
                current_sigma = diffuser.sigmas[step_idx].item()
                # Labels for diffusion model
                labels = torch.ones(init_val_H.shape[0]).cuda() * step_idx
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
                    grad_noise = np.sqrt(2 * alpha * noise_boost) * \
                        torch.randn_like(current) 
                    
                    # Apply update
                    current = current + \
                        alpha * (score - meas_grad /\
                                 (local_noise/2. + current_sigma ** 2)) + grad_noise
                            
                    # Store loss
                    oracle_log[
                        spacing_idx, pilot_alpha_idx, step_factor_idx, snr_idx, trailing_idx] = \
                        (torch.sum(torch.square(torch.abs(current - oracle)), dim=(-1, -2))/\
                        torch.sum(torch.square(torch.abs(oracle)), dim=(-1, -2))).cpu().numpy()
                            
                    # Decide to early stop if saving
                    if args.save_channels:
                        if trailing_idx == target_stop:
                            saved_H[pilot_alpha_idx, snr_idx] = \
                                current.cpu().numpy()
                            # Full stop
                            mark_break = True
                            break
                            
                    # Increment count
                    trailing_idx = trailing_idx + 1
               
                # Full stop
                if args.save_channels and mark_break:
                    print('Early stopping at step %d!' % target_stop)
                    break
            
    # Delete validation dataset
    del val_dataset, val_loader
    torch.cuda.empty_cache()

# Save results to file based on noise
save_dict = {'spacing_range': spacing_range,
            'pilot_alpha_range': pilot_alpha_range,
            'step_factor_range': step_factor_range,
            'snr_range': snr_range,
            'val_config': val_config,
            'oracle_log': oracle_log
            }
if args.save_channels:
    save_dict['saved_H']  = saved_H
    save_dict['oracle_H'] = oracle_H
torch.save(save_dict,
           result_dir + '/%s_noise%.1e.pt' % (args.channel, noise_boost))
