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

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--channel', type=str, default='CDL-C')
parser.add_argument('--spacing', type=float, default=0.5)
parser.add_argument('--alpha_step_range', nargs='+', type=float,
                    default=[3e-11, 6e-11, 1e-10, 3e-10])
parser.add_argument('--beta_noise_range', nargs='+', type=float,
                    default=[0.1, 0.01, 0.001])
parser.add_argument('--pilot_alpha', type=float, default=0.6)
args = parser.parse_args()

# Disable TF32 due to potential precision issues
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False
torch.backends.cudnn.benchmark        = True
# GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# Target file
target_dir  = './models/score/%s' % args.channel
target_file = os.path.join(target_dir, 'final_model.pt')
contents    = torch.load(target_file)
config      = contents['config']

# Instantiate model
diffuser = NCSNv2Deepest(config)
diffuser = diffuser.cuda()
# Load weights
diffuser.load_state_dict(contents['model_state']) 
diffuser.eval()

# Train and validation seeds
train_seed, val_seed = 1234, 4321
# Get training dataset for normalization
config.data.channel = args.channel
dataset = Channels(train_seed, config, norm=config.data.norm_channels)

# Range of SNR, test channels and hyper-parameters
snr_range          = np.arange(-10, 32.5, 2.5)
alpha_step_range   = np.asarray(args.alpha_step_range)
beta_noise_range   = np.asarray(args.beta_noise_range)
noise_range        = 10 ** (-snr_range / 10.) * config.data.image_size[1]

# Global results
nmse_log = np.zeros((len(alpha_step_range), len(beta_noise_range), len(snr_range),
                     int(config.model.num_classes * config.sampling.steps_each),
                     100)) # Should match data
result_dir = './results/score'
os.makedirs(result_dir, exist_ok=True)
    
# Wrap hyper-parameters
meta_params = itertools.product(alpha_step_range, beta_noise_range)

# For each hyper-combo
for meta_idx, (alpha_step, beta_noise) in tqdm(enumerate(meta_params)):
    # Unwrap indices
    alpha_idx, beta_idx = np.unravel_index(
        meta_idx, (len(alpha_step_range), len(beta_noise_range)))
    
    # Get a validation dataset and adjust parameters
    val_config = copy.deepcopy(config)
    val_config.data.channel      = args.channel
    val_config.data.spacing_list = [args.spacing]
    val_config.data.num_pilots   = int(np.floor(
        config.data.image_size[1] * args.pilot_alpha))
    val_dataset = Channels(val_seed, val_config, norm=[dataset.mean, dataset.std])
    val_loader  = DataLoader(val_dataset, batch_size=len(val_dataset),
        shuffle=False, num_workers=0, drop_last=True)
    val_iter = iter(val_loader)
    print('There are %d validation channels' % len(val_dataset))
        
    # Get all validation data explicitly
    val_sample = next(val_iter)
    _, val_P, _ = \
        val_sample['H'].cuda(), val_sample['P'].cuda(), val_sample['Y'].cuda()
    # Transposed pilots
    val_P = torch.conj(torch.transpose(val_P, -1, -2))
    val_H_herm = val_sample['H_herm'].cuda()
    val_H = val_H_herm[:, 0] + 1j * val_H_herm[:, 1]
    # Initial estimates
    init_val_H = torch.randn_like(val_H)
    
    # For each SNR value
    for snr_idx, local_noise in tqdm(enumerate(noise_range)):
        val_Y     = torch.matmul(val_P, val_H)
        val_Y     = val_Y + \
            np.sqrt(local_noise) * torch.randn_like(val_Y) 
        current   = init_val_H.clone()
        y         = val_Y
        forward   = val_P
        forward_h = torch.conj(torch.transpose(val_P, -1, -2))
        oracle    = val_H # Ground truth channels
        # Count every step
        trailing_idx = 0
        
        for step_idx in tqdm(range(val_config.model.num_classes)):
            # Compute current step size and noise power
            current_sigma = diffuser.sigmas[step_idx].item()
            # Labels for diffusion model
            labels = torch.ones(init_val_H.shape[0]).cuda() * step_idx
            labels = labels.long()
            
            # Compute annealed step size
            alpha = alpha_step * \
                (current_sigma / val_config.model.sigma_end) ** 2
            
            # For each step spent at that noise level
            for inner_idx in range(val_config.sampling.steps_each):
                # Compute score using real view of data
                current_real = torch.view_as_real(current).permute(0, 3, 1, 2)
                with torch.no_grad():
                    score = diffuser(current_real, labels)
                # View as complex
                score = \
                    torch.view_as_complex(score.permute(0, 2, 3, 1).contiguous())
                
                # Compute gradient for measurements in un-normalized space
                meas_grad = torch.matmul(forward_h, 
                             torch.matmul(forward, current) - y)
                # Sample noise
                grad_noise = np.sqrt(2 * alpha * beta_noise) * \
                    torch.randn_like(current) 
                
                # Apply update
                current = current + alpha * (score - meas_grad /\
                             (local_noise/2. + current_sigma ** 2)) + grad_noise
                        
                # Store loss
                nmse_log[alpha_idx, beta_idx, snr_idx, trailing_idx] = \
                    (torch.sum(torch.square(torch.abs(current - oracle)), dim=(-1, -2))/\
                    torch.sum(torch.square(torch.abs(oracle)), dim=(-1, -2))).cpu().numpy()
                trailing_idx = trailing_idx + 1

# Average estimation error and best stopping point
avg_nmse  = np.mean(nmse_log, axis=-1)
best_nmse = np.min(avg_nmse, axis=-1)

# Find best hyper-parameters for each SNR point
best_alpha_snr, best_beta_snr = [], []
for snr_idx in range(len(snr_range)):
    local_nmse = best_nmse[..., snr_idx].flatten()
    best_idx   = np.argmin(local_nmse)
    best_alpha_idx, best_beta_idx = np.unravel_index(
        best_idx, (len(alpha_step_range), len(beta_noise_range)))
    best_alpha_snr.append(alpha_step_range[best_alpha_idx])
    best_beta_snr.append(beta_noise_range[best_beta_idx])
    
# Plot all curves
plt.rcParams['font.size'] = 14
plt.figure(figsize=(10, 10))
for alpha_idx, local_alpha in enumerate(alpha_step_range):
    for beta_idx, local_beta in enumerate(beta_noise_range):
        plt.plot(snr_range, 10*np.log10(best_nmse[alpha_idx, beta_idx]),
                 linewidth=4, label='Alpha=%.2e, Beta=%.2e' % (local_alpha, local_beta))
plt.grid(); plt.legend()
plt.title('Score-based hyperparameter search')
plt.xlabel('SNR [dB]'); plt.ylabel('NMSE [dB]')
plt.tight_layout()
plt.savefig(os.path.join(result_dir, '%s-hyperparameters.png' % args.channel), dpi=300, 
            bbox_inches='tight')
plt.close()

# Save full results to file
torch.save({'nmse_log': nmse_log,
            'avg_nmse': avg_nmse,
            'best_nmse': best_nmse,
            'best_alpha_snr': best_alpha_snr,
            'best_beta_snr': best_beta_snr,
            'snr_range': snr_range,
            'alpha_step_range': alpha_step_range,
            'beta_noise_range': beta_noise_range,
            'config': config, 'args': args
            }, os.path.join(result_dir, '%s-hyperparameters.pt' % args.channel))