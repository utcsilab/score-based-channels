#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 14:32:31 2021

@author: marius
"""

import os
# GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "";
# All the threading in the world
num_threads = 2
os.environ["OMP_NUM_THREADS"]        = str(num_threads)
os.environ["OMP_DYNAMIC"]            = "false"
os.environ["OPENBLAS_NUM_THREADS"]   = str(num_threads)
os.environ["MKL_NUM_THREADS"]        = str(num_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
os.environ["NUMEXPR_NUM_THREADS"]    = str(num_threads)

import numpy as np
import sigpy as sp
import torch, sys, itertools, copy, argparse
sys.path.append('./')

from tqdm import tqdm as tqdm
from loaders          import Channels
from torch.utils.data import DataLoader
torch.set_num_threads(num_threads)

from scipy.fft import ifft
from matplotlib import pyplot as plt

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='CDL-C')
parser.add_argument('--channel', type=str, default='CDL-C')
parser.add_argument('--antennas', nargs='+', type=int, default=[64, 256])
parser.add_argument('--array', type=str, default='UPA')
parser.add_argument('--spacing', nargs='+', type=float, default=[0.25])
parser.add_argument('--alpha', nargs='+', type=float,
                    default=[0.0625, 0.125, 0.1875,
                             0.25, 0.375, 0.5, 0.6, 0.8, 1.])
parser.add_argument('--lmbda', nargs='+', type=float,
                    default=[0.1, 0.3, 0.6, 1., 3.])
parser.add_argument('--lifting', type=int, default=4)
parser.add_argument('--lr', nargs='+', type=float,
                    default=[3e-3])
args = parser.parse_args()

# Always !!!
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False

# Target weights
target_weights = './models_jan29_2022_CDL-C/\
numLambdas1_lambdaMin0.5_lambdaMax0.5_sigmaT27.8/final_model.pt'
contents = torch.load(target_weights, map_location=torch.device('cpu'))
# Extract config
config = contents['config']
config.sampling.sigma = 0. # Nothing here
config.data.channel   = args.model # Always for training!
# More stuff
config.data.array        = args.array
config.data.image_size   = [args.antennas[0], args.antennas[1]]
config.data.spacing_list = [args.spacing[0]]
# Universal seeds
train_seed, val_seed = 1234, 4321
# Get training config
dataset = Channels(train_seed, config, norm=config.data.norm_channels)

# Range of SNR, test channels and hyper-parameters
snr_range          = np.asarray(np.arange(-10, 17.5, 2.5))
spacing_range      = np.asarray(args.spacing) # From a pre-defined index
alpha_range        = np.asarray(args.alpha)
lmbda_range        = np.asarray(args.lmbda) # L1-reg
lr_range           = np.asarray(args.lr) # Step size
lifting            = int(args.lifting)
noise_range        = 10 ** (-snr_range / 10.)
gd_iter            = 1000 # Steps

# Limited number of samples
kept_samples = 50

# Global results
oracle_log = np.zeros((len(spacing_range), len(alpha_range),
                       len(lmbda_range), len(lr_range),
                       len(snr_range), kept_samples)) # Should match data
complete_log = np.zeros((len(spacing_range), len(alpha_range),
                       len(lmbda_range), len(lr_range),
                       len(snr_range), gd_iter, kept_samples)) 
result_dir = 'results_l1_baseline_lifted%d/model_%s_channel_%s' % (
    lifting, args.model, args.channel)
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
    
# Wrap sparsity, steps and spacings
meta_params = itertools.product(spacing_range, alpha_range,
                                lmbda_range, lr_range)

# For each hyper-combo
for meta_idx, (spacing, alpha, lmbda, lr) in tqdm(enumerate(meta_params)):
    # Unwrap indices
    spacing_idx, alpha_idx, lmbda_idx, lr_idx = np.unravel_index(
        meta_idx, (len(spacing_range), len(alpha_range),
                   len(lmbda_range), len(lr_range)))
    
    # Get a validation dataset and adjust parameters
    val_config = copy.deepcopy(config)
    # !!! Replace channel
    val_config.purpose      = 'val'
    val_config.data.channel = args.channel
    val_config.data.spacing_list = [spacing]
    val_config.data.num_pilots   = \
        int(np.floor(args.antennas[1] * alpha))
    
    # Create locals
    # !!! Normalization
    val_dataset = Channels(val_seed, val_config,
                           norm=[dataset.mean, dataset.std])
    val_loader  = DataLoader(val_dataset, batch_size=len(val_dataset),
        shuffle=False, num_workers=0, drop_last=True)
    val_iter = iter(val_loader) # For validation
        
    # Always the same initial points and data for validation
    val_sample = next(val_iter)
    del val_iter, val_loader
    val_P = val_sample['P']
    # Transpose pilots
    val_P = torch.conj(torch.transpose(val_P, -1, -2))
    val_H_herm = val_sample['H_herm']
    val_H = val_H_herm[:, 0] + 1j * val_H_herm[:, 1]
    
    # De-tensorize
    val_P = val_P.resolve_conj().numpy()
    val_H = val_H.resolve_conj().numpy()
    
    # Keep only relevant samples
    val_P = val_P[:kept_samples, ...]
    val_H = val_H[:kept_samples, ...]
    
    # Dictionary matrices for ULA
    left_dict  = np.conj(ifft(np.eye(val_H[0].shape[0]),
                              n=val_H[0].shape[0]*lifting, norm='ortho'))
    right_dict = ifft(np.eye(val_H[0].shape[1]), 
                      n=val_H[0].shape[1]*lifting, norm='ortho').T
    # Lifted shape
    lifted_shape = (val_H[0].shape[0]*lifting, val_H[0].shape[1]*lifting)
    
    # Proximal op
    prox_op = sp.prox.L1Reg(lifted_shape, lmbda) # 
    
    # For each SNR value
    for snr_idx, local_noise in tqdm(enumerate(noise_range)):
        val_Y     = np.matmul(val_P, val_H)
        val_Y     = val_Y + \
            np.sqrt(local_noise) / np.sqrt(2.) * \
                (np.random.normal(size=val_Y.shape) + \
                 1j * np.random.normal(size=val_Y.shape))
        
        # !!! For each sample
        for sample_idx in tqdm(range(val_Y.shape[0])):
            # Create forward and regularization ops
            array_op = sp.linop.Compose(
                (sp.linop.MatMul((
                    lifted_shape[0], val_H[sample_idx].shape[1]), left_dict),
                sp.linop.RightMatMul(lifted_shape, right_dict)))
            # Big one
            fw_op  = sp.linop.Compose(
                (sp.linop.MatMul(val_H[sample_idx].shape, val_P[sample_idx]),
                 array_op))
            
            def gradf(x):
                return fw_op.H * (fw_op * x - val_Y[sample_idx])
            
            val_H_hat = np.zeros(lifted_shape, complex)
            alg       = sp.alg.GradientMethod(
                gradf, val_H_hat, lr, proxg=prox_op, max_iter=gd_iter,
                accelerate=True)
            
            for step_idx in range(gd_iter):
                alg.update()
                # Convert estimate to IFFT
                est_H = array_op(val_H_hat)
                
                # Log exact errors
                complete_log[spacing_idx, alpha_idx, 
                             lmbda_idx, lr_idx, snr_idx, step_idx,
                            sample_idx] = \
                    (np.sum(np.square(
                        np.abs(est_H - val_H[sample_idx])), axis=(-1, -2)))/\
                     np.sum(np.square(
                         np.abs(val_H[sample_idx])), axis=(-1, -2))
            
            # Convert final estimate to IFFT
            est_H = array_op(val_H_hat)
            
            # Estimate error
            oracle_log[spacing_idx, alpha_idx, 
                       lmbda_idx, lr_idx, snr_idx, sample_idx] = \
                (np.sum(np.square(
                    np.abs(est_H - val_H[sample_idx])), axis=(-1, -2)))/\
                 np.sum(np.square(
                     np.abs(val_H[sample_idx])), axis=(-1, -2))
                
# Save to file
torch.save({'complete_log': complete_log,
            'snr_range': snr_range,
            'spacing_range': spacing_range,
            'alpha_range': alpha_range,
            'lmbda_range': lmbda_range,
            'lr_range': lr_range
            }, result_dir + \
            '/l1_results_Nt%d_Nr%d_fineAlpha.pt' % (
                args.antennas[1], args.antennas[0]))

# Plot
plt.figure(); 
plt.plot(10*np.log10(np.mean(np.squeeze(complete_log), axis=-1)).T);
plt.grid(); plt.show()