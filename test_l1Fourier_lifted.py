#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# We do not use GPUs for SigPy
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "";
# Set threading for multiple libraries to prevent resource hogging
num_threads = 1
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
from dotmap    import DotMap
from matplotlib import pyplot as plt

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str, default='CDL-C')
parser.add_argument('--test', type=str, default='CDL-C')
parser.add_argument('--antennas', nargs='+', type=int, default=[16, 64])
parser.add_argument('--array', type=str, default='ULA')
parser.add_argument('--spacing', type=float, default=0.5)
parser.add_argument('--alpha', nargs='+', type=float, default=[0.6])
parser.add_argument('--lmbda', nargs='+', type=float, default=[0.3])
parser.add_argument('--lifting', type=int, default=4)
parser.add_argument('--steps', type=int, default=1000)
parser.add_argument('--lr', nargs='+', type=float, default=[3e-3])
args = parser.parse_args()

# Create and populate minimal configuration
config = DotMap()
# For CS methods, train/test affects normalization
config.data.channel        = args.train
config.data.array          = args.array
config.data.image_size     = [args.antennas[0], args.antennas[1]]
config.data.num_pilots     = args.antennas[1]
config.data.spacing_list   = [args.spacing]
config.data.noise_std      = 1 # Dummy value
config.data.mixed_channels = False
# Seeds for train and test datasets
train_seed, val_seed = 1234, 4321
# Load training dataset
dataset = Channels(train_seed, config, norm='global')

# Range of SNR, test channels and hyper-parameters
snr_range          = np.asarray(np.arange(-10, 35, 5))
spacing_range      = np.asarray([args.spacing]) # Antenna spacing
alpha_range        = np.asarray(args.alpha) # Fraction of pilots
lmbda_range        = np.asarray(args.lmbda) # L1 regularization strength
lr_range           = np.asarray(args.lr) # Step size
lifting            = int(args.lifting)
# SNR is defined as Nr / noise_power 
# assuming average unit-power entries in MIMO channel matrix
noise_range        = 10 ** (-snr_range / 10.) * args.antennas[1]
gd_iter            = args.steps # Number of optimization steps

# Limit number of samples for faster results
kept_samples = 50

# Global results
nmse_log     = np.zeros((len(spacing_range), len(alpha_range),
                         len(lmbda_range), len(lr_range),
                         len(snr_range), kept_samples)) # Should match data
complete_log = np.zeros((len(spacing_range), len(alpha_range),
                         len(lmbda_range), len(lr_range),
                         len(snr_range), gd_iter, kept_samples)) 
result_dir = './results/l1CS_lifted%d/train-%s_test-%s' % (
    lifting, args.train, args.test)
os.makedirs(result_dir, exist_ok=True)
    
# Wrap sparsity, steps and spacings
meta_params = itertools.product(spacing_range, alpha_range,
                                lmbda_range, lr_range)

# For each hyper-combo
for meta_idx, (spacing, alpha, lmbda, lr) in tqdm(enumerate(meta_params)):
    # Unwrap indices
    spacing_idx, alpha_idx, lmbda_idx, lr_idx = np.unravel_index(
        meta_idx, (len(spacing_range), len(alpha_range),
                   len(lmbda_range), len(lr_range)))
    
    # Prepare validation configuration
    val_config = copy.deepcopy(config)
    val_config.data.channel      = args.test
    val_config.data.spacing_list = [spacing]
    val_config.data.num_pilots   = int(np.floor(args.antennas[1] * alpha))
    
    # Normalize test data using training data statistics
    val_dataset = Channels(val_seed, val_config, norm=[dataset.mean, dataset.std])
    val_loader  = DataLoader(val_dataset, batch_size=len(val_dataset),
        shuffle=False, num_workers=0, drop_last=True)
    val_iter = iter(val_loader)
        
    # Get all validation data explicitly
    val_sample = next(val_iter)
    del val_iter, val_loader # Free up memory
    val_P      = val_sample['P']
    val_P      = torch.conj(torch.transpose(val_P, -1, -2))
    val_H_herm = val_sample['H_herm']
    val_H      = val_H_herm[:, 0] + 1j * val_H_herm[:, 1]
    
    # Convert to numpy
    val_P = val_P.resolve_conj().numpy()
    val_H = val_H.resolve_conj().numpy()
    # Keep limited number of samples
    val_P = val_P[:kept_samples, ...]
    val_H = val_H[:kept_samples, ...]
    
    # Dictionary matrices for ULA/UPA array shapes
    left_dict  = np.conj(ifft(np.eye(val_H[0].shape[0]),
                              n=val_H[0].shape[0]*lifting, norm='ortho'))
    right_dict = ifft(np.eye(val_H[0].shape[1]), 
                      n=val_H[0].shape[1]*lifting, norm='ortho').T
    # Lifted shape
    lifted_shape = (val_H[0].shape[0]*lifting, val_H[0].shape[1]*lifting)
    
    # Proximal op for sigpy
    prox_op = sp.prox.L1Reg(lifted_shape, lmbda)
    
    # Run CS for each SNR value
    for snr_idx, local_noise in tqdm(enumerate(noise_range)):
        val_Y     = np.matmul(val_P, val_H)
        val_Y     = val_Y + np.sqrt(local_noise) / np.sqrt(2.) * \
                (np.random.normal(size=val_Y.shape) + \
                 1j * np.random.normal(size=val_Y.shape))
        
        # For each sample
        for sample_idx in tqdm(range(val_Y.shape[0])):
            # Create forward and regularization ops
            array_op = sp.linop.Compose(
                (sp.linop.MatMul((
                    lifted_shape[0], val_H[sample_idx].shape[1]), left_dict),
                sp.linop.RightMatMul(lifted_shape, right_dict)))
            fw_op  = sp.linop.Compose(
                (sp.linop.MatMul(val_H[sample_idx].shape, val_P[sample_idx]),
                 array_op))
            
            # Gradient function in closed form
            def gradf(x):
                return fw_op.H * (fw_op * x - val_Y[sample_idx])
            
            # Initial point and instantiate algorithm object
            val_H_hat = np.zeros(lifted_shape, complex)
            alg       = sp.alg.GradientMethod(
                gradf, val_H_hat, lr, proxg=prox_op, max_iter=gd_iter,
                accelerate=True)
            
            # For each optimization step
            for step_idx in range(gd_iter):
                # Run update step
                alg.update()
                
                # Convert current estimate from Fourier domain to spatial domain
                est_H = array_op(val_H_hat)
                
                # Log estimation errors
                complete_log[spacing_idx, alpha_idx, 
                             lmbda_idx, lr_idx, snr_idx, step_idx,
                            sample_idx] = \
                    (np.sum(np.square(
                        np.abs(est_H - val_H[sample_idx])), axis=(-1, -2)))/\
                     np.sum(np.square(
                         np.abs(val_H[sample_idx])), axis=(-1, -2))
            
            # Convert final estimate to IFFT
            est_H = array_op(val_H_hat)
            
            # Save NMSE for each sample
            nmse_log[spacing_idx, alpha_idx, 
                     lmbda_idx, lr_idx, snr_idx, sample_idx] = \
                (np.sum(np.square(
                    np.abs(est_H - val_H[sample_idx])), axis=(-1, -2)))/\
                 np.sum(np.square(
                     np.abs(val_H[sample_idx])), axis=(-1, -2))

# Use average estimation error to find best hyper-parameters
avg_nmse = np.mean(nmse_log, axis=-1)

best_nmse  = np.zeros((len(alpha_range), len(snr_range)))
best_lmbda = np.zeros((len(alpha_range), len(snr_range)))
best_lr    = np.zeros((len(alpha_range), len(snr_range)))
# For each alpha and SNR value
for alpha_idx, local_alpha in enumerate(alpha_range):
    for snr_idx, local_snr in enumerate(snr_range):
        local_nmse = avg_nmse[0, alpha_idx, ..., snr_idx]
        local_nmse = local_nmse.flatten()
        best_idx   = np.argmin(local_nmse)
        lmbda_idx, lr_idx = \
            np.unravel_index(best_idx, (len(lmbda_range), len(lr_range)))
        # Store and verbose
        best_nmse[alpha_idx, snr_idx]  = local_nmse[best_idx]
        best_lmbda[alpha_idx, snr_idx] = lmbda_range[lmbda_idx]
        best_lr[alpha_idx, snr_idx]    = lr_range[lr_idx]
        print('SNR = %.2f dB, NMSE = %.2f dB using lambda = %.1e and step size = %.1e' % (
            local_snr, 10*np.log10(best_nmse[alpha_idx, snr_idx]),
            best_lmbda[alpha_idx, snr_idx], best_lr[alpha_idx, snr_idx]))

# Plot results for all alpha values
plt.rcParams['font.size'] = 14
plt.figure(figsize=(10, 10))
for alpha_idx, local_alpha in enumerate(alpha_range):
    plt.plot(snr_range, 10*np.log10(best_nmse[alpha_idx]),
             linewidth=4, label='Alpha=%.2f' % local_alpha)
plt.grid(); plt.legend()
plt.title('Compressed Sensing fsAD, lifting = %d' % args.lifting)
plt.xlabel('SNR [dB]'); plt.ylabel('NMSE [dB]')
plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'results.png'), dpi=300, 
            bbox_inches='tight')
plt.close()
    
# Save full results to file
torch.save({'complete_log': complete_log,
            'nmse_log': nmse_log,
            'best_nmse': best_nmse,
            'best_lmbda': best_lmbda,
            'best_lr': best_lr,
            'snr_range': snr_range,
            'spacing_range': spacing_range,
            'alpha_range': alpha_range,
            'lmbda_range': lmbda_range,
            'lr_range': lr_range,
            'config': config, 'args': args
            }, os.path.join(result_dir, 'results.pt'))