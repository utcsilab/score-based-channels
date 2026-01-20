#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np

from matplotlib import pyplot as plt

## Blind SNR
# Target channel
target_channel = 'CDL-C'
our_stop_array = [1104, 1161, 1260] # Best 'N' in the paper
our_beta       = 3e-4 # Best 'beta' in the paper
l1_stop_array  = [291, 190, 248]
alpha_array    = [1.0, 0.8, 0.6]

# Plot
plt.rcParams['font.size'] = 22
linewidth  = 3.5
markersize = 12
colors     = ['r', 'g', 'b']
markers    = ['*', 'o', 's']
plt.figure(figsize=(24, 10))

plt.subplot(1, 2, 1)
# Plot a curve for each sparsity value 'alpha'
for target_alpha in range(3):
    # Our results
    our_dir = 'results_ours_saved_channels_seed9999'
    our_file = our_dir + '/%s_noise%.1e.pt' % (target_channel, our_beta)
    contents = torch.load(our_file)
    snr_range = contents['snr_range']
    our_log = contents['oracle_log'].squeeze()
    # Downselect alpha, timestep and average
    our_results = \
        np.mean(our_log[target_alpha, :, our_stop_array[target_alpha]], axis=-1)
    our_results = 10 * np.log10(our_results)
        
    # L1 results
    l1_dir   = 'results_l1_saved_channels_seed9999'
    l1_file  = l1_dir + '/%s_spacing0.50/l1_results_lmbda3.00e-02.pt' % (
        target_channel)
    contents = torch.load(l1_file)
    l1_log = contents['oracle_log'].squeeze()
    # Downselect alpha, timestep and average
    l1_results = \
        np.mean(l1_log[target_alpha, :, l1_stop_array[target_alpha]-1], axis=-1)
    l1_results = 10 * np.log10(l1_results)
        
    # Plot
    plt.plot(snr_range, our_results, linewidth=linewidth,
             linestyle='solid', marker=markers[target_alpha],
             color=colors[target_alpha], 
             label=r'Ours, CDL-C $\alpha=$%.1f' % alpha_array[target_alpha],
             markersize=markersize)
    
    plt.plot(snr_range, l1_results, linewidth=linewidth,
             linestyle='dotted', marker=markers[target_alpha],
             color=colors[target_alpha], 
             label=r'Lasso, CDL-C $\alpha=$%.1f' % alpha_array[target_alpha],
             markersize=markersize)
# Plot
plt.grid()
plt.xlabel('SNR [dB]')
plt.ylabel('NMSE [dB]')
plt.title('Blind (Unknown SNR)')
plt.legend()
plt.ylim([-25, 6])
# Save plot
plt.savefig('CDL_C_SNRblind_results.png', bbox_inches = 'tight',
    pad_inches = 0.05, dpi=300)

## Known SNR
# Target channel
target_channel = 'CDL-C'
alpha_array    = [1.0, 0.8, 0.6]
step_snr_match = [
    [ 525,  687,  816, 1000, 1137, 1435, 1623, 1765, 1938, 2141, 2270], # Best 'N' from Alpha = 1.0
    [ 523,  612,  790,  995, 1122, 1437, 1538, 1843, 2028, 2246, 2437], # Best 'N' from Alpha = 0.8, CDL-D
    [ 429,  583,  864,  885, 1213, 1353, 1541, 1652, 1870, 2216, 2328], # Best 'N' from Alpha = 0.6, CDL-D
    ]
our_beta       = 1e-3 # Best 'beta' in the paper, using same for all SNR points

# Plot
plt.rcParams['font.size'] = 22
linewidth  = 3.5
markersize = 12
colors     = ['r', 'g', 'b']
markers    = ['*', 'o', 's']
# plt.figure(figsize=(12, 10))

plt.subplot(1, 2, 2)
for target_alpha in range(3):
    # Our results
    our_dir = 'results_ours_saved_channels_SNR_known_seed9999'
    our_file = our_dir + '/%s_noise%.1e.pt' % (target_channel, our_beta)
    contents = torch.load(our_file)
    snr_range = contents['snr_range']
    our_log = contents['oracle_log'].squeeze()
    # Downselect alpha, timestep and average
    our_square_log = our_log[target_alpha, :, step_snr_match[target_alpha]]
    # Downselect diagonals
    our_diag_log = np.asarray([our_square_log[idx, idx] for idx in range(len(snr_range))])
    our_results  = np.mean(our_diag_log, axis=-1)
    our_results  = 10 * np.log10(our_results)
    
    # L1 results
    l1_dir   = 'results_l1_saved_channels_seed9999'
    l1_file  = l1_dir + '/%s_spacing0.50/l1_results_lmbda3.00e-02.pt' % (
        target_channel)
    contents = torch.load(l1_file)
    l1_log = contents['oracle_log'].squeeze()
    # Pick best results here
    avg_l1_log = np.mean(l1_log[target_alpha], axis=-1)
    # Downselect alpha, timestep and average
    l1_results = 10 * np.log10(np.min(avg_l1_log[
        :, np.arange(l1_stop_array[target_alpha]-1)], axis=-1))
    
    # Plot
    plt.plot(snr_range, our_results, linewidth=linewidth,
             linestyle='solid', marker=markers[target_alpha],
             color=colors[target_alpha], 
             label=r'Ours, CDL-C $\alpha=$%.1f' % alpha_array[target_alpha],
             markersize=markersize)
    
    plt.plot(snr_range, l1_results, linewidth=linewidth,
             linestyle='dotted', marker=markers[target_alpha],
             color=colors[target_alpha], 
             label=r'Lasso, CDL-C $\alpha=$%.1f' % alpha_array[target_alpha],
             markersize=markersize)
    
# Plot
plt.grid()
plt.xlabel('SNR [dB]')
plt.ylabel('NMSE [dB]')
plt.title('Known SNR')
plt.legend()
plt.ylim([-25, 6])
plt.subplots_adjust(wspace=0.36)
# Save-sausage
plt.savefig('CDL_C_SNRknown_results.png', bbox_inches = 'tight',
    pad_inches = 0.05, dpi=300)
