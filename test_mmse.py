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
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--model', type=str, default='CDL-C')
parser.add_argument('--channel', type=str, default='CDL-C')
parser.add_argument('--start_point', type=str, default='Noise')
parser.add_argument('--spacing', nargs='+', type=float, default=[0.5])
parser.add_argument('--pilot_alpha', nargs='+', 
                    type=float, default=[0.6])
parser.add_argument('--steps_each', type=int, default=3)
parser.add_argument('--normalize_grad', type=bool, default=False)
parser.add_argument('--dc_boost', type=float, default=1)
parser.add_argument('--num_classes', type=int, default=2311)
args = parser.parse_args()

# Always !!!
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False
# Sometimes
torch.backends.cudnn.benchmark = True

# GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu);

# Target weights
deep_mimo_set = ['DeepMIMO_outdoor', 'DeepMIMO_indoor_1',
                 'DeepMIMO_indoor_3_nlos']
if args.model == 'CDL-D':
            target_weights = './models_oct14/\
numLambdas2_lambdaMin0.1_lambdaMax0.5_sigmaT39.1/final_model.pt'
elif args.model == 'CDL-C':
    target_weights = './models_jan29_2022_CDL-C/\
numLambdas1_lambdaMin0.5_lambdaMax0.5_sigmaT27.8/final_model.pt'
elif np.isin(args.model, ['CDL-B', 'CDL-A']):
    target_weights = './models_feb2_%s/\
numLambdas1_lambdaMin0.5_lambdaMax0.5_sigmaT31.2/final_model.pt' % (
args.model)
elif np.isin(args.model, deep_mimo_set):
    core_scenario = args.model[9:]
    target_weights = './models_feb2_DeepMIMO_%s/\
numLambdas1_lambdaMin0.5_lambdaMax0.5_sigmaT27.8/final_model.pt' % (
core_scenario)
elif args.model == 'all':
    target_weights = './models_feb23_multi/\
numLambdas1_lambdaMin0.5_lambdaMax0.5_sigmaT27.8/final_model.pt'

contents = torch.load(target_weights)
# Extract config
config = contents['config']
config.sampling.sigma = 0. # Nothing here
config.purpose = 'train'

# More sigmas
config.model.num_classes = args.num_classes
# config.model.sigma_begin = 2.
config.model.sigma_rate  = \
    (config.model.sigma_end / config.model.sigma_begin) ** \
        (1 / (config.model.num_classes - 1))

# Get a model
diffuser = NCSNv2Deepest(config)
diffuser = diffuser.cuda()
# !!! Load weights WITHOUT SIGMAS
status = diffuser.load_state_dict(contents['model_state'], strict=True)
diffuser.eval()

# Universal seeds
train_seed, val_seed = 1234, 4321
# Get training config
config.data.channel = args.model
config.data.array   = 'ULA'
dataset = Channels(train_seed, config, norm=config.data.norm_channels)

# Choose the core step size (epsilon)
config.sampling.steps_each   = args.steps_each
config.sampling.actual_steps = config.sampling.steps_each * \
    np.ones(config.model.num_classes, dtype=int)
# Total overall steps
total_steps = np.sum(config.sampling.actual_steps)
fixed_step_size = 1.

# Range of SNR, test channels and hyper-parameters
snr_range         = np.arange(-30, 17.5, 2.5) # np.arange(-10, 17.5, 2.5)
spacing_range     = np.asarray(args.spacing) # From a pre-defined index
pilot_alpha_range = np.asarray(args.pilot_alpha)
noise_range       = 10 ** (-snr_range / 10.)

# !!! Keep a fixed number of samples
kept_samples = 100
mmse_avg     = 50 # How many samples are we averaging per sample?

# Global results
oracle_log = np.zeros((len(spacing_range), len(pilot_alpha_range),
                       len(snr_range), total_steps, kept_samples, mmse_avg)) # Should match data
result_dir = 'TWC_rebuttal_MMSE_aug6_seed%d' % val_seed
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

# Wrap sparsity and spacings
meta_params = itertools.product(spacing_range, pilot_alpha_range)

# Saved channels - before MMSE averaging
saved_H = np.zeros((len(spacing_range), len(pilot_alpha_range),
                    len(snr_range), kept_samples, mmse_avg, 64, 16), 
                   dtype=np.complex64)
# Find best hyper-parameters for current model
hyper_file = 'our_hyperparams_%s.pt' % args.model
contents   = torch.load(hyper_file)
# Extract stuff
best_step  = contents['best_step_idx']
best_noise = contents['best_noise_idx']
best_stop  = contents['best_stop_idx']

# For each hyper-combo
for meta_idx, (spacing, pilot_alpha) in \
        tqdm(enumerate(meta_params)):
    # Unwrap indices
    spacing_idx, pilot_alpha_idx = np.unravel_index(
        meta_idx, (len(spacing_range), len(pilot_alpha_range)))
    
    # Get a validation dataset and adjust parameters
    val_config = copy.deepcopy(config)
    val_config.purpose           = 'val'
    val_config.data.channel      = args.channel
    val_config.data.spacing_list = [spacing]
    val_config.data.num_pilots   = int(np.floor(config.data.num_pilots * pilot_alpha))
    
    # Create locals
    if np.isin(args.model, deep_mimo_set):
        norm = [0., 1.]
    else:
        norm = [dataset.mean, dataset.std] # Use CDL-D values
        
    val_dataset = Channels(val_seed, val_config, norm=norm)
    val_loader  = DataLoader(val_dataset, batch_size=kept_samples,
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
    
    # Save oracle once
    oracle_H = val_H.cpu().numpy()
    # Save all samples at multiple SNR levels
    global_Y, global_P, global_H = [], [], []
    
    # For each SNR value
    for snr_idx, local_noise in tqdm(enumerate(noise_range)):
        # !!! Use exact hyper-parameters
        val_config.model.step_size = fixed_step_size * best_step[pilot_alpha_idx, snr_idx]
        noise_boost = best_noise[pilot_alpha_idx, snr_idx]
        target_stop = best_stop[pilot_alpha_idx, snr_idx]
         
        # Generate measurements ad-hoc
        local_Y   = torch.matmul(val_P, val_H)
        local_Y   = local_Y + \
            np.sqrt(local_noise) / np.sqrt(2.) * torch.randn_like(local_Y)
        
        # For each sample, add exact copies to collections
        for sample_idx in range(len(val_H)):
            global_Y.append(torch.tile(local_Y[sample_idx][None, ...].clone(),
                                       (mmse_avg, 1, 1)))
            global_P.append(torch.tile(val_P[sample_idx][None, ...].clone(),
                                       (mmse_avg, 1, 1)))
            global_H.append(torch.tile(val_H[sample_idx][None, ...].clone(),
                                       (mmse_avg, 1, 1)))
        
        # Tensorize
        global_Y = torch.cat(global_Y)
        global_P = torch.cat(global_P)
        global_H = torch.cat(global_H)
    
        # Initial point
        if args.start_point == 'Noise':
            current = torch.randn_like(global_H)
        elif args.start_point == 'Adjoint':
            current = torch.matmul(
                torch.conj(torch.transpose(global_P, -1, -2)), global_Y)
        elif args.start_point == 'LS':
            current = torch.linalg.lstsq(
                global_P.cpu(), global_Y.cpu(), driver='gelsd').cuda()
        
        y         = global_Y
        forward   = global_P
        forward_h = torch.conj(torch.transpose(global_P, -1, -2))
        oracle    = global_H
        
        # Stop the count!
        trailing_idx = 0
        mark_break   = False
        
        # Inference
        with torch.no_grad():
            for step_idx in tqdm(range(val_config.model.num_classes)):
                # Compute current step size and noise power
                current_sigma = diffuser.sigmas[step_idx].item()
                # Labels for diffusion model
                labels = torch.ones(global_H.shape[0]).cuda() * step_idx
                labels = labels.long()
                
                # For each step spent at that noise level
                for inner_idx in range(config.sampling.actual_steps[step_idx]):
                    # Compute score
                    current_real = torch.view_as_real(current).permute(0, 3, 1, 2)
                    # Get score
                    score = diffuser(current_real, labels)
                    # View as complex
                    score = \
                        torch.view_as_complex(score.permute(0, 2, 3, 1).contiguous())
                        
                    # Compute alpha
                    alpha = val_config.model.step_size * \
                        (current_sigma / val_config.model.sigma_end) ** 2
                    
                    # Compute gradient for measurements in un-normalized space
                    meas_term = torch.matmul(forward, current) - y
                    meas_grad = torch.matmul(forward_h, meas_term)
                    
                    # Annealing noise
                    grad_noise = np.sqrt(2 * alpha * noise_boost) * \
                        torch.randn_like(current)
                    
                    # Apply update
                    current = current + \
                        alpha * (score - args.dc_boost * meas_grad /\
                                 (local_noise/2. + current_sigma ** 2)) + grad_noise
                            
                    # Store loss
                    oracle_log[
                        spacing_idx, pilot_alpha_idx, snr_idx, trailing_idx] = \
                        np.reshape(
                            (torch.sum(torch.square(torch.abs(current - oracle)),
                                   dim=(-1, -2))/\
                        torch.sum(torch.square(torch.abs(oracle)),
                                  dim=(-1, -2))).view(len(snr_range), -1).cpu().numpy(),
                                (kept_samples, mmse_avg))
                            
                    # Early stop
                    if trailing_idx == target_stop:
                        # Full stop
                        mark_break = True
                        break
                    # Increment count
                    trailing_idx = trailing_idx + 1
               
                # Full stop
                if mark_break:
                    print('Early stopping at step %d!' % target_stop)
                    # Store channel estimates
                    saved_H[spacing_idx, pilot_alpha_idx, snr_idx] = \
                        copy.deepcopy(
                            torch.reshape(current, (kept_samples, mmse_avg, 64, 16)).cpu().numpy())
                    break
                
    # Delete validation dataset
    del val_dataset, val_loader
    torch.cuda.empty_cache()
    
# Save results to file based on noise
save_dict = {'spacing_range': spacing_range,
            'pilot_alpha_range': pilot_alpha_range,
            'args': args,
            'config': config,
            'snr_range': snr_range,
            'val_config': val_config,
            'oracle_log': oracle_log,
            'oracle_H': oracle_H,
            'saved_H': saved_H
            }
torch.save(save_dict,
           result_dir + '/model_%s_channel_%s.pt' % (
               args.model, args.channel))
