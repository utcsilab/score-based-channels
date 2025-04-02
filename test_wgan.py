#!/usr/bin/env python3

import torch, random, itertools
import os, argparse, copy

import numpy as np
from loaders import Channels
from tqdm import tqdm

import aux_gan as dcgan
from torch.utils.data import DataLoader

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu',     type=int, default=1)
parser.add_argument('--mode',    type=str, default='single')
parser.add_argument('--model',   type=str, default='CDL-C')
parser.add_argument('--channel', type=str, default='CDL-C')
parser.add_argument('--spacing', type=float, default=0.5)
parser.add_argument('--l2lam_range', nargs='+', type=float,
                    default=[1e-1, 3e-1, 1., 3.])
parser.add_argument('--lr_range', nargs='+', type=float,
                    default=[0.03, 0.01, 0.003, 0.001])
parser.add_argument('--alpha_range', nargs='+', type=float,
                    default=[0.6, 0.8, 1.])
args = parser.parse_args()

# !!! Always !!! Otherwise major headache on RTX 3090 cards
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False
torch.backends.cudnn.benchmark        = True

# GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu);

# Seeding
manualSeed = 2020 # fix seed
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Target file
target_dir = 'wgan_%s_%.2f/extra1' % (args.model, args.spacing)
target_file = os.path.join(target_dir, 'weights_epoch6000.pt')
contents    = torch.load(target_file)
# Get config
config      = contents['config']

# Get training dataset
train_seed, val_seed = 1234, 4321
dataset    = Channels(train_seed, config, norm=config.data.norm_channels)
dataloader = torch.utils.data.DataLoader(dataset, 
         batch_size=config.batchSize, shuffle=True, num_workers=2)

# Extract stuff
ngpu = 1 # Always
nz   = int(config.nz)
ngf  = int(config.ngf)
ndf  = int(config.ndf)
nc   = int(config.nc)
n_extra_layers = int(config.n_extra_layers)

# Get generator and load weights
netG = dcgan.DCGAN_G_Ours(config.imageSize, nz, nc, ngf, ngpu,
                          n_extra_layers+config.extra_gen_layers)
# !!! Load weights
netG.load_state_dict(contents['gen_state'])
netG = netG.cuda()
netG.eval()

# Hyper-combinations
channel      = args.channel
snr_range    = np.arange(-10, 17.5, 2.5)
noise_range  = 10 ** (-snr_range / 10.)
total_steps  = 5000 # Comparable
spacing_list = np.asarray([args.spacing])
l2lam_range  = np.asarray(args.l2lam_range)
lr_range     = np.asarray(args.lr_range)
pilot_alpha_range  = np.asarray(args.alpha_range)
# All combos
meta_params = itertools.product(l2lam_range, lr_range, pilot_alpha_range)

# Number of samples to keep
kept_samples = 100

# Logs
oracle_log = np.zeros((len(l2lam_range), len(lr_range), len(pilot_alpha_range),
           len(snr_range), total_steps, kept_samples)) # Last one is num channels
meas_log, reg_log = np.copy(oracle_log), np.copy(oracle_log)
result_dir = target_dir
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# !!! Always use exactly the same initial points
np.random.seed(2021)
global_init_z = np.random.normal(size=(kept_samples, nz, 1, 1))
# Logging
val_Y_log     = []

# For each hyper-combo
for meta_idx, (l2_lam, lr, pilot_alpha) in tqdm(enumerate(meta_params)):
    # Unwrap indicess
    l2_idx, lr_idx, alpha_idx = np.unravel_index(
        meta_idx, (len(l2lam_range), len(lr_range), len(pilot_alpha_range)))
    
    # Get a validation dataset and adjust parameters
    val_config = copy.deepcopy(config)
    val_config.data.spacing_list = spacing_list
    val_config.data.num_pilots   = \
        int(np.floor(config.data.num_pilots * pilot_alpha))
        
    # Create locals
    val_config.data.channel = args.channel
    val_dataset = Channels(val_seed, val_config, 
                           norm=[dataset.mean, dataset.std])
    val_loader  = DataLoader(val_dataset, batch_size=kept_samples,
        shuffle=False, num_workers=0, drop_last=True)
    val_iter = iter(val_loader) # For validation
    
    # Always the same initial points and data for validation
    val_sample = next(val_iter)
    val_H, val_P, _ = \
        val_sample['H'].cuda(), val_sample['P'].cuda(), val_sample['Y'].cuda()
    # Complexity
    val_H = val_H[:, 0] + 1j * val_H[:, 1]
    
    # For each SNR value
    for snr_idx, local_noise in tqdm(enumerate(noise_range)):
        val_Y     = torch.matmul(val_H, val_P)
        val_Y     = val_Y + \
            np.sqrt(local_noise) / np.sqrt(2.) * torch.randn_like(val_Y) 
        oracle    = val_H
        
        # Log measurements
        val_Y_log.append(val_Y.cpu().numpy())
        
        # Get a variable and latent optimizer - !! always the same !!
        init_z   = np.copy(global_init_z)
        latent_z = torch.tensor(init_z, dtype=torch.float32,
                                requires_grad=True, device='cuda:0')
        optimizer = torch.optim.Adam(params=[latent_z], lr=lr)
        
        # Optimize
        for step_idx in tqdm(range(total_steps)):
            # Generate channel and complexify
            gen_channels = netG(latent_z)
            gen_channels = gen_channels[:, 0] + 1j * gen_channels[:, 1]
            # Regenerate measurements
            gen_meas = torch.matmul(gen_channels, val_P)
            
            # Get measurement error and regularization
            meas_loss = torch.sum(torch.square(torch.abs(gen_meas - val_Y)),
                                  axis=(-1, -2))
            reg_loss  = torch.sum(torch.square(torch.abs(latent_z)),
                                  axis=(-1, -2, -3))
            
            # Final loss
            loss = torch.mean(meas_loss + l2_lam * reg_loss)
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Store logs
            with torch.no_grad():
                oracle_log[l2_idx, lr_idx, alpha_idx, snr_idx, step_idx] = \
                    (torch.sum(torch.square(torch.abs(gen_channels - oracle)),
                               dim=(-1, -2))/\
                    torch.sum(torch.square(torch.abs(oracle)), 
                              dim=(-1, -2))).detach().cpu().numpy()
            meas_log[l2_idx, lr_idx, alpha_idx, snr_idx, step_idx] = \
                meas_loss.detach().cpu().numpy()       
            reg_log[l2_idx, lr_idx, alpha_idx, snr_idx, step_idx]  = \
                reg_loss.detach().cpu().numpy()                       
                 
    # Delete validation dataset
    del val_dataset, val_loader
    torch.cuda.empty_cache()

# Save results to file
torch.save({'spacing_range': spacing_list,
            'pilot_alpha_range': pilot_alpha_range,
            'config': config,
            'snr_range': snr_range,
            'val_config': val_config,
            'l2lam_range': l2lam_range,
            'lr_range': lr_range,
            'oracle_log': oracle_log,
            'meas_log': meas_log,
            'reg_log': reg_log,
            'args': args
            }, result_dir + '/wgan_results_model%s_channel%s_DETAILED.pt' % (
                args.model, args.channel))