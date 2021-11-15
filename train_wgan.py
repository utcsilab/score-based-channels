#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 15:27:51 2021

@author: yanni
"""

from __future__ import print_function
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import os

import numpy as np
from loaders import Channels

import aux_gan as dcgan
from dotmap import DotMap

# !!! Always !!! Otherwise major headache on RTX 3090 cards
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False
torch.backends.cudnn.benchmark        = True

# GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "3";

# Seeding
manualSeed = 2020 # fix seed
np.random.seed(2020)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Configuration
config = DotMap()
config.imageSize = [16, 64] # Shortest side
config.nc        = 2
config.data.spacing_list  = [0.1]
config.data.norm_channels = 'entrywise'
# Models
config.nz        = 60
config.ndf       = 64
config.ngf       = 128
# Training
config.niter     = 3000
config.batchSize = 200
config.lrD       = 5e-5
config.lrG       = 5e-5
config.beta1     = 0.5
# WGAN
config.clamp_lower = -0.01
config.clamp_upper = 0.01
config.Diters      = 5

# Get channel dataset
train_seed = 1234
config.data.image_size = config.imageSize
config.data.num_pilots = config.data.image_size[1]
config.data.noise_std  = 0.
dataset    = Channels(train_seed, config, norm=config.data.norm_channels)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batchSize,
                                         shuffle=True, num_workers=2)
# Extract stuff
ngpu = 1 # Always
nz   = int(config.nz)
ngf  = int(config.ngf)
ndf  = int(config.ndf)
nc   = int(config.nc)
# More capacity for lambda/2 or mixed
if np.isin(0.5, config.data.spacing_list):
    config.n_extra_layers = 1
else:
    config.n_extra_layers = 0
n_extra_layers = int(config.n_extra_layers)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Get generator
netG = dcgan.DCGAN_G_Ours(config.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
netG.apply(weights_init)
netG = netG.cuda()

# Get discriminator
netD = dcgan.DCGAN_D(config.imageSize, nz, nc, ndf, ngpu, n_extra_layers)
netD.apply(weights_init)
netD = netD.cuda()

input = torch.FloatTensor(config.batchSize, 2, config.imageSize[0], config.imageSize[1])
noise = torch.FloatTensor(config.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(config.batchSize, nz, 1, 1).normal_(0, 1)
one  = torch.FloatTensor([1])
mone = one * -1
# Move to CUDA
input = input.cuda()
one, mone = one.cuda(), mone.cuda()
noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

# Setup optimizers
optimizerD = optim.RMSprop(netD.parameters(), lr = config.lrD)
optimizerG = optim.RMSprop(netG.parameters(), lr = config.lrG)

# Logs
if len(config.data.spacing_list) == 1:
    model_dir = 'wgan_CDL_D_%.2f' % config.data.spacing_list[0]
else:
    model_dir = 'wgan_CDL_D_min%.2f_max%.2f' % (
        np.min(config.data.spacing_list), np.max(config.data.spacing_list))
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
d_log, g_log = [], []
d_real_log, d_fake_log = [], []

# Here we go
gen_iterations = 0
for epoch in range(config.niter):
    data_iter = iter(dataloader)
    i = 0
    while i < len(dataloader):
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        # train the discriminator Diters times
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 100
        else:
            Diters = config.Diters
        j = 0
        while j < Diters and i < len(dataloader):
            j += 1

            # clamp parameters to a cube
            for p in netD.parameters():
                p.data.clamp_(config.clamp_lower, config.clamp_upper)

            sample = data_iter.next()
            i += 1

            # train with real
            real_cpu = sample['H'].cuda()
            netD.zero_grad()
            batch_size = real_cpu.size(0)

            input.resize_as_(real_cpu).copy_(real_cpu)
            inputv = Variable(input)

            errD_real = netD(inputv)
            errD_real.backward(one)

            # train with fake
            noise.resize_(config.batchSize, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise, volatile = True) # totally freeze netG
            fake = Variable(netG(noisev).data)
            inputv = fake
            errD_fake = netD(inputv)
            errD_fake.backward(mone)
            errD = errD_real - errD_fake
            optimizerD.step()

        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False # to avoid computation
        netG.zero_grad()
        # in case our last batch was the tail batch of the dataloader,
        # make sure we feed a full batch of noise
        noise.resize_(config.batchSize, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        errG = netD(fake)
        errG.backward(one)
        optimizerG.step()
        gen_iterations += 1

        print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
            % (epoch, config.niter, i, len(dataloader), gen_iterations,
            errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))
        
        # Log stuff
        d_log.append(errD.data[0])
        g_log.append(errG.data[0])
        d_real_log.append(errD_real.data[0])
        d_fake_log.append(errD_fake.data[0])
        
    # Save models periodically
    if np.mod(epoch+1, 500) == 0:
        torch.save({'gen_state': netG.state_dict(),
                    'disc_state': netD.state_dict(),
                    'gen_opt_state': optimizerG.state_dict(),
                    'disc_opt_state': optimizerD.state_dict(),
                    'config': config}, model_dir + '/weights_epoch%d.pt' % (epoch+1))