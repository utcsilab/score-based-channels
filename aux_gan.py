#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.parallel
import numpy as np

class DCGAN_D(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_D, self).__init__()
        self.ngpu = ngpu
        assert np.min(isize) % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial:{0}-{1}:conv'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial:{0}:relu'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = np.min(isize) / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}:{1}:batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid:{0}-{1}:conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid:{0}:relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf  = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 16
        main.add_module('final:{0}-{1}:conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, (4, 16), 1, (0, 0), bias=False))
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else: 
            output = self.main(input)
        
        # !!! This directly outputs average over batch
        return output.mean(0).view(1)
    
class DCGAN_G_Ours(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN_G_Ours, self).__init__()
        # Parameters
        self.Nr = isize[0]
        self.Nt = isize[1]
        
        # Dense layer
        dense = nn.Sequential()
        output_dense = int(ngf * isize[0] * isize[1] / 16)
        dense.add_module('dense_input', nn.Linear(nz, output_dense))
        
        # Convolutional layers
        conv = nn.Sequential()
        # First block
        conv.add_module('up1', nn.Upsample(scale_factor=(2, 2),
                                           mode='nearest'))
        conv.add_module('conv1', nn.Conv2d(ngf, ngf, 5, padding=2))
        conv.add_module('bn1', nn.BatchNorm2d(ngf))
        conv.add_module('relu1', nn.ReLU())
        
        # Second
        conv.add_module('up2', nn.Upsample(scale_factor=(2, 2),
                                           mode='nearest'))
        conv.add_module('conv2', nn.Conv2d(ngf, ngf, 5, padding=2))
        conv.add_module('bn2', nn.BatchNorm2d(ngf))
        conv.add_module('relu2', nn.ReLU())
        
        # Extra layers
        for t in range(n_extra_layers):
            conv.add_module('extra_conv%d' % t,
                            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False))
            conv.add_module('extra_bn%d' % t,
                            nn.BatchNorm2d(ngf))
            conv.add_module('extra_relu%d' % t,
                            nn.ReLU())

        # Output conv
        conv.add_module('conv_out', nn.Conv2d(ngf, nc, 5, padding=2))
        
        # Modules
        self.dense = dense
        self.conv  = conv
        
    def forward(self, z):
        # Dense
        hidden = self.dense(z.squeeze())
        
        # Reshape to hidden image
        hidden = hidden.view(-1, 128, self.Nr//4, self.Nt//4)
        
        # Conv
        output = self.conv(hidden)
        
        return output
