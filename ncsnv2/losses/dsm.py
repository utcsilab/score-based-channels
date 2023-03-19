#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

def anneal_dsm_score_estimation(scorenet, samples, sigmas,
                                labels=None, anneal_power=2.):
    # This always enters during training
    if labels is None:
        # Randomly sample sigma
        labels = torch.randint(0, len(sigmas), 
                   (samples.shape[0],), device=samples.device)
    
    used_sigmas = sigmas[labels].view(samples.shape[0], * ([1] * len(samples.shape[1:])))
    noise       = torch.randn_like(samples) * used_sigmas
    
    perturbed_samples = samples + noise
    
    # Desired output
    target = - 1 / (used_sigmas ** 2) * noise
    
    # Actual output
    scores = scorenet(perturbed_samples, labels)
    
    # L2 regression
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    
    # Multiply each sample by its weight
    loss = 1 / 2. * ((scores - target) ** 2).sum(
        dim=-1) * used_sigmas.squeeze() ** anneal_power

    return loss.mean(dim=0)