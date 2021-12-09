# diffusion-channels
Source code for paper "Deep Diffusion Models for Robust Channel Estimation".

Generic flow:
1. Use 'matlab/main.m' to generate training, validation and test channels.
2. Use 'train.py' to train a deep diffusion model for channel estimation with the default parameters used in the paper.
3. Use 'hyperparam_tuning.py' to find 'beta' and 'N', exactly like in the paper.
4. Use 'inference.py' to perform inference.

Full credits for the ncsnv2 repository go to: https://github.com/utcsilab/diffusion-channels/tree/main/ncsnv2
