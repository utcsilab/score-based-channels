# diffusion-channels
Source code for paper "Deep Diffusion Models for Robust Channel Estimation".

Generic flow:
1. Use 'matlab/main.mat' to generate training, validation and test channels.
2. Use 'train.py' to train a deep diffusion model for channel estimation.
3. Use 'hyperparam_tuning.py' to find 'beta' and 'N'.
4. Use 'inference.py' to perform inference.

Full credits for the ncsnv2 repository go to: https://github.com/utcsilab/diffusion-channels/tree/main/ncsnv2
