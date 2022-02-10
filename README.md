# diffusion-channels
Source code for paper "Deep Diffusion Models for Robust Channel Estimation".

Generic flow:
1. Use 'matlab/main.m' to generate training, validation and test channels.
2. Use 'train.py' to train a deep diffusion model for channel estimation with the default parameters used in the paper.
3. Use 'hyperparam_tuning.py' to find 'beta' and 'N', exactly like in the paper.
4. Use 'inference.py' to perform inference.

Full credits for the ncsnv2 repository go to: https://github.com/utcsilab/diffusion-channels/tree/main/ncsnv2

# Citation
Please include the following citation when using or referencing this codebase:

```
@article{arvinte2021deep,
  title={Deep Diffusion Models for Robust Channel Estimation},
  author={Arvinte, Marius and Tamir, Jonathan I},
  journal={arXiv preprint arXiv:2111.08177},
  year={2021}
}
```
