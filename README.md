# diffusion-channels
Source code for paper "Deep Diffusion Models for Robust Channel Estimation".

Generic flow:
1. Use 'matlab/main.m' to generate training, validation and test channels.
2. Use 'train.py' to train a deep diffusion model for channel estimation with the default parameters used in the paper.
3. Use 'hyperparam_tuning.py' to find 'beta' and 'N', exactly like in the paper.

3.1. The script will contain a saved variable ```oracle_log```, which contains the NMSE with respect to the ground truth channels, for all the hyper-parameters, noise levels, and each invididual sample.

3.2. Averaging the error across all samples ```(axis=-1)``` and using ```argmax``` over the corresponding axes will return the best hyper-parameters for each invididual SNR point (in a loop, assuming known SNR, or also averaged across SNR in the blind setting).

4. Use 'inference.py' to perform inference with the hyper-parameters found before.
5. (Optional, incomplete) Use ```train_wgan.py``` to train a WGAN model.

# TODO
- Complete code for extensive baselines.
- Support for saving the channels to a file, and adding a .mat script that runs end-to-end simulations with the estimated channels (for a quick solution: look into the ```pymatbridge``` Python package which allows for Matlab engine calls from Python).

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
