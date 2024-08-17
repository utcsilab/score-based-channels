# MIMO Channel Estimation using Score-Based (Diffusion) Generative Models

[![IEEE Xplore](https://img.shields.io/badge/IEEE-TWC-00629B.svg)](https://ieeexplore.ieee.org/abstract/document/9957135)
[![arXiv](https://img.shields.io/badge/arXiv-2204.07122-b31b1b.svg)](https://arxiv.org/abs/2204.07122)

This repository contains source code for [MIMO Channel Estimation using Score-Based Generative Models](https://arxiv.org/abs/2204.07122), and contains code for training and testing a score-based generative model on channels from the Clustered Delay Line (CDL) family of models, as well as other algorithms.

## Requirements
Python 3.10, 3.11, and 3.12 with virtual environment support `sudo apt install python3.1X-venv`. Tested on Ubuntu 20.04 and 22.04. MATLAB license required to run MATLAB scripts.

## Getting Started
After cloning the repository, run the following commands for Python 3.10 (similar for other versions of Python):
- `cd score-based-channels`
- `python3.10 -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`

This will create a self-contained virtual environment in the base directory, activate it, and install all required packages.

### Pre-generated Data
Train and validation data for CDL-C channels can be directly downloaded from the command line using the following:
```
mkdir data
curl -L https://utexas.box.com/shared/static/nmyg5s06r6m2i5u0ykzlhm4vjiqr253m.mat --output ./data/CDL-C_Nt64_Nr16_ULA0.50_seed1234.mat
curl -L https://utexas.box.com/shared/static/2a7tavjo9hk3wyhe9vv0j7s2l6en4mj7.mat --output ./data/CDL-C_Nt64_Nr16_ULA0.50_seed4321.mat
```

For other channel distributions (CDL-A, CDL-B, CDL-D) shown in the paper the used training and validation data can be downloaded from the following public repository:

https://utexas.box.com/s/f7g7yqdw5w0fea0b59aym3xsvbvw1uch

Once downloaded, place these files in the `data` folder under the main directory.

### Pre-trained Models
A pre-trained diffusion model for CDL-C channels can be directly downloaded from the command line using the following:
```
mkdir -p models/score/CDL-C
curl -L https://utexas.box.com/shared/static/4nubcpvpuv3gkzfk8dgjo6ay0ssps66w.pt --output ./models/score/CDL-C/final_model.pt
```

This will create the nested directories `models/score/CDL-C` and place the weights there. Weights for models trained on other distributions (CDL-A, CDL-B, CDL-D, Mixed) shown in the paper can be downloaded from the following public repository:

https://utexas.box.com/s/m58udx6h0glwxua88zgdwrff87jvy3qw

Once downloaded, places these files in their matching directory structure as `final_model.pt`.

## Training Diffusion Models on MIMO Channels
After downloading the example CDL-C data, a diffusion model can be trained by running:
```
python train_score.py
```

The model is trained for 400 epochs by default, and the last model weights will be automatically saved in the `model` folder under the appropriate structure. To train on other channel distributions, see the `--train` argument.

## Channel Estimation with Diffusion Models
To run channel estimation with the CDL-C data and the pretrained model run:
```
python test_score.py
````

This will run channel estimation in the setting of Figure 5c of the paper, and will reproduce the `Score-based (CDL-C)` curve:

<img src="https://github.com/utcsilab/score-based-channels/blob/main/figures/fig5c_legend.png" width="860" height="600">

Running the above command will automatically plot and save results in the `results/score/train-CDL-C_test-CDL-C` folder. To run channel estimation on other channel distributions, see the `--train` and `--test` arguments, which dictate what pretrained model should be used and what the test distribution is, respectively.

## Hyper-parameter Tuning
Tuning the inference hyper-parameters `alpha` (the step size in Annealed Langevin Dynamics), `beta` (a multiplier for the noise added in each step of Annealed Langevin Dynamics), and `N` (the number of inference steps in Annealed Langevin Dynamics) can be done for the CDL-C pretrained model by running:
```
python tune_hparams_score.py
```

This will perform a grid search for the best values of `alpha`, `beta`, and `N` and will save the results in the `results/score` folder. To modify the searched values and the model that is being tuned, see the `alpha_step_range`, `beta_noise_range`, and `channel` arguments respectively.

## Generating Channel Data
For completeness, we also include the Matlab scripts used to generated all training and testing datasets in the `matlab` folder. The main script to run is `matlab/generate_data.m`.

# Citations
Full credits for the ncsnv2 repository go to: https://github.com/ermongroup/ncsnv2

Please include the following citation when using or referencing this codebase:
```
@ARTICLE{9957135,
  author={Arvinte, Marius and Tamir, Jonathan I.},
  journal={IEEE Transactions on Wireless Communications}, 
  title={MIMO Channel Estimation Using Score-Based Generative Models}, 
  year={2023},
  volume={22},
  number={6},
  pages={3698-3713},
  doi={10.1109/TWC.2022.3220784}}
```

Previous related publications are:
```
@inproceedings{arvinte2022score1,
  title={Score-Based Generative Models for Wireless Channel Modeling and Estimation},
  author={Arvinte, Marius and Tamir, Jonathan},
  booktitle={ICLR Workshop on Deep Generative Models for Highly Structured Data},
  year={2022}
}

@inproceedings{arvinte2022score2,
  title={Score-Based Generative Models for Robust Channel Estimation},
  author={Arvinte, Marius and Tamir, Jonathan I},
  booktitle={2022 IEEE Wireless Communications and Networking Conference (WCNC)},
  pages={453--458},
  year={2022},
  organization={IEEE}
}
```
