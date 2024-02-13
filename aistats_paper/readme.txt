# A Cubic-regularized Policy Newton Algorithm for Reinforcement Learning

This directory contains the source code of the experiments as shown in the main paper. The directory is structured as follows:

aistats_paper \
├── deep  # experiments related to ACR-PN specifically using deep nueral networks \
├    ├── acrpn.py  # runner for ACR-PN algorithm \
├    ├── optimizers.py  # implementation of algorithm \
├    ├── sgd.py  # runner for REINFORCE benchmark \
├    └── utils.py  # helper functions / methods \
└── linear  # experiments related to CR-PN and REINFORCE using linear function approximators \
     ├── agent.py  # base class \
     ├── crpn.py  # runner for CR-PN algorithm \
     ├── crpn_linear.py  # main class for CR-PN \
     ├── sgd.py  # runner for REINFORCE algorithm \
     └── sgd_linear.py  # main class for REINFORCE

## Installation

Need installation of gymnasium, pytorch, etc.

## Usage

You may run the following command on terminal for example:
```
python deep/acrpn.py --exp-name acrpn_test --env-seed -1 --save True --track False --capture-video False --alpha 10000
```
