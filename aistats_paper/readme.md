# A Cubic-regularized Policy Newton Algorithm for Reinforcement Learning

This directory contains the source code of the experiments as shown in the main paper. The directory is structured as follows:

aistats_paper  # main folder \
├── deep  # NN experiments related to ACR-PN \
├&emsp;     ├── acrpn.py  # runner for ACR-PN algorithm \
├&emsp;     ├── optimizers.py  # implementation of algorithm \
├&emsp;     ├── sgd.py  # runner for REINFORCE benchmark \
├&emsp;     └── utils.py  # helper functions / methods \
└── linear  # shallow experiments related to CR-PN and REINFORCE \
&nbsp; &emsp;     ├── agent.py  # base class \
&nbsp; &emsp;     ├── crpn.py  # runner for CR-PN algorithm \
&nbsp; &emsp;     ├── crpn_linear.py  # main class for CR-PN \
&nbsp; &emsp;     ├── sgd.py  # runner for REINFORCE algorithm \
&nbsp; &emsp;     └── sgd_linear.py  # main class for REINFORCE

## Installation

Need installation of gymnasium, pytorch, etc. 

## Usage

You may run the following command on terminal for example:

```bash
python deep/acrpn.py --exp-name acrpn_test --env-seed -1 --save True --track False --capture-video False --alpha 10000
```
