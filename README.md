# A Cubic-regularized Policy Newton Algorithm for Reinforcement Learning

This directory contains the source code of the experiments as shown in the main paper that can be found here https://proceedings.mlr.press/v238/maniyar24a.html. The directory is structured as follows:


+ `aistats_paper`  : _experiments mentioned in the main paper_
  + `experiments_deep.py`  : _experiments related to ACR-PN using deep neural networks_
  + `experiments_linear.py`  : _experiments related to CR-PN using linear function approximation_

+ `deep` : _source code for deep experiments_
  + `acrpn.py` : _runner for ACR-PN algorithm_
  + `reinforce.py`  : _runner for REINFORCE benchmark_
  
+ `linear` : _source code for linear experiments_
  + `crpn.py` : _runner for CR-PN algorithm_
  + `reinforce.py` : _runner for REINFORCE benchmark_

## Installation

Need installation of gymnasium, pytorch, etc.

+ Installing `gymnasium` on windows : https://youtu.be/gMgj4pSHLww?si=1H-IStte7aDONybT
+ Installing `pytorch` : https://pytorch.org/get-started/locally/
+ Rest you can use `conda` or `pip`.

## Usage

You may run the following command on terminal for example:
```
python deep/acrpn.py --exp-name acrpn_test --env-seed -1 --save False --track False --alpha 10000
```
