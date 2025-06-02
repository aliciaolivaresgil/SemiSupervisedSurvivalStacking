# SemiSupervisedSurvivalStacking

This repository contains the code of a modification of Survival Stacking [1], to transform the survival analysis problem into a semi-supervised binary classification problem. 

[1] Craig, Erin, Chenyang Zhong, and Robert Tibshirani. "Survival stacking: casting survival analysis as a classification problem." arXiv preprint arXiv:2107.13480 (2021).

It also includes the experimentation to compare both strategies on a set of narrow and wide datasets. 

## Requirements
### Data
All datasets are available in the [SurvSet](https://github.com/ErikinBC/SurvSet) repository. See `Survset_datasets_example.ipynb`. 
### Conda environments 
All the Python code was executed using the conda environments available in this repository: 
- `sksurv.yml`: Used to execute all the experimetns except the Hierarchical Bayesian tests.
- `baycomp.yml`: Used to execute the Bayesian tests.

To install these conda environments: 
```
conda env create -f sksurv.yml
```
To activate the environment: 
```
conda activate sksurv
```
## Usage
In order to reproduce the results, follow these steps: 
### 1. Comparison
Run comparison of the supervised and the semi-supervised survival stacking approaches: 
```
python Comparison_super.py
```
```
python Comparison_semi.py
```
### 2. Hierarchical Bayesian statistical tests
To check whether the differences between the two strategies are significant, a Hierarchical Bayesian test is run: 
```
python Statistical_Tests.py
```
