# Trajectory Prediction with Compressed 3D Environment Representation using Tensor Train Decomposition #

In this work, we use TT-SDF as an environment descriptor, to predict good initial trajectory for warm starting trajectory optimization. This repository contains the code for this work.

## Installation Procedure ##
Install pinocchio, crocoddyl, & tensorly:
```bash
conda install -c conda-forge crocoddyl
conda install -c conda-forge pinocchio
conda install -c tensorly tensorly
```

Install transforms3d:
```bash
pip install transforms3d
```

Install pybullet:
```bash
pip install pybullet
```

Install tensorly:

## How to use the codes ##
For running the specific experiments in the paper, you can look at the following notebooks:
```bash
generate_data_pointmass.ipynb,
generate_data_quadcopter.ipynb,
learn_data_pointmass.ipynb,
learn_data_quadcopter.ipynb
warmstart.ipynb,
warmstart_quadcopter.ipynb
```
in the notebook folder.
