# Trajectory Prediction with Compressed 3D Environment Representation using Tensor Train Decomposition #

In this work, we use TT-SDF as an environment descriptor, to predict good initial trajectory for warm starting trajectory optimization. This repository contains the code for this work.

## Dependencies##
```bash
conda install -c conda-forge crocoddyl
conda install -c conda-forge pinocchio
conda install -c tensorly tensorly
conda install tensorflow tensorflow_probability
pip install transforms3d
pip install pybullet
pip install tqdm
pip install pandas
pip install trimesh
pip install scikit-learn
pip install casadi
pip install scikit-image
pip install pyrender
pip install meshio
pip install mesh-to-sdf
```

Install sdf:
```bash
see https://github.com/fogleman/sdf
```


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
