# Installation

1. Create a conda environment

```
conda create -n numcc python=3.9
conda activate numcc
```

2. Install PyTorch, PyTorch3D, and other packages

```
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install jupyter
pip install scikit-image matplotlib imageio plotly opencv-python timm open3d pyntcloud h5py omegaconf
conda install pytorch3d==0.7.2 -c pytorch3d
```
