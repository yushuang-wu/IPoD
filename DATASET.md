# Dataset Preparation

## CO3D v2
1. Please follow [the official instruction](https://github.com/facebookresearch/co3d) to download the dataset.
2. We use the dataset provider in [Implicitron](https://github.com/facebookresearch/pytorch3d/tree/main/pytorch3d/implicitron) for data loading. To speed up the loading, we cache the loaded meta data. Please run 
```
cd scripts
python prepare_co3d.py --co3d_path [path to CO3D data]
```
to generate the cache. The cached data take ~4.1GB of space.
