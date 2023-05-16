# SCoDA
> SCoDA: Domain Adaptive Shape Completion for Real Scans <br />
> [GAP Lab](https://gaplab.cuhk.edu.cn/), [Yushuang Wu](https://scholar.google.com/citations?hl=zh-CN&user=x5gpN0sAAAAJ)

![Teaser](figures/teaser_2k.png)

[Paper](https://arxiv.org/abs/2304.10179.pdf) - 
[Project Website](https://yushuang-wu.github.io/SCoDA/) -
[Arxiv](https://arxiv.org/abs/2304.10179) -
Published in CVPR 2023.

#### Citation

If you find our code or paper useful for your project, please consider citing:

    @inproceedings{wu2023scoda,
      title={SCoDA: Domain Adaptive Shape Completion for Real Scans},
      author={Yushuang, Wu and Zizheng, Yan and Ce, Chen and Lai, Wei and Xiao, Li and Guanbin, Li and Yihao, Li and Shuguang, Cui and Xiaoguang, Han},
      booktitle={The IEEE/CVF Computer Vision and Pattern Recognition Conference (CVPR)},
      year={2023},
    }
    
## ScanSalon

We build a new dataset, ScanSalon, for the shape completion of real scans, with 800 mesh-scan pairs in 6 classes: chair, desk, sofa, bed, lamp, car. We provide videos in our [project page](https://yushuang-wu.github.io/SCoDA/) for data sample visualization. 

![Dataset](figures/dataset_vis.png)

> ScanSalon Data: At [Google Drive](https://drive.google.com/drive/folders/1JrBxlBufivinI5_Xyi-1wBz2quU-DThj?usp=sharing) (Real data only).  <br />
> ShapeNet Data: Turn to [ShapeNet](https://shapenet.org) for synthetic data download.  <br />
> Processing: Please refer to the README and scripts in the zip package.
    
## Install

Our implementation is based on IF-Net as the basic framework for reconstruction. Please refer to the "Install" part of [IF-Net](https://github.com/jchibane/if-net) for the installation of our method. 

## Running

1. Following steps in data_processing/mesh-fusion to get the water-tight ScanSalon meshes. <br />
2. Following steps in [Mesh2PC](https://github.com/kochanha/Mesh-to-Pointcloud-using-Blensor) to get simulated scans from ShapeNet meshes. <br /> 
3. Following steps in data_processing/process.sh to preprocess all data. <br />
4. Run train_ddp.sh to train the model in a parrallel way. <br />
5. After training by around 30-50 epochs, run generate_ddp.sh to generate meshes in the test set. 

![Methodology](figures/method.png)
