# 2 GPUs, 1 min 47 s / epoch (only training), 2 min 12 s / epoch (only evaluation)
# CUDA_VISIBLE_DEVICES=0,1 python train_ddp.py -std_dev 0.1 0.01 -res 32 -m ShapeNetPoints -batch_size 4 -pointcloud -pc_samples 1000 -exp_name Try -gpus 2
# 4 GPUs, 54 s / epoch (only training), 1 min 1 s / epoch (only evaluation)
CUDA_VISIBLE_DEVICES=4,5,6,7 python train_ddp.py -std_dev 0.1 0.01 -res 32 -m ShapeNetPoints -batch_size 4 -pointcloud -pc_samples 1000 -class_name chair -num_sp_mesh_sample 6000 -exp_name Chair_scoda03_l6_ -gpus 4