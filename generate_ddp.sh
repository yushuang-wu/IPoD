# 1 GPU, it took 18 min 18 s for total generating.
CUDA_VISIBLE_DEVICES=0,1 python generate_ddp.py -std_dev 0.1 0.01 -res 32 -pc_samples 1000 -m ShapeNetPoints -checkpoint 90 -batch_points 200000 -pointcloud -dataset scannet -retrieval_res 128 -class_name chair -exp_name Chair_scoda_ -gpus 2
# 4 GPUs, it took 4 min 31 s for total generating.
# CUDA_VISIBLE_DEVICES=0,1,2,3 python generate_ddp.py -std_dev 0.1 0.01 -res 32 -pc_samples 1000 -m ShapeNetPoints -checkpoint 2 -batch_points 200000 -pointcloud -dataset scannet -retrieval_res 128 -exp_name Try -gpus 4