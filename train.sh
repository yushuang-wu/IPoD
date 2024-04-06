PYTHONHASHSEED=42 CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 \
torchrun --nproc_per_node 1 main_ipod.py \
--batch_size 8 \
--num_workers 8 \
--exp_name IPoD_train \
--train_epoch_len_multiplier 4 \
--accum_iter 16 \
--holdout_categories \
--n_queries 2048 \
--n_query_udf 10240 \
--val_every 5 \
--viz_every 100