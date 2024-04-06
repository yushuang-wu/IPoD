# Copyright 2023 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import torch


def get_args_parser():
    parser = argparse.ArgumentParser('MCC', add_help=False)

    # Model
    parser.add_argument('--input_size', default=224, type=int,
                        help='Images input size')
    parser.add_argument('--occupancy_weight', default=1.0, type=float,
                        help='A constant to weight the occupancy loss')
    parser.add_argument('--rgb_weight', default=0.01, type=float,
                        help='A constant to weight the color prediction loss')
    parser.add_argument('--drop_path', default=0.1, type=float,
                        help='drop_path probability')
    parser.add_argument('--regress_color', action='store_true',
                        help='If true, regress color with MSE. Otherwise, 256-way classification for each channel.')

    # Training
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU for training (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--eval_batch_size', default=2, type=int,
                        help='Batch size per GPU for evaluation (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--accum_iter', default=8, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 512')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warmup LR')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='Clip gradient at the specified norm')

    # Job
    parser.add_argument('--job_dir', default='',
                        help='Path to where to save, empty for no saving')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Path to where to save, empty for no saving')
    parser.add_argument('--exp_name', default='test',
                        help='Experiment name')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int,
                        help='Random seed.')
    parser.add_argument('--resume', default='',
                        help='Resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Start epoch')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of workers for training data loader')
    parser.add_argument('--num_eval_workers', default=4, type=int,
                        help='Number of workers for evaluation data loader')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='Url used to set up distributed training')

    # Experiments
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--run_viz', action='store_true',
                        help='Specify to run only the visualization/inference given a trained model.')
    parser.add_argument('--max_n_viz_obj', default=10, type=int,
                        help='Max number of objects to visualize during training.')

    # Data
    parser.add_argument('--train_epoch_len_multiplier', default=32, type=int,
                        help='# examples per training epoch is # objects * train_epoch_len_multiplier')
    parser.add_argument('--eval_epoch_len_multiplier', default=1, type=int,
                        help='# examples per eval epoch is # objects * eval_epoch_len_multiplier')

    # CO3D
    parser.add_argument('--co3d_path', type=str, default='/mnt/vlab/yushuang/co3d/co3d_data',
                        help='Path to CO3D v2 data.')
    parser.add_argument('--holdout_categories', action='store_true',
                        help='If true, hold out 10 categories and train on only the remaining 41 categories.')
    parser.add_argument('--co3d_world_size', default=3.0, type=float,
                        help='The world space we consider is \in [-co3d_world_size, co3d_world_size] in each dimension.')

    # Hypersim
    parser.add_argument('--use_hypersim', action='store_true',
                        help='If true, use hypersim, else, co3d.')
    parser.add_argument('--hypersim_path', default="hypersim_data", type=str,
                        help="Path to Hypersim data.")

    # Data aug
    parser.add_argument('--random_scale_delta', default=0.2, type=float, 
                        help='Random scaling each example by a scaler \in [1 - random_scale_delta, 1 + random_scale_delta].') #0.2
    parser.add_argument('--random_shift', default=1.0, type=float,
                        help='Random shifting an example in each axis by an amount \in [-random_shift, random_shift]')  #1.0
    parser.add_argument('--random_rotate_degree', default=180, type=int, 
                        help='Random rotation degrees.')

    # Smapling, evaluation, and coordinate system
    parser.add_argument('--shrink_threshold', default=10.0, type=float,
                        help='Any points with distance beyond this value will be shrunk.')
    parser.add_argument('--semisphere_size', default=6.0, type=float,
                        help='The Hypersim task predicts points in a semisphere in front of the camera.'
                             'This value specifies the size of the semisphere.')
    parser.add_argument('--eval_granularity', default=0.1, type=float,
                        help='Granularity of the evaluation points.')
    parser.add_argument('--viz_granularity', default=0.1, type=float,
                        help='Granularity of points in visaulizatoin.')

    parser.add_argument('--eval_score_threshold', default=0.4, type=float,
                        help='Score threshold for evaluation.')
    parser.add_argument('--eval_dist_threshold', default=0.1, type=float,
                        help='Points closer than this amount to a groud-truth is considered correct.')
    parser.add_argument('--train_dist_threshold', default=0.1, type=float,
                        help='Points closer than this amount is considered positive in training.')
    
    # More args
    parser.add_argument('--print_every', default=200, type=int)
    parser.add_argument('--val_every', default=20, type=int)
    parser.add_argument('--viz_every', default=20, type=int)
    parser.add_argument('--save_every', default=1, type=int)
    parser.add_argument('--run_val', action='store_true')

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--n_queries', default=550, type=int)
    parser.add_argument('--fix', default=0, type=int)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--cd_norm', default=1, type=int)


    parser.add_argument('--n_groups', default=200, type=int)
    parser.add_argument('--nneigh', default=4, type=int)
    parser.add_argument('--nn_seen', default=4, type=int)
    parser.add_argument('--xyz_size', default=112, type=int)
    parser.add_argument('--hr', action='store_true')
    parser.add_argument('--xyz_size_hr', default=224, type=int)
    parser.add_argument('--no_fine', default=0, type=int)
    parser.add_argument('--geo', type=str, default='udf', choices=['udf','occ'])
    parser.add_argument('--reset_geo', action='store_true')

    parser.add_argument('--udf_threshold', default=0.23, type=float)
    parser.add_argument('--udf_n_iter', default=10, type=int)
    parser.add_argument('--repulsive', default=1, type=int, choices=[0,1])
    parser.add_argument('--n_query_udf', default=12000, type=int)

    parser.add_argument('--save_pc', action='store_true')
    parser.add_argument('--one_class', default='', type=str)

    return parser

def build_loader(args, num_tasks, global_rank, is_train, dataset_type, collate_fn, dataset_maps):
    '''Build data loader'''
    dataset = dataset_type(args, is_train=is_train, dataset_maps=dataset_maps)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset, num_replicas=num_tasks, rank=global_rank, shuffle=is_train
    )

    batch_size = args.batch_size if is_train else args.eval_batch_size

    if args.geo == 'udf' and is_train == False:
        batch_size = 1

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        sampler=sampler_train,
        num_workers=args.num_workers if is_train else args.num_eval_workers,
        pin_memory=args.pin_mem,
        collate_fn=collate_fn,
    )
    return data_loader

def build_loader_subset(args, num_tasks, global_rank, is_train, dataset_type, collate_fn, dataset_maps):
    '''Build data loader'''
    dataset = dataset_type(args, is_train=is_train, dataset_maps=dataset_maps)

    subset_indices = list(range(20)) # select your indices here as a list
    subset = torch.utils.data.Subset(dataset, subset_indices)

    sampler_train = torch.utils.data.DistributedSampler(
        subset, num_replicas=num_tasks, rank=global_rank, shuffle=is_train
    )

    data_loader = torch.utils.data.DataLoader(
        subset, batch_size=2,
        sampler=sampler_train,
        num_workers=args.num_workers if is_train else args.num_eval_workers,
        pin_memory=args.pin_mem,
        collate_fn=collate_fn,
    )
    return data_loader