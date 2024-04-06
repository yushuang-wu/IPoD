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

import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from src.model.ipod import IPoD_transfomer
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.co3d_dataset import CO3DV2Dataset, co3dv2_collate_fn
# from util.hypersim_dataset import HyperSimDataset, hypersim_collate_fn
from src.engine.engine import train_one_epoch, eval_one_epoch, eval_one_epoch_udf
from src.engine.engine_viz import run_viz, run_viz_udf
from util.co3d_utils import get_all_dataset_maps

from pathlib import Path
from parser_and_builder import *
import torch.nn as nn
from pytorch3d.loss import chamfer_distance

import warnings
warnings.filterwarnings("ignore")


def main(args):
    misc.init_distributed_mode(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # define the model
    model = NUMCC(args=args)
    model.to(device)
    model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 512

    print("base lr: %.2e" % (args.blr))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)


    if args.reset_geo:
        with torch.no_grad():
            model_without_ddp.fc_out[1].weight[0] = torch.rand(512)*0.0001
            model_without_ddp.fc_out[1].bias[0] = 0.001

    if args.use_hypersim:
        dataset_type = HyperSimDataset
        collate_fn = hypersim_collate_fn
        dataset_maps = None
    else:
        dataset_type = CO3DV2Dataset
        collate_fn = co3dv2_collate_fn
        dataset_maps = get_all_dataset_maps(args.co3d_path, args.holdout_categories, one_class = args.one_class)

    dataset_viz = dataset_type(args, is_train=False, is_viz=True, dataset_maps=dataset_maps, fix=args.fix)
    sampler_viz = torch.utils.data.DistributedSampler(dataset_viz, num_replicas=num_tasks, rank=global_rank, shuffle=False)

    data_loader_viz = torch.utils.data.DataLoader(
        dataset_viz, batch_size=1,
        sampler=sampler_viz,
        num_workers=args.num_eval_workers,
        pin_memory=args.pin_mem,
        collate_fn=collate_fn,
    )

    if args.run_viz != True:
        data_loader_train = build_loader(
                args, num_tasks, global_rank,
                is_train=True,
                dataset_type=dataset_type, collate_fn=collate_fn, dataset_maps=dataset_maps)

        data_loader_val = build_loader(
                args, num_tasks, global_rank,
                is_train=False,
                dataset_type=dataset_type, collate_fn=collate_fn, dataset_maps=dataset_maps)

    # Define loss functions
    loss_fns = {}
    if args.geo == 'occ':
        loss_fns[args.geo] = nn.BCEWithLogitsLoss()
        eval_fn = eval_one_epoch
        viz_fn = run_viz
    elif args.geo == 'udf':
        loss_fns[args.geo] = nn.L1Loss()
        eval_fn = eval_one_epoch_udf
        viz_fn = run_viz_udf
    loss_fns['rgb'] = nn.CrossEntropyLoss()
    loss_fns['centers'] = chamfer_distance

    # Create experiment directory
    output_dir = os.path.join('experiments', args.exp_name)
    Path(os.path.join(output_dir, 'viz')).mkdir(parents= True, exist_ok=True)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    if args.run_viz:
        viz_fn(model, data_loader_viz, device, args=args, epoch=None)
        if args.run_val == False:
            return

    if args.run_val:
        eval_fn(model, data_loader_val, device, loss_fns=loss_fns, args=args)
        return


    for epoch in range(args.start_epoch, args.epochs):
        print(f'Epoch {epoch}:')
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            
        # val_stats = eval_fn(model, data_loader_val, device, loss_fns=loss_fns, args=args)

        train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler, loss_fns=loss_fns, args=args, output_dir=output_dir)
        misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, 
                            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, output_dir=output_dir, last = True)

        val_stats = {}
        if (epoch % args.val_every == args.val_every-1 or epoch + 1 == args.epochs) or args.debug:

            val_stats = eval_fn(model, data_loader_val, device, loss_fns=loss_fns, args=args)

        if output_dir and (epoch % args.save_every == args.save_every-1 or epoch + 1 == args.epochs):
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, 
                            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, output_dir=output_dir)

        log_stats = {**{f'train_{k}': round(v, 6) for k, v in train_stats.items()},
                     **{f'{k}': round(v, 4) for k, v in val_stats.items()},
                     'epoch': epoch,}

        if output_dir and misc.is_main_process():
            with open(os.path.join(output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        if ((epoch % args.viz_every == args.viz_every-1 or epoch + 1 == args.epochs) or args.debug):
            viz_fn(model, data_loader_viz, device, args=args, epoch=epoch)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()
    
    print(args)
    main(args)

