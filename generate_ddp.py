'''
SCoDA Code v1
'''
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import Dataset, DataLoader
from data_processing.evaluation import eval_mesh
import argparse
import models.local_model_ddp as model
import models.data.data_region_ddp as voxelized_data
import torch.optim as optim
from torch.nn import functional as F
import os
import mcubes
import trimesh
import numpy as np
import torch.nn as nn
import torch
import data_processing.implicit_waterproofing as iw
from glob import glob
from  collections import OrderedDict
import warnings

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Run generation')

    parser.add_argument('-gpus', default=2, type=int)
    parser.add_argument('-dataset', default='shapenet', type=str)
    parser.add_argument('-pointcloud', dest='pointcloud', action='store_true')
    parser.add_argument('-voxels', dest='pointcloud', action='store_false')
    parser.set_defaults(pointcloud=False)
    parser.add_argument('-exp_name', default='Pretrain', type=str)
    parser.add_argument('-pc_samples', default=300, type=int)
    parser.add_argument('-dist',
                        '--sample_distribution',
                        default=[0.5, 0.5],
                        nargs='+',
                        type=float)
    parser.add_argument('-threshold', default=0.5, type=float)
    parser.add_argument('-std_dev',
                        '--sample_sigmas',
                        default=[],
                        nargs='+',
                        type=float)
    parser.add_argument('-res', default=32, type=int)
    parser.add_argument('-decoder_hidden_dim', default=256, type=int)
    parser.add_argument('-mode', default='test', type=str)
    parser.add_argument('-retrieval_res', default=256, type=int)
    parser.add_argument('-checkpoint', type=int)
    parser.add_argument('-batch_points', default=1000000, type=int)
    parser.add_argument('-m', '--model', default='LocNet', type=str)

    parser.add_argument('-class_name', default='chair', type=str)
    parser.add_argument('-data_root', default='/home/wuyushuang/data', type=str)
    parser.add_argument('-num_sp_mesh_sample', default=2000, type=int)

    args = parser.parse_args()
    return args


class ScodaGenerator(LightningModule):
    def __init__(self, hparams):
        super(ScodaGenerator, self).__init__()
        self.save_hyperparameters(hparams)
        # exp name
        self.exp_name = 'i{}_dist-{}sigmas-{}v{}_m{}'.format(
            str(hparams.exp_name) + str(hparams.pc_samples),
            ''.join(str(e) + '_' for e in hparams.sample_distribution),
            ''.join(str(e) + '_' for e in hparams.sample_sigmas), hparams.res,
            hparams.model)
        self.checkpoint_dir = './experiments/{}/checkpoints/'.format(self.exp_name)
        self.checkpoint = hparams.checkpoint
        self.output_dir = './experiments/{}/evaluation_{}_@{}_{}/'.format(
            self.exp_name, hparams.checkpoint, hparams.retrieval_res, hparams.dataset)
        os.makedirs(self.output_dir, exist_ok=True)
        # hparams
        self.threshold = hparams.threshold
        self.resolution = hparams.retrieval_res
        self.batch_points = hparams.batch_points
        self.min = -0.5
        self.max = 0.5
        # init models
        self.model_sp = model.ShapeNetPoints()
        self.model_sc = model.ScanNetPoints()
        self.impli_fu = model.ImplicitFunction()
        self.load_weights() # ! load weights of models
        # dataset
        self.dataset = voxelized_data.VoxelizedScanNet(
            hparams.mode,
            num_sp_mesh_sample=hparams.num_sp_mesh_sample, 
            data_root=hparams.data_root, 
            class_name=hparams.class_name,
            voxelized_pointcloud=hparams.pointcloud,
            pointcloud_samples=hparams.pc_samples,
            res=hparams.res,
            sample_distribution=hparams.sample_distribution,
            sample_sigmas=hparams.sample_sigmas,
            num_sample_points=100,
            batch_size=1,
            num_workers=0)

        grid_points = iw.create_grid_points_from_bounds(self.min, self.max, self.resolution)
        grid_points[:, 0], grid_points[:, 2] = grid_points[:, 2], grid_points[:, 0].copy()

        a = self.max + self.min
        b = self.max - self.min

        grid_coords = 2 * grid_points - a
        grid_coords = grid_coords / b

        grid_coords = torch.from_numpy(grid_coords).to(self.device, dtype=torch.float)
        grid_coords = torch.reshape(grid_coords, (1, len(grid_points), 3)).to(self.device)
        self.grid_points_split = torch.split(grid_coords, self.batch_points, dim=1)

    def load_weights(self):
        if self.checkpoint is None:
            checkpoints = glob(os.path.join(self.checkpoint_dir, '/*'))
            if len(checkpoints) == 0:
                raise RuntimeError('No checkpoints found in {}.'.format(self.checkpoint_dir))

            checkpoints = [os.path.splitext(os.path.basename(path))[0][6:] for path in checkpoints]
            checkpoints = np.array(checkpoints, dtype=int)
            checkpoints = np.sort(checkpoints)
            self.checkpoint_path = os.path.join(self.checkpoint_dir, 'epoch={}.ckpt'.format(checkpoints[-1]))
        else:
            self.checkpoint_path = os.path.join(self.checkpoint_dir, 'epoch={}.ckpt'.format(self.checkpoint))
        if not os.path.exists(self.checkpoint_path):
            raise RuntimeError('Checkpoint {} does not exist.'.format(self.checkpoint_path))
        ckpt = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))
        model_sp_weights = OrderedDict()
        model_sc_weights = OrderedDict()
        impli_fu_weights = OrderedDict()
        for k in ckpt['state_dict'].keys():
            if k.startswith('model_sp'):
                model_sp_weights[k[9:]] = ckpt['state_dict'][k]
            if k.startswith('model_sc'):
                model_sc_weights[k[9:]] = ckpt['state_dict'][k]
            if k.startswith('impli_fu'):
                impli_fu_weights[k[9:]] = ckpt['state_dict'][k]
        self.model_sp.load_state_dict(model_sp_weights)
        self.model_sc.load_state_dict(model_sc_weights)
        self.impli_fu.load_state_dict(impli_fu_weights)
        print('Loaded checkpoint from: {}'.format(self.checkpoint_path))

    def generate_mesh(self, data):
        inputs = data['inputs'].to(self.device)
        logits_list = []
        for p in self.grid_points_split:
            points = p.to(self.device)
            with torch.no_grad():
                # logits, _, _ = self.model(points, inputs)
                features_sp = self.model_sp(points, inputs)
                features = self.model_sc(points, inputs, features_sp)
                logits = self.impli_fu(features)
            logits_list.append(logits.squeeze(0).detach().cpu())
        logits = torch.cat(logits_list, dim=0)
        return logits.numpy()

    def mesh_from_logits(self, logits):
        logits = np.reshape(logits, (self.resolution,) * 3)
        # padding to ba able to retrieve object close to bounding box bondary
        logits = np.pad(logits, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=0)
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        vertices, triangles = mcubes.marching_cubes(
            logits, threshold)
        # remove translation due to padding
        vertices -= 1
        # rescale to original scale
        step = (self.max - self.min) / (self.resolution - 1)
        vertices = np.multiply(vertices, step)
        vertices += [self.min, self.min, self.min]
        return trimesh.Trimesh(vertices, triangles)

    def test_dataloader(self):
        return self.dataset.get_loader_val(shuffle=True)

    def on_test_start(self):
        torch.cuda.empty_cache()
        self.model_sp.eval()
        self.model_sc.eval()
        self.impli_fu.eval()

    def test_step(self, batch, batch_idx):
        path = os.path.normpath(batch['path'][0])
        export_path = os.path.join(self.output_dir, 'generation', path.split(os.sep)[-2], path.split(os.sep)[-1])
        if os.path.exists(export_path):
            print('Path exists - skip! {}'.format(export_path))
        else:
            logits = self.generate_mesh(batch)
            mesh = self.mesh_from_logits(logits)
            export_dir = os.path.join(self.output_dir, 'generation', path.split(os.sep)[-2])
            off_name = os.path.basename(path)
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)
            mesh.export(os.path.join(export_dir, off_name))

if __name__ == "__main__":
    hparams = parse_args()
    print(hparams)
    seed_everything(123)  # set random seed
    pl_system = ScodaGenerator(hparams)
    # ! any problem refer to https://pytorch-lightning.readthedocs.io/en/latest/advanced/model_parallel.html#ddp-optimizations
    # ! if encounter problem, change find_unused_parameters=True
    # ! for single gpu, set strategy=None and devices=1
    pl_trainer = Trainer(
        max_epochs=5,
        check_val_every_n_epoch=1,
        callbacks=[TQDMProgressBar(refresh_rate=1)],
        logger=False,
        enable_model_summary=True,
        accelerator='gpu',
        strategy=DDPStrategy(find_unused_parameters=False),
        devices=hparams.gpus, # gpu number
        num_nodes=1,
        num_sanity_val_steps=-1,
        precision=32  # 16 will be faster
    )
    pl_trainer.test(pl_system)
