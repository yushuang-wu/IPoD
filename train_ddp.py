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
import statistics
import models.local_model_ddp as model
import models.data.data_region as voxelized_data
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
import warnings

warnings.filterwarnings("ignore")


def parse_args():
    # python train.py -posed -dist 0.5 0.5 -std_dev 0.15 0.05 -res 32 -batch_size 40 -m
    parser = argparse.ArgumentParser(description='Run Model')

    parser.add_argument('-pointcloud', dest='pointcloud', action='store_true')
    parser.add_argument('-voxels', dest='pointcloud', action='store_false')
    parser.set_defaults(pointcloud=False)
    parser.add_argument('-exp_name', default='pretrain', type=str)
    parser.add_argument('-pc_samples', default=300, type=int)
    parser.add_argument('-dist',
                        '--sample_distribution',
                        default=[0.5, 0.5],
                        nargs='+',
                        type=float)
    parser.add_argument('-std_dev',
                        '--sample_sigmas',
                        default=[0.15, 0.015],
                        nargs='+',
                        type=float)
    parser.add_argument('-batch_size', default=30, type=int)
    parser.add_argument('-res', default=32, type=int)
    parser.add_argument('-gpus', default=2, type=int)
    parser.add_argument('-m', '--model', default='LocNet', type=str)
    parser.add_argument('-o', '--optimizer', default='Adam', type=str)

    parser.add_argument('-class_name', default='chair', type=str)
    parser.add_argument('-data_root', default='/home/wuyushuang/data', type=str)
    parser.add_argument('-num_sp_mesh_sample', default=2000, type=int)


    args = parser.parse_args()

    return args


class ScodaTrainDataset(Dataset):
    def __init__(self, train_sp_sup, train_sc_sup, train_sc_unsup):
        self.train_sp_sup_loader = train_sp_sup.get_loader()
        self.train_sc_sup_loader = train_sc_sup.get_loader()
        self.train_sc_unsup_loader = train_sc_unsup.get_loader()

    def __getitem__(self, index):
        return {
            'sp': next(self.train_sp_sup_loader),
            'sc': next(self.train_sc_sup_loader),
            'unsup': next(self.train_sc_unsup_loader)
        }

    def __len__(self):
        return len(self.train_sc_unsup_loader)


class ScodaValDataset(Dataset):
    def __init__(self, val_sc_sup):
        self.val_sc_sup = val_sc_sup
        self.num_batches = int(
            len(self.val_sc_sup) / self.val_sc_sup.batch_size)
        self.iter = self.val_sc_sup.get_loader_val().__iter__()

    def __getitem__(self, index):
        try:
            val_batch = self.iter.next()
        except:
            self.iter = self.val_sc_sup.get_loader_val().__iter__()
            val_batch = self.iter.next()
        return val_batch

    def __len__(self):
        return self.num_batches


class ScodaSystem(LightningModule):
    def __init__(self, hparams):
        super(ScodaSystem, self).__init__()
        self.save_hyperparameters(hparams)
        # exp name
        self.exp_name = 'i{}_dist-{}sigmas-{}v{}_m{}'.format(
            str(hparams.exp_name) + str(hparams.pc_samples),
            ''.join(str(e) + '_' for e in hparams.sample_distribution),
            ''.join(str(e) + '_' for e in hparams.sample_sigmas), hparams.res,
            hparams.model)
        self.exp_path = './experiments/{}/'.format(self.exp_name)
        os.makedirs(self.exp_path, exist_ok=True)
        # init models
        self.model_sp = model.ShapeNetPoints()
        self.model_sc = model.ScanNetPoints()
        self.impli_fu = model.ImplicitFunction()
        # training params
        self.batch_size = hparams.batch_size
        self.init_lr = 1e-5 if self.exp_name == 'Finetune' else 1e-4
        self.val_min = None
        # init datasets
        self.train_sp_sup = voxelized_data.VoxelizedShapeNet(
            mode='train',
            num_sp_mesh_sample=hparams.num_sp_mesh_sample, 
            data_root=hparams.data_root, 
            class_name=hparams.class_name,
            voxelized_pointcloud=hparams.pointcloud,
            pointcloud_samples=hparams.pc_samples,
            res=hparams.res,
            sample_distribution=hparams.sample_distribution,
            sample_sigmas=hparams.sample_sigmas,
            num_sample_points=50000,
            batch_size=hparams.batch_size,
            num_workers=hparams.batch_size)
        self.train_sc_sup = voxelized_data.VoxelizedScanNet(
            mode='train_sup',
            data_root=hparams.data_root, 
            class_name=hparams.class_name,
            voxelized_pointcloud=hparams.pointcloud,
            pointcloud_samples=hparams.pc_samples,
            res=hparams.res,
            sample_distribution=hparams.sample_distribution,
            sample_sigmas=hparams.sample_sigmas,
            num_sample_points=50000,
            batch_size=int(hparams.batch_size / 2),
            num_workers=int(hparams.batch_size / 2))
        self.train_sc_unsup = voxelized_data.VoxelizedScanNet(
            mode='train_unsup',
            data_root=hparams.data_root, 
            class_name=hparams.class_name,
            voxelized_pointcloud=hparams.pointcloud,
            pointcloud_samples=hparams.pc_samples,
            res=hparams.res,
            sample_distribution=hparams.sample_distribution,
            sample_sigmas=hparams.sample_sigmas,
            num_sample_points=50000,
            batch_size=hparams.batch_size,
            num_workers=hparams.batch_size)
        self.val_sc_sup = voxelized_data.VoxelizedScanNet(
            mode='val',
            data_root=hparams.data_root, 
            class_name=hparams.class_name,
            voxelized_pointcloud=hparams.pointcloud,
            pointcloud_samples=hparams.pc_samples,
            res=hparams.res,
            sample_distribution=hparams.sample_distribution,
            sample_sigmas=hparams.sample_sigmas,
            num_sample_points=50000,
            batch_size=1,
            num_workers=1)
        ### for adversarial learning ###
        # init loss
        self.mse_loss = nn.MSELoss()
        self.max, self.min, self.threshold = 0.5, -0.5, 0.5
        self.resolution1, self.resolution2 = 64, 32
        res1, res2 = self.resolution1, self.resolution2

        batch_points = res1 * res1 * res1
        grid_points = iw.create_grid_points_from_bounds(
            self.min, self.max, res1)
        grid_points[:,
                    0], grid_points[:,
                                    2] = grid_points[:,
                                                     2], grid_points[:,
                                                                     0].copy()
        grid_coords = 2 * grid_points  # -0) /1
        grid_coords = torch.from_numpy(grid_coords).to(self.device,
                                                       dtype=torch.float)
        grid_coords = torch.reshape(grid_coords,
                                    (len(grid_points), 3)).to(self.device)
        self.grid_coords_f = grid_coords.repeat(hparams.batch_size, 1, 1)

        batch_points = res2 * res2 * res2
        grid_points = iw.create_grid_points_from_bounds(
            self.min, self.max, res2)
        grid_points[:,
                    0], grid_points[:,
                                    2] = grid_points[:,
                                                     2], grid_points[:,
                                                                     0].copy()
        grid_coords = 2 * grid_points  # -0) /1
        grid_coords = torch.from_numpy(grid_coords).to(self.device,
                                                       dtype=torch.float)
        grid_coords = torch.reshape(grid_coords,
                                    (len(grid_points), 3)).to(self.device)
        self.grid_coords_c = grid_coords.repeat(hparams.batch_size, 1, 1)

    def configure_optimizers(self):
        # filter parameters
        net_params = list(self.model_sp.parameters()) + list(
            self.model_sc.parameters()) + list(self.impli_fu.parameters())
        # optimizer
        self.net_opt = optim.Adam(net_params, lr=self.init_lr)
        return self.net_opt

    def train_dataloader(self):
        return DataLoader(ScodaTrainDataset(self.train_sp_sup,
                                            self.train_sc_sup,
                                            self.train_sc_unsup),
                          batch_size=None)

    def val_dataloader(self):
        return DataLoader(ScodaValDataset(self.val_sc_sup), batch_size=None)

    def forward(self, batch, mode):
        self.impli_fu.train()
        if mode == 'sp':
            self.model_sp.train()
            self.model_sc.eval()
            return self.compute_loss(batch, mode)
        elif mode == 'sc':
            self.model_sp.eval()
            self.model_sc.train()
            return self.compute_loss(batch, mode)
        elif mode == 'unsup':
            self.model_sp.eval()
            self.model_sc.train()
            return self.compute_loss(batch, mode)
        else:
            raise NotImplementedError

    def compute_loss(self, batch, mode):
        device = self.device
        b = batch.get('inputs').shape[0]
        if mode == 'sp':
            occ = batch.get('occupancies').to(device)
            p = batch.get('grid_coords').to(device)
            inputs = batch.get('inputs').to(device)
            features = self.model_sp(p, inputs)
            logits = self.impli_fu(features)
            loss_i = F.binary_cross_entropy_with_logits(logits, occ, reduction='mean')
            loss1 = loss_i.sum(-1).mean()
        elif mode == 'sc':
            occ = batch.get('occupancies').to(device)
            p = batch.get('grid_coords').to(device)
            inputs = batch.get('inputs').to(device)
            features_sp = self.model_sp(p, inputs)
            features = self.model_sc(p, inputs, features_sp)
            logits = self.impli_fu(features)
            loss_i = F.binary_cross_entropy_with_logits(logits, occ, reduction='mean')
            loss1 = loss_i.sum(-1).mean()
        elif mode == 'unsup':
            p = self.grid_coords_c[:b].to(device) # TODO add to(device) for ddp
            inputs1 = batch.get('input0').to(device)
            inputs2 = batch.get('input2').to(device)
            features_sp1 = self.model_sp(p, inputs1).detach()
            features_sp2 = self.model_sp(p, inputs2)
            features1 = self.model_sc(p, inputs1, features_sp1).detach()
            features2 = self.model_sc(p, inputs2, features_sp2)
            logit1 = self.impli_fu(features1).detach()
            logit2 = self.impli_fu(features2)
            res, thresh = self.resolution2, self.threshold/2.0
            logit1 = logit1.view(b, res, res, res).flatten()
            logit2 = logit2.view(b, res, res, res).flatten()
            mask = (logit1.ge(thresh) | logit1.le(0.1)).cuda().float()
            loss_i = F.binary_cross_entropy_with_logits(logit2, logit1.ge(thresh).float(), reduction='none') * mask
            loss1 = loss_i.mean()
        else:
            raise NotImplementedError
        return loss1

    def training_step(self, batch, batch_idx):
        loss_sp_sup = self(batch['sp'], 'sp')
        loss_sc_sup = self(batch['sc'], 'sc')
        loss_sc_unsup = self(batch['unsup'], 'unsup')
        loss = (loss_sp_sup + loss_sc_sup + loss_sc_unsup)/3.0
        self.log('train_step/lr', self.net_opt.param_groups[0]['lr'], on_step=True, on_epoch=False)
        self.log('train_step/loss', loss.item(), on_step=True, on_epoch=False)
        self.log('train_step/loss_sp_sup', loss_sp_sup.item(), on_step=True, on_epoch=False)
        self.log('train_step/loss_sc_sup', loss_sc_sup.item(), on_step=True, on_epoch=False)
        self.log('train_step/loss_sc_unsup', loss_sc_unsup.item(), on_step=True, on_epoch=False)
        return {'loss': loss, 'loss_sp_sup':loss_sp_sup, 'loss_sc_sup':loss_sc_sup, 'loss_sc_unsup':loss_sc_unsup}

    def training_epoch_end(self, outputs):
        losses = torch.stack([x['loss'] for x in outputs])
        mean_loss = all_gather_ddp_if_available(losses).mean()
        losses_sp_sup = torch.stack([x['loss_sp_sup'] for x in outputs])
        mean_loss_sp_sup = all_gather_ddp_if_available(losses_sp_sup).mean()
        losses_sc_sup = torch.stack([x['loss_sc_sup'] for x in outputs])
        mean_loss_sc_sup = all_gather_ddp_if_available(losses_sc_sup).mean()
        losses_sc_unsup = torch.stack([x['loss_sc_unsup'] for x in outputs])
        mean_loss_sc_unsup = all_gather_ddp_if_available(losses_sc_unsup).mean()
        if self.trainer.is_global_zero:
            print(f"\ntrain epoch: {self.current_epoch}, v_num: {self.logger.version}, ", end='')
            print(f"lr: {self.net_opt.param_groups[0]['lr']:.5f}, ", end='')
            print(f'loss: {mean_loss:.5f}, ', end='')
            print(f'loss_sp_sup {mean_loss_sp_sup:.5f}, ', end='')
            print(f'loss_sc_sup {mean_loss_sc_sup:.5f}, ', end='')
            print(f'loss_sc_unsup {mean_loss_sc_unsup:.5f}.', end='\n')
        self.log('train_epoch/lr', self.net_opt.param_groups[0]['lr'], on_step=False, on_epoch=True)
        self.log('train_epoch/loss', mean_loss, on_step=False, on_epoch=True)
        self.log('train_epoch/loss_sp_sup', mean_loss_sp_sup, on_step=False, on_epoch=True)
        self.log('train_epoch/loss_sc_sup', mean_loss_sc_sup, on_step=False, on_epoch=True)
        self.log('train_epoch/loss_sc_unsup', mean_loss_sc_unsup, on_step=False, on_epoch=True)

    def mesh_from_logits(self, logits):
        logits = np.reshape(logits, (self.resolution1,) * 3)

        # padding to ba able to retrieve object close to bounding box bondary
        logits = np.pad(logits, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=0)
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        vertices, triangles = mcubes.marching_cubes(logits, threshold)

        # remove translation due to padding
        vertices -= 1

        # rescale to original scale
        step = (self.max - self.min) / (self.resolution1 - 1)
        vertices = np.multiply(vertices, step)
        vertices += [self.min, self.min, self.min]

        return trimesh.Trimesh(vertices, triangles)

    def evaluate(self, batch):
        device = self.device
        n_points = 100000
        b = batch.get('inputs').shape[0]

        grid_coords = self.grid_coords_f[:b].to(device) # TODO add to(device) for ddp
        inputs = batch.get('inputs').to(device)
        features_sp = self.model_sp(grid_coords, inputs)
        features = self.model_sc(grid_coords, inputs, features_sp)
        logits = self.impli_fu(features)
        logits = logits.cpu().numpy()

        IoU_list, Chamfer_list = [], []
        paths = batch.get('path')
        for i, path in enumerate(paths):
            root, file_name = '/'.join(path.split('/')[:-1]), path.split('/')[-1]
            mesh_pred = self.mesh_from_logits(logits[i])
            mesh_gt = trimesh.load(f'{root}/mesh/{file_name}')

            try:
                eval = eval_mesh(mesh_pred, mesh_gt, self.min, self.max, n_points)
            except:
                continue
            IoU_list += [eval['iou']]
            Chamfer_list += [eval['chamfer_l2']]

        if len(Chamfer_list) == 0:
            print('Bad Validation, Return Empty List!!!')
            return 0., 100.

        return sum(IoU_list)/len(IoU_list), sum(Chamfer_list)/len(Chamfer_list)

    def on_validation_start(self):
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        val_loss = self.compute_loss(batch, 'sc')
        self.log('val_step/val_loss', val_loss.item(), on_step=True, on_epoch=False)
        if self.current_epoch % 1 == 0:
        # if self.current_epoch % 1 == 0: #! used for debugging
            if self.current_epoch > 0:
                iou, cfd = self.evaluate(batch)
                self.log('val_step/iou', iou, on_step=True, on_epoch=False)
                self.log('val_step/cfd', cfd, on_step=True, on_epoch=False)
                return {'val_loss': val_loss, 'iou': iou, 'cfd': cfd}
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        val_losses = torch.stack([x['val_loss'] for x in outputs])
        mean_val_loss = all_gather_ddp_if_available(val_losses).mean()
        if self.trainer.is_global_zero: # ! do it in the main process
            print(f"\n\nval epoch: {self.current_epoch}, v_num: {self.logger.version}, ", end='')
            print(f'val_loss: {mean_val_loss:.5f}', end='')
        self.log('val_epoch/val_loss', mean_val_loss, on_step=False, on_epoch=True)
        if self.current_epoch % 1 == 0:
        # if self.current_epoch % 1 == 0: #! used for debugging
            if self.current_epoch > 0:
                iou = torch.from_numpy(np.stack([x['iou'] for x in outputs])).to(self.device)
                cfd = torch.from_numpy(np.stack([x['cfd'] for x in outputs])).to(self.device)
                mean_iou = all_gather_ddp_if_available(iou).mean()
                mean_cfd = all_gather_ddp_if_available(cfd).mean()
                if self.trainer.is_global_zero: # ! do it in the main process
                    print(f", iou: {mean_iou:.5f}, cfd: {mean_cfd:.5f}", end='')
                self.log('val_epoch/iou', mean_iou, on_step=False, on_epoch=True)
                self.log('val_epoch/cfd', mean_cfd, on_step=False, on_epoch=True)
            if self.trainer.is_global_zero: # ! do it in the main process
                if self.val_min is None:
                    self.val_min = mean_val_loss
                if mean_val_loss < self.val_min:
                    self.val_min = mean_val_loss
                    save_dir = os.path.join(self.exp_path, 'records', f'version_{self.logger.version}')
                    os.makedirs(save_dir, exist_ok=True)
                    # TODO modified for ddp save np in different version dir
                    for path in glob(os.path.join(save_dir, 'val_min=*')):
                        os.remove(path)
                    np.save(os.path.join(save_dir, f'val_min={self.current_epoch}.npy'),np.array([self.current_epoch, mean_val_loss.cpu().numpy()], dtype=np.float32))
        if self.trainer.is_global_zero: # ! do it in the main process
            print('.', end='\n')

if __name__ == "__main__":
    ckpt_path=None # ! resume from checkpoint, None for training from scratch
    hparams = parse_args()
    print(hparams)
    seed_everything(123)  # set random seed
    pl_system = ScodaSystem(hparams)
    os.makedirs(os.path.join(pl_system.exp_path, 'checkpoints'), exist_ok=True)
    # ! only keep the latest experiment's ckpts (latest experiment version)
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(pl_system.exp_path, 'checkpoints'),
        filename='{epoch}',
        save_weights_only=False,
        every_n_epochs=1,
        # every_n_epochs=1, #! used for debugging
        save_on_train_epoch_end=True,
        save_top_k=-1)
    pl_callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]
    os.makedirs(os.path.join(pl_system.exp_path, 'logs'), exist_ok=True)
    pl_logger = TensorBoardLogger(save_dir=os.path.join(pl_system.exp_path, 'logs'),
                                  name=None,
                                  default_hp_metric=False)
    # ! any problem refer to https://pytorch-lightning.readthedocs.io/en/latest/advanced/model_parallel.html#ddp-optimizations
    # ! if encounter problem, change find_unused_parameters=True
    # ! for single gpu, set strategy=None and devices=1
    pl_trainer = Trainer(
        max_epochs=101,
        check_val_every_n_epoch=1,
        callbacks=pl_callbacks,
        logger=pl_logger,
        enable_model_summary=True,
        accelerator='gpu',
        strategy=DDPStrategy(find_unused_parameters=False),
        devices=hparams.gpus, # gpu number
        num_nodes=1,
        num_sanity_val_steps=-1,
        precision=32  # 16 will be faster
    )
    # training model
    pl_trainer.fit(pl_system, ckpt_path=ckpt_path)
