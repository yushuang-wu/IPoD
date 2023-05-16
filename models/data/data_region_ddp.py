from __future__ import division
import sys
sys.path.append('data_processing')

from torch.utils.data import Dataset
from scipy.spatial import cKDTree as KDTree
import os
import numpy as np
import pickle
import imp
import trimesh
import torch
import random
import trimesh
from plyfile import PlyData
import implicit_waterproofing as iw

# num_sp_mesh_sample = 5000

class InfDataloader:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)


class VoxelizedShapeNet(Dataset):

    def __init__(self, mode, res = 32,  voxelized_pointcloud = False, 
                 pointcloud_samples = 3000, num_sp_mesh_sample = 2000, 
                 data_root = '', class_name = '',
                 batch_size = 64, num_sample_points = 1024, num_workers = 12, 
                 sample_distribution = [1], sample_sigmas = [0.015], **kwargs):

        data_path = f'{data_root}/shapenet_data/{class_name}'

        self.sample_distribution = np.array(sample_distribution)
        self.sample_sigmas = np.array(sample_sigmas)

        assert np.sum(self.sample_distribution) == 1
        assert np.any(self.sample_distribution < 0) == False
        assert len(self.sample_distribution) == len(self.sample_sigmas)

        self.path = data_path
        self.mode = mode
        
        data = os.listdir(f'{data_path}/region_split') #mesh
        random.shuffle(data)
        self.data = random.sample(data, num_sp_mesh_sample)
        print('No. of ShapeNet Sample:', len(self.data))

        self.res = res

        self.num_sample_points = num_sample_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.voxelized_pointcloud = voxelized_pointcloud
        self.pointcloud_samples = pointcloud_samples

        # compute number of samples per sampling method
        self.num_samples = np.rint(self.sample_distribution * num_sample_points).astype(np.uint32)

        bb_min, bb_max = -0.5, 0.5
        self.grid_points = iw.create_grid_points_from_bounds(bb_min, bb_max, res)
        self.kdtree = KDTree(self.grid_points)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path_root = self.path
        file_id = self.data[idx].split('.')[0]

        point_cloud_path = f'{path_root}/pcd/{file_id}.xyz.ply'
        plydata = PlyData.read(point_cloud_path)
        vert_x = plydata['vertex']['x']
        vert_y = plydata['vertex']['y']
        vert_z = plydata['vertex']['z']
        point_cloud = np.vstack([vert_x, vert_y, vert_z]).T

        all_indices = list(range(point_cloud.shape[0]))
        indices = all_indices
        if self.mode != 'test':
            region_split_path = f'{path_root}/region_split/{file_id}.xyz.npy'
            region_split = np.load(region_split_path)
            sample_num = random.choice([4,5,6,7])
            split = random.sample([i for i in range(8)], sample_num)
            indices = []
            for i in split:
                indices += np.where(region_split==i)[0].tolist()
            if len(indices) > self.pointcloud_samples:
                indices = random.sample(indices, self.pointcloud_samples)

        point_cloud = point_cloud[indices]

        occupancies = np.zeros(len(self.grid_points), dtype=np.int8)
        _, idx = self.kdtree.query(point_cloud)
        occupancies[idx] = 1
        input = np.reshape(occupancies, (self.res,)*3)

        points = []
        coords = []
        occupancies = []

        for i, num in enumerate(self.num_samples):
            boundary_samples_path = f'{path_root}/boundary/boundary{self.sample_sigmas[i]}_{file_id}.npz'
            boundary_samples_npz = np.load(boundary_samples_path)
            boundary_sample_points = boundary_samples_npz['points']
            boundary_sample_coords = boundary_samples_npz['grid_coords']
            boundary_sample_occupancies = boundary_samples_npz['occupancies']
            subsample_indices = np.random.randint(0, len(boundary_sample_points), num)
            points.extend(boundary_sample_points[subsample_indices])
            coords.extend(boundary_sample_coords[subsample_indices])
            occupancies.extend(boundary_sample_occupancies[subsample_indices])

        assert len(points) == self.num_sample_points
        assert len(occupancies) == self.num_sample_points
        assert len(coords) == self.num_sample_points

        # voxelized_path = f'{path_root}/voxelized_pt10w_res32/voxel_{file_id}.npy'
        # voxelized_gt = np.unpackbits(np.load(voxelized_path))
        # voxelized_gt = np.reshape(voxelized_gt, (self.res,)*3).astype(np.float32)

        return {'grid_coords': np.array(coords, dtype=np.float32), 
                'occupancies': np.array(occupancies, dtype=np.float32),
                'points': np.array(points, dtype=np.float32), 
                'inputs': np.array(input, dtype=np.float32), 
                'path': f'{path_root}/{file_id}'}# 'voxelized_gt': voxelized_gt,

    def get_loader(self, shuffle=True):

        dataloader = torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn)
        return InfDataloader(dataloader)
    
    def get_loader_val(self, shuffle=True):

        dataloader = torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn)
        return dataloader

    def worker_init_fn(self, worker_id):

        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)



class VoxelizedScanNet(Dataset):

    def __init__(self, mode, res = 32,  voxelized_pointcloud = False, 
                 pointcloud_samples = 3000, data_root = '', class_name = '',
                 batch_size = 64, num_sample_points = 1024, num_workers = 12, 
                 sample_distribution = [1], sample_sigmas = [0.015], **kwargs):

        data_path = f'{data_root}/scannet_data/{class_name}'

        self.sample_distribution = np.array(sample_distribution)
        self.sample_sigmas = np.array(sample_sigmas)

        assert np.sum(self.sample_distribution) == 1
        assert np.any(self.sample_distribution < 0) == False
        assert len(self.sample_distribution) == len(self.sample_sigmas)

        self.path = data_path
        self.mode = mode

        pc_all = os.listdir(f'{data_path}/pcd')
        pc_sup = os.listdir(f'{data_path}/mesh')
        pc_unsup = [i for i in pc_all if i not in pc_sup]

        self.sup_list = pc_sup
        import json
        with open(f'{data_path}/split_list.json', 'r') as f:
            train_list, test_list = json.load(f)
        if mode == 'train_sup':
            self.data = [d for d in pc_sup if d in train_list]
        elif mode == 'val':
            self.data = [d for d in pc_sup if d in test_list]
        elif mode == 'train_unsup':
            self.data = pc_unsup
            print(f'NO. of sup: {len(pc_sup)}, NO. of unsup: {len(pc_unsup)}')
            print(f'NO. of train: {len(train_list)}, NO. of val: {len(test_list)}')
        else:
            self.data = pc_unsup
        random.shuffle(self.data)

        self.res = res

        self.num_sample_points = num_sample_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.voxelized_pointcloud = voxelized_pointcloud
        self.pointcloud_samples = pointcloud_samples

        # compute number of samples per sampling method
        self.num_samples = np.rint(self.sample_distribution * num_sample_points).astype(np.uint32)

        bb_min, bb_max = -0.5, 0.5
        self.grid_points = iw.create_grid_points_from_bounds(bb_min, bb_max, res)
        self.kdtree = KDTree(self.grid_points)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path_root = self.path
        file_name = self.data[idx]

        point_cloud_path = f'{path_root}/pcd/{file_name}'
        plydata = PlyData.read(point_cloud_path)
        vert_x = plydata['vertex']['x']
        vert_y = plydata['vertex']['y']
        vert_z = plydata['vertex']['z']
        point_cloud = np.vstack([vert_x, vert_y, vert_z]).T

        all_indices = list(range(point_cloud.shape[0]))
        indices0 = all_indices
        if self.mode == 'test' or self.mode == 'val':
            indices1 = all_indices
            indices2 = all_indices
        else:
            region_split_path = f'{path_root}/region_split/{file_name[:-4]}.npy'
            region_split = np.load(region_split_path)
            split1 = random.sample([i for i in range(8)], 4)
            split2 = [i for i in range(8) if i not in split1]
            indices1 = []
            for i in split1:
                indices1 += np.where(region_split==i)[0].tolist()
            indices2 = []
            for i in split2:
                indices2 += np.where(region_split==i)[0].tolist()

            if len(all_indices) > self.pointcloud_samples:
                indices0 = random.sample(all_indices, self.pointcloud_samples)
            if len(indices1) > self.pointcloud_samples:
                indices1 = random.sample(indices1, self.pointcloud_samples)
            if len(indices2) > self.pointcloud_samples:
                indices2 = random.sample(indices2, self.pointcloud_samples)

        point_cloud0 = point_cloud[indices0]
        point_cloud1 = point_cloud[indices1]
        point_cloud2 = point_cloud[indices2]
        
        occupancies0 = np.zeros(len(self.grid_points), dtype=np.int8)
        _, idx = self.kdtree.query(point_cloud0)
        occupancies0[idx] = 1
        input0 = np.reshape(occupancies0, (self.res,)*3)

        occupancies1 = np.zeros(len(self.grid_points), dtype=np.int8)
        _, idx = self.kdtree.query(point_cloud1)
        occupancies1[idx] = 1
        input1 = np.reshape(occupancies1, (self.res,)*3)

        occupancies2 = np.zeros(len(self.grid_points), dtype=np.int8)
        _, idx = self.kdtree.query(point_cloud2)
        occupancies2[idx] = 1
        input2 = np.reshape(occupancies2, (self.res,)*3)

        points = []
        coords = []
        occupancies = []
        voxelized_gt = np.array([])

        if self.mode == 'train_sup' or self.mode == 'val':
            for i, num in enumerate(self.num_samples):
                boundary_samples_path = f'{path_root}/boundary/boundary{self.sample_sigmas[i]}_{file_name[:-4]}.npz'
                boundary_samples_npz = np.load(boundary_samples_path)
                boundary_sample_points = boundary_samples_npz['points']
                boundary_sample_coords = boundary_samples_npz['grid_coords']
                boundary_sample_occupancies = boundary_samples_npz['occupancies']
                subsample_indices = np.random.randint(0, len(boundary_sample_points), num)
                points.extend(boundary_sample_points[subsample_indices])
                coords.extend(boundary_sample_coords[subsample_indices])
                occupancies.extend(boundary_sample_occupancies[subsample_indices])

            assert len(points) == self.num_sample_points
            assert len(occupancies) == self.num_sample_points
            assert len(coords) == self.num_sample_points

            # voxelized_path = f'{path_root}/voxelized_pt10w_res32/{file_name[:-4]}.npy'
            # voxelized_gt = np.unpackbits(np.load(voxelized_path))
            # voxelized_gt = np.reshape(voxelized_gt, (self.res,)*3).astype(np.float32)

        return {'grid_coords': np.array(coords, dtype=np.float32), 
                'occupancies': np.array(occupancies, dtype=np.float32),
                'points': np.array(points, dtype=np.float32), 
                'input0': np.array(input0, dtype=np.float32),
                'inputs': np.array(input1, dtype=np.float32), 
                'input2': np.array(input2, dtype=np.float32), 
                'path': f'{path_root}/{file_name}'}# 'voxelized_gt': voxelized_gt, 

    def get_loader(self, shuffle=True):

        dataloader = torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn)
        return InfDataloader(dataloader)

    def get_loader_val(self, shuffle =True):

        dataloader = torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, 
                num_workers=self.num_workers, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn)
        return dataloader

    def worker_init_fn(self, worker_id):

        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)
