import os
import glob
import multiprocessing as mp
from multiprocessing import Pool
import trimesh
import math as m
import numpy as np

import re
import time
from plyfile import PlyData, PlyElement

dataset = 'shapenet'
categ = 'chair'
root = f'/home/wuyushuang/data/{dataset}_data/{categ}'
files = glob.glob(f'{root}/mesh_raw/*')

out_dir = f'{root}/mesh'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

out_dir = f'{root}/pcd'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

def rotate(path):

    inst_name = os.path.basename(path)[:-4]#.split('_m')[0]
    pc_name = inst_name + '.xyz.ply'
    output_pc_name = inst_name + '.xyz.ply'
    output_mesh_name = inst_name + '.ply'

    input_mesh = path
    input_pc = f'{root}/pcd_raw/{pc_name}'

    output_mesh = f'{root}/mesh/{output_mesh_name}'
    output_pc = f'{root}/pcd/{output_pc_name}'

    # if os.path.exists(output_mesh):
    #     print('skip')
    #     return

    theta = m.pi
    c, s = m.cos(theta), m.sin(theta)
    tran_matrix = np.array([[ c, 0, s, 0],
                            [ 0, 1, 0, 0],
                            [-s, 0, c, 0],
                            [ 0, 0, 0, 1]])
    rotate_pc = np.matrix([[ c, 0, s],
                           [ 0, 1, 0],
                           [-s, 0, c]])

    mesh = trimesh.load(input_mesh, process=False)
    mesh.apply_transform(tran_matrix)

    total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
    centers = (mesh.bounds[1] + mesh.bounds[0]) / 2.0

    mesh.apply_translation(-centers)
    mesh.apply_scale(1/total_size)
    mesh.export(output_mesh)
    
    plydata = PlyData.read(input_pc)
    vert_x = plydata['vertex']['x']
    vert_y = plydata['vertex']['y']
    vert_z = plydata['vertex']['z']
    xyz = np.vstack([vert_x, vert_y, vert_z]).T
    xyz = xyz.dot(rotate_pc.T) # rotate first
    xyz = (xyz - centers) / total_size
    pc = trimesh.Trimesh(vertices=xyz, faces=[])
    pc.export(output_pc)
 
    print('Finished {}'.format(path))

print(mp.cpu_count(), len(files))
p = Pool(mp.cpu_count())
p.map(rotate, files)

# for f in files:
#     rotate(f)
#     break
    
