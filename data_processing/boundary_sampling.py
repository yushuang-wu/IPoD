import trimesh
import numpy as np
import implicit_waterproofing as iw
import glob
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import os
import traceback

dataset = 'shapenet'
categ = 'chair'
root = f'/home/wuyushuang/Dataset/{dataset}_data/{categ}'
files = glob.glob(f'{root}/mesh/*.ply')

out_dir = f'{root}/boundary'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

def boundary_sampling(file_path):

    file_name = os.path.basename(file_path)
    off_path = file_path # actually is ply file
    out_file = f'{out_dir}/boundary{args.sigma}_{file_name[:-4]}.npz'

    # if os.path.exists(out_file):
    #     print('exists', out_file)
    #     return

    mesh = trimesh.load(off_path)
    points = mesh.sample(sample_num)

    boundary_points = points + args.sigma * np.random.randn(sample_num, 3)
    grid_coords = boundary_points.copy()
    grid_coords[:, 0], grid_coords[:, 2] = boundary_points[:, 2], boundary_points[:, 0]

    grid_coords = 2 * grid_coords

    occupancies = iw.implicit_waterproofing(mesh, boundary_points)[0]

    np.savez(out_file, points = boundary_points, occupancies = occupancies, grid_coords = grid_coords)
    print('Finished {}'.format(out_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run boundary sampling'
    )
    parser.add_argument('-sigma', type=float)

    args = parser.parse_args()

    sample_num = 100000

    print(mp.cpu_count(), len(files))
    p = Pool(mp.cpu_count())
    p.map(boundary_sampling, files)

    # for f in files:
    #     boundary_sampling(f)
    #     break
