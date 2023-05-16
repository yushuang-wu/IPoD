from sklearn.cluster import KMeans
import numpy as np
import os, glob
import multiprocessing as mp
from multiprocessing import Pool
from plyfile import PlyData

dataset = 'shapenet'
categ = 'lamp'
root = f'/home/wuyushuang/Dataset/{dataset}_data/{categ}'
files = glob.glob(f'{root}/pcd/*.ply')

out_dir = f'{root}/region_split'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

def cluster(file_path):

    name_idx = -4 if dataset=='scannet' else -8
    file_name = os.path.basename(file_path)
    out_path = f'{out_dir}/{file_name[:name_idx]}.xyz.npy'

    # if os.path.exists(out_path):
    #     print('exists', out_path)
    #     return

    plydata = PlyData.read(file_path)
    vert_x = plydata['vertex']['x']
    vert_y = plydata['vertex']['y']
    vert_z = plydata['vertex']['z']
    point_cloud = np.vstack([vert_x, vert_y, vert_z]).T

    kmeans = KMeans(n_clusters=8, random_state=0).fit(point_cloud)
    region_indices = kmeans.labels_

    np.save(out_path, region_indices)
    print('saved:', out_path)
    

if dataset == 'shapenet':
    import random
    random.shuffle(files)
    files = random.sample(files, 500)


print(mp.cpu_count(), len(files))
p = Pool(8)#mp.cpu_count())
p.map(cluster, files)

# for f in files:
#     cluster(f)
#     break