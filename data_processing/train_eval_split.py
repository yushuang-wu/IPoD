import json, os, random, glob

dataset = 'scannet'
categ = 'lamp'
root = f'/home/wuyushuang/data/{dataset}_data/{categ}'
files = glob.glob(f'{root}/pcd/*.ply')

pc_all = os.listdir(f'{root}/pcd/')
pc_sup = os.listdir(f'{root}/mesh/')
pc_unsup = [i for i in pc_all if i not in pc_sup]

train_scenes_list = '/home/wuyushuang/data/our_train_split.txt'
test_scenes_list = '/home/wuyushuang/data/our_test_split.txt'

with open(train_scenes_list, 'r') as f:
    train_scenes = f.readlines()
    train_scenes = [s.strip() for s in train_scenes]
with open(test_scenes_list, 'r') as f:
    test_scenes = f.readlines()
    test_scenes = [s.strip() for s in test_scenes]

print(len(train_scenes), train_scenes[:4])

data_train, data_test = [], []
for pc in pc_sup:
    scene = ('_').join(pc.split('_')[:2])
    if scene in test_scenes:
        data_test += [pc]
    else:
        data_train += [pc]
        if scene not in train_scenes:
            print('wrong')

if categ == 'bed':
    data_train = random.sample(data_train, 18)
    data_test = [i for i in pc_sup if i not in data_train]
if categ == 'lamp':
    data_train = random.sample(data_train, 10)
    data_test = [i for i in pc_sup if i not in data_train]
print(len(data_train), len(data_test))

with open(f'{root}/split_list.json', 'w') as f:
    json.dump([data_train, data_test], f)