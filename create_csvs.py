import pandas as pd

# bfw
bfw = pd.read_csv('data/bfw/bfw.csv')
bfw = bfw.rename(columns={
    'p1': 'path1',
    'p2': 'path2',
    'label': 'same',
    'vgg2': 'facenet-webface',
    'resnet50': 'facenet',
    'senet50': 'arcface'
})
bfw['same'] = bfw['same'].replace([1, 0], [True, False])

# rfw
rfw = pd.read_csv('data/rfw/rfw.csv')

dfs = {'bfw': bfw,
       'rfw': rfw
}
cos_sim_to_change = {
    'bfw': ['facenet-webface', 'arcface'],
    'rfw': ['facenet', 'facenet-webface']
}

for dataset, pretrained_models in cos_sim_to_change.items():
    for pretrained_model in pretrained_models:
        current_csv = pd.read_csv('similarities/' + pretrained_model + '_' + dataset + '_cosin_sim.csv')
        dfs[dataset][pretrained_model] = current_csv['cos_sim']

# save files
bfw.to_csv('data/bfw/bfw_w_sims.csv', index=False)
rfw.to_csv('data/rfw/rfw_w_sims.csv', index=False)
