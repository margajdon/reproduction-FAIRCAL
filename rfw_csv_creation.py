import os
import numpy as np
import pandas as pd

def extract_pairs_txt(file_path, group):
    df = pd.read_csv(file_path, header=None)[0].str.split('\t', expand=True)
    df.columns = ['id1', 'num1', 'id2', 'num2']

    # cond is a mask that filters whether the pair belongs ot the same face
    # 1: same/genuine face, 0: different/ungenuine face
    cond = df['num2'].isnull()

    df.loc[cond, 'num2'] = df.loc[cond, 'id2']
    df.loc[cond, 'id2'] = df.loc[cond, 'id1']
    df['ethnicity'] = group
    df['pair'] = 'Imposter'
    df['pair'].loc[cond] = 'Genuine'
    df['same'] = cond
    df['fold'] = df.index.map(lambda x: x // 600)
    df['facenet'] = np.nan
    df['facenet-webface'] = np.nan
    df['arcface'] = np.nan
    return df

def get_rfw_df():
    data_folder = "data"
    dataset = 'rfw'
    data_type = 'txts'
    groups = ['African', 'Asian', 'Caucasian', 'Indian']

    df_list = []
    for group in groups:
        for path, subdirs, files in os.walk(os.path.join(data_folder, dataset, data_type, group)):
            for f in files:
                if f.endswith('_pairs.txt'):
                    df_list.append(extract_pairs_txt(os.path.join(path, f), group))

    return pd.concat(df_list, ignore_index=True)

if __name__ == '__main__':
    rfw_df = get_rfw_df()
    rfw_df.to_csv('data/rfw/rfw.csv', index=False)
