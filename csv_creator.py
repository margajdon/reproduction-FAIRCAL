import os
import numpy as np
import pandas as pd
import pickle
import time


class CsvCreator:
    @staticmethod
    def extract_pairs_txt(file_path, group):
        """
        This method uses the text files of the RFW dataset to create a template file with all the different
        images pairs and their respective features for a given group.

        The features created are the following:
        'id1': Folder/person id 1
        'num1': Picture number 1
        'id2': Folder/person id 2
        'num2': Picture number 2,
        'ethnicity': Ethnicity (African, Asian, Caucasian or Indian)
        'pair': Descriptor (Genuine or Imposter)
        'same': Boolean
        'fold': Split fold
        'facenet': Cosine similarity of the image pair using facenet
        'facenet-webface': Cosine similarity of the image pair using facenet-webface
        'arcface': Cosine similarity of the image pair using arcface
        """
        df = pd.read_csv(file_path, header=None)[0].str.split('\t', expand=True)
        df.columns = ['id1', 'num1', 'id2', 'num2']

        # cond is a condition mask that filters whether the pair belongs to the same face
        # 1: same/genuine face, 0: different/imposter face
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

    @staticmethod
    def get_rfw_df():
        """
        This method creates a single dataframe with all the groups using the extract_pairs_txt method.

        More information about the features generated can be found in the extract_pairs_txt method.
        """

        data_folder = "data"
        dataset = 'rfw'
        data_type = 'txts'
        groups = ['African', 'Asian', 'Caucasian', 'Indian']

        df_list = []
        for group in groups:
            for path, subdirs, files in os.walk(os.path.join(data_folder, dataset, data_type, group)):
                for f in files:
                    if f.endswith('_pairs.txt'):
                        df_list.append(CsvCreator.extract_pairs_txt(os.path.join(path, f), group))

        return pd.concat(df_list, ignore_index=True)

    @staticmethod
    def remove_misclassified_data(df):
        """
        This method is used to remove the images of 3 individuals that were placed in the wrong group from the input
        dataframe.
        """
        samples_to_remove = {
            'm.0gc2xf9': 'African',
            'm.0z08d8y': 'Asian',
            'm.05xpnv': 'African',
        }
        for k, v in samples_to_remove.items():
            remove_cond = (df['id1'] == k) & (df['ethnicity'] == v)
            df = df[~remove_cond].reset_index(drop=True)
        return df

    @staticmethod
    def set_cosine_columns_to_nan(df):
        """
        Helper method to set the 3 cosine similarity columns for the models to NAN to ensure that new data is
        generated.
        """
        for c in ['facenet', 'facenet-webface', 'arcface']:
            df[c] = np.nan
        return df

    @staticmethod
    def load_and_prep_bfw():
        """
        This method reads the BFW csv dataset and performs cleaning on the dataframe before returning it.
        """
        # bfw
        bfw = pd.read_csv('data/bfw/bfw.csv').drop(columns=['vgg16', 'resnet50', 'senet50'])
        bfw = bfw.rename(columns={
            'p1': 'path1',
            'p2': 'path2',
            'label': 'same'
        })
        bfw['unique_key'] = bfw['path1'].astype(str) + '_' + bfw['path2'].astype(str)
        assert bfw['unique_key'].unique().shape[0] == bfw.shape[0]
        bfw['same'] = bfw['same'].replace([1, 0], [True, False])
        bfw = CsvCreator.set_cosine_columns_to_nan(bfw)
        return bfw

    @staticmethod
    def load_and_prep_rfw():
        """
        This method loads the RFW csv dataset as a dataframe, creates identifiers for the image ids and returns the
        dataframe.
        """
        # rfw
        rfw = pd.read_csv('data/rfw/rfw.csv')
        rfw = CsvCreator.remove_misclassified_data(rfw)
        rfw['image_id_1_clean'] = rfw['id1'].astype(str) + '_' + rfw['num1'].astype(str)
        rfw['image_id_2_clean'] = rfw['id2'].astype(str) + '_' + rfw['num2'].astype(str)
        rfw['unique_key'] = rfw['image_id_1_clean'].astype(str) + '_' + rfw['image_id_2_clean'].astype(str)
        assert rfw['unique_key'].unique().shape[0] == rfw.shape[0]
        rfw = CsvCreator.set_cosine_columns_to_nan(rfw)
        return rfw

    @staticmethod
    def clean_cosine_sim_data(df, dataset):
        """
        This method is used to create a unique key column for either the RFW or BFW datasets.
        """
        if dataset == 'bfw':
            df['unique_key'] = df['p1'].astype(str) + '_' + df['p2'].astype(str)
        else:
            df = CsvCreator.remove_misclassified_data(df)
            df['unique_key'] = df['image_id_1_clean'].astype(str) + '_' + df['image_id_2_clean'].astype(str)
        assert df['unique_key'].unique().shape[0] == df.shape[0]
        return df


def create_rfw_csv_template():
    """
    This function is used to create a csv template for the RFW dataset.

    It first generates the dataframe using the get_rfw_df method of the CsvCreator class and then stores the data as a
    csv under 'data/rfw/rfw.csv'.
    """
    rfw_df = CsvCreator.get_rfw_df()
    rfw_df.to_csv('data/rfw/rfw.csv', index=False)


def create_similarity_data():
    """
    This function derives the cosine similarity columns for both the RFW and BFW datasets and then saves the outputs
    as csvs.

    For BFW, both facenet-webface and arcface embeddings are available. For RFW, both facenet and facenet-webface
    embeddings are available.
    For each combination, a unique key is created to identify the images in the pairs. The embedding is then mapped
    to the image in each pair. The cosine similarity is derived by comparing the embeddings of the pair. Finally,
    two csvs with the cosine similarities is generated:
    - data/bfw/bfw_w_sims.csv
    - data/rfw/rfw_w_sims.csv
    These csvs feed into the fairness analysis.
    """

    # Record time
    start = time.time()
    # Print log
    print('Preparing the bfw and rfw datasets...')

    # Load the bfw and rfw files and clean the dataframes
    dfs = {
        'bfw': CsvCreator.load_and_prep_bfw(),
        'rfw': CsvCreator.load_and_prep_rfw()
    }
    # Create a dictionary to loop on the dataset-model pairs
    cos_sim_to_change = {
        'bfw': ['facenet-webface', 'arcface'],
        'rfw': ['facenet', 'facenet-webface']
    }

    # Add the cosine similarity column of each model to rfw and bfw
    for dataset, pretrained_models in cos_sim_to_change.items():
        for pretrained_model in pretrained_models:
            current_csv = pickle.load(open('similarities/' + pretrained_model + '_' + dataset + '_cosin_sim.pk', 'rb'))
            current_csv = CsvCreator.clean_cosine_sim_data(current_csv, dataset)

            similarity_map = dict(zip(current_csv['unique_key'], current_csv['cos_sim']))
            dfs[dataset][pretrained_model] = dfs[dataset]['unique_key'].map(similarity_map)

    # Print log
    print('Preparing the bfw and rfw datasets completed!')
    print('Outputting the csvs...')

    # Output the bfw and rfw files with the cosine similarities for the models
    dfs['bfw'].to_csv('data/bfw/bfw_w_sims.csv', index=False)
    dfs['rfw'].to_csv('data/rfw/rfw_w_sims.csv', index=False)

    # Print log
    print('Outputting the bfw and rfw csvs completed!')
    # Logging
    print(f'create_similarity_data took {round(time.time() - start)} seconds!')
