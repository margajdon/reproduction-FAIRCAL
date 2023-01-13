import os

def prepare_dir(file_path):
    dir_path = '/'.join(file_path.split('/')[:-1])
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        pass