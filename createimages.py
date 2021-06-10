
import os
import shutil
import numpy as np

def create_new_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(path + ' created')
    else:
        print(path + ' path already exists')
#    return None

def clear_old_images(path):
    num_skipped = 0

    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            if not fname.startswith('.'):
                num_skipped += 1
                # Delete image
                os.remove(fpath)

    print(f'Deleted {num_skipped} images')


def get_images(oldpath, newpath, df, genres, frac):
    print('Clearing any previous samples...')
    if len(os.listdir(newpath)) > 1:
        clear_old_images(newpath)

    copied = 0

    for i in genres:
        paths = list(df.filepath.loc[df.genre == i].sample(frac=frac, replace=False))
        imgpath = os.path.join(newpath, i)
        create_new_folder(imgpath)
        for filepath in paths:
            shutil.copy(filepath, imgpath)
            copied += 1

    print(f'Generated {copied} new images')