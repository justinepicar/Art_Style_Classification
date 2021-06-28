
import os
import shutil
import pandas as pd

def create_new_folder(path):
    '''
    Creates a new folder under given path, if it doesn't already exist
    :param path: Path where folder is created
    :return: None
    '''

    if not os.path.exists(path):
        os.makedirs(path)
        print(path + ' created')
    else:
        print(path + ' path already exists')
#    return None

def delete_folder(folder_path):
    '''
    Deletes empty directory
    :param path: directory passed to function
    :return: None
    '''
    if os.path.exists(folder_path) and not os.path.isfile(folder_path):
        # Checking if the directory is empty or not
        if not os.listdir(folder_path):
            print(f'Empty directory. Delete {folder_path}')
            os.rmdir(folder_path)
        else:
            print(f' {folder_path} not empty directory')
    else:
        print("The path is either for a file or not valid")

def clear_old_images(path):
    '''
    Clears sample of images from each class before resampling a new batch
    :param path: Gives path of folder that will be cleared of images
    :return: None
    '''
    print('Clearing any previous samples...')

    num_skipped = 0

    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            if not fname.startswith('.'):
                num_skipped += 1
                # Delete image
                os.remove(fpath)
        delete_folder(folder_path)

    print(f'Deleted {num_skipped} images')


def get_sample(newpath, df, genres, frac):
    '''
    Gets sample of images and sorts them with folders
    labeled by genre
    :param oldpath: path of original images classified by artist
    :param newpath: new path for images classified by genre
    :param df: dataframe used with filenames and filepaths
    :param genres: art style or genre
    :param frac: fraction of images to sample
    :return: None
    '''

    print(f'Checking if {newpath} exists...')
    if not os.path.exists(newpath):
        create_new_folder(newpath)
    else:
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

def get_image_info(directory):
    ''' this function returns labels, filename, and
        the full filepath of each image
        param: directory
        return: list of labels, filename, fullpath'''

    labels = []
    fullpath = []
    filename = []

    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)  # images/name
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)  # images/name/pic.jpg
            if fpath is not None:
                filename.append(fname)
                fullpath.append(fpath)
                labels.append(folder_name)
    zipped = zip(['label', 'filename', 'filepath'], [labels, filename, fullpath])

    return pd.DataFrame(dict(list(zipped)))