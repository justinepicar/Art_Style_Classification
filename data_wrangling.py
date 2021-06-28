import os
import shutil
import pandas as pd


def create_new_folder(path):
    """
    Creates a new folder under given path, if it doesn't already exist
    :param path: Path where folder is created
    :return: None
    """

    if not os.path.exists(path):
        os.makedirs(path)
        print(path + ' created')
    else:
        print(path + ' path already exists')


#    return None

def delete_folder(folder_path):
    """
    Deletes empty directory
    :param folder_path: directory passed to function
    :return: None
    """
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
    """
    Clears sample of images from each class before resampling a new batch
    :param path: Gives path of folder that will be cleared of images
    :return: None
    """
    print('Clearing any previous samples...')

    num_skipped = 0

    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            if not filename.startswith('.'):
                num_skipped += 1
                # Delete image
                os.remove(filepath)
        delete_folder(folder_path)

    print(f'Deleted {num_skipped} images')


def get_sample(new_path, df, genres, frac):
    """
    Gets sample of images and sorts them with folders
    labeled by genre
    :param new_path: new path for images classified by genre
    :param df: dataframe used with filenames and filepath
    :param genres: art style or genre
    :param frac: fraction of images to sample
    :return: None
    """

    print(f'Checking if {new_path} exists...')
    if not os.path.exists(new_path):
        create_new_folder(new_path)
    else:
        clear_old_images(new_path)

    copied = 0

    for i in genres:
        paths = list(df.filepath.loc[df.genre == i].sample(frac=frac, replace=False))
        img_path = os.path.join(new_path, i)
        create_new_folder(img_path)
        for filepath in paths:
            shutil.copy(filepath, img_path)
            copied += 1

    print(f'Generated {copied} new images')


def get_image_info(directory):
    """ this function returns labels, filename, and
        the full filepath of each image
        param: directory
        return: list of labels, filename, filepath"""

    labels = []
    fullpath = []
    filename = []

    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            if fpath is not None:
                filename.append(fname)
                fullpath.append(fpath)
                labels.append(folder_name)
    zipped = zip(['label', 'filename', 'filepath'], [labels, filename, fullpath])

    return pd.DataFrame(dict(list(zipped)))
