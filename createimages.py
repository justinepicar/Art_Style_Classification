
import os
import shutil
import numpy as np

def create_new_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(path + ' created')
    else:
        print(path + ' path already exists')
    return None

def clear_old_images(path, df):

    num_skipped = 0

    for i in range(len(df)):
        imgpath = os.path.join(path, df.genre[i])
        if os.path.exists(imgpath):
            for fname in os.listdir(imgpath):
                if not fname.startswith('.'):
                    fpath = os.path.join(imgpath, fname)
                    num_skipped += 1
                    #Delete image
                    os.remove(fpath)

    print(f'Deleted {num_skipped} images')

    return None


def get_random_imgs(oldpath, newpath, df, perc):
    '''
    generate a random sample of files with
    at most the given qty from each artist
    to each newly created folder
    returns dataframe of files sampled with labels'''

    sample_list = []  # pd.DataFrame(columns=['label', 'filename'])
    copied = 0

    for i in range(len(df)):
        rawpath = os.path.join(oldpath, df.name[i])
        imgpath = os.path.join(newpath, df.genre[i])
        files = [f for f in os.listdir(rawpath)]
        sample = round(perc * df.paintings[i])
        random_files = np.random.choice(files, sample, replace=False)
        # check if folder exists; if not, create a new folder
        create_new_folder(imgpath)
        #create a dataframe to return label and filename
        for copy in random_files:
            path_to_copy = os.path.join(rawpath, copy)
            shutil.copy(path_to_copy, imgpath)
            sample_list.append([df.genre[i], os.path.join(imgpath, copy)])
            copied += 1

    print(f'Generated {copied} new images')

    return sample_list