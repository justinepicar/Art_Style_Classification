
import os
import shutil
import numpy as np

def clear_old_images(path, df):
    num_skipped = 0

    for i in range(len(df)):
        imgpath = os.path.join(path, df.genre[i])
        if os.path.exists(imgpath):
            for fname in os.listdir(imgpath):
                if not fname.startswith('.'):
                    fpath = os.path.join(imgpath, fname)
                    #print(fpath)
                    num_skipped += 1
                    #Delete image
                    os.remove(fpath)

    print(f'Deleted {num_skipped} images') #from % imagepath

    return None


def get_random_imgs(oldpath, newpath, df, qty):
    '''
    generate a random sample of files with
    at most the given qty from each artist
    to each newly created folder
    returns dataframe of files sampled with labels'''

    sample_list = []  # pd.DataFrame(columns=['label', 'filename'])
    copied = 0

    for i in range(len(df)):
        rawpath = os.path.join(oldpath, df.name[i]) #not the issue
        imgpath = os.path.join(newpath, df.genre[i])  # not the issue
        files = [f for f in os.listdir(rawpath)]  # if os.path.isdir(f)]
        sample = int(round(qty * df['% of paintings'][i])) #not the issue
        if len(os.listdir(rawpath)) >= sample:
            random_files = np.random.choice(files, sample)
        else:
            random_files = os.listdir(rawpath)
        # check if folder exists; if it exists, clear any old residual images before new samples
        if not os.path.exists(newpath):
            create_new_folder(imgpath)
        #create a dataframe to return label and filename
        for copy in random_files:
            path_to_copy = os.path.join(rawpath, copy)
            shutil.copy(path_to_copy, imgpath)
            sample_list.append([df.genre[i], copy])  # append(path_to_copy) #df.genre[i] and copy
            copied += 1

    print(f'Generated {copied} new images')

    return sample_list