
import os
import numpy as np

def clear_old_images(path, df):
    num_skipped = 0
    #listdir = os.listdir(imagepath)
    #number_files = len(listdir)

    #if number_files >= df.sample_quantity.sum(): #num_folders >= len(artlabels.genre.unique())
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
    #print("Deleted %d images" % num_skipped) #from % imagepath

    return num_skipped


def get_random_imgs(oldpath, newpath, df, qty):
    '''
    generate a random sample of files with
    at most the given qty from each artist
    to each newly created folder
    returns dataframe of files sampled with labels'''

    sample_list = []  # pd.DataFrame(columns=['label', 'filename'])
    num_skipped = 0
    copied = 0

    if os.path.exists(newpath):
        num_skipped = clear_old_images(newpath, df)

    for i in range(len(df)):
        rawpath = os.path.join(oldpath, df.name[i])
        files = [f for f in os.listdir(rawpath)]  # if os.path.isdir(f)]
        random_files = np.random.choice(files, int(round(qty * df['% of paintings'][i])))
        imgpath = os.path.join(newpath, df.genre[i])
        # check if folder exists; if it exists, clear any old residual images before new samples
        if os.path.exists(imgpath):
            pass
        #    num_skipped+=clear_old_images(imgpath)
        else:
            create_new_folder(imgpath)
        # create a dataframe to return label and filename
        for copy in random_files:
            path_to_copy = os.path.join(rawpath, copy)
            shutil.copy(path_to_copy, imgpath)
            sample_list.append([df.genre[i], copy])  # append(path_to_copy) #df.genre[i] and copy
            copied += 1

    print("Deleted %d images" % num_skipped)
    print("Generated %d new images" % copied)

    return sample_list