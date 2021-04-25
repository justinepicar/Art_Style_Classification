
import os
import numpy as np
from sklearn.decomposition import PCA
from math import ceil
from keras.preprocessing import image

#turn images into a matrix
def img2np(path, size = (64, 64)):
     # iterating through each file
    for fn in os.listdir(path):
        fp = os.path.join(path, fn)
        current_image = image.load_img(fp, target_size = size,
                                       color_mode = 'grayscale')
        # covert image to a matrix
        img_ts = image.img_to_array(current_image)
        # turn that into a vector / 1D array
        img_ts = [img_ts.ravel()]
        try:
            # concatenate different images
            full_mat = np.concatenate((full_mat, img_ts))
        except UnboundLocalError:
            # if not assigned yet, assign one
            full_mat = img_ts
    return full_mat


# average image
def find_mean_img(full_mat, title, size=(64, 64)):
    # calculate the average
    mean_img = np.mean(full_mat, axis=0)
    # reshape it back to a matrix
    mean_img = mean_img.reshape(size)
    # plt.imshow(mean_img, vmin=0, vmax=255)
    # plt.title(f'Average {title}')
    # plt.axis('off')
    # plt.show()

    return mean_img


def find_std_img(full_mat, title, size=(64, 64)):
    # calculate the average
    std_img = np.std(full_mat, axis=0)
    # reshape it back to a matrix
    std_img = std_img.reshape(size)

    return std_img


def eigenimages(full_mat, title, n_comp=0.7, size=(64, 64)):
    # fit PCA to describe n_comp * variability in the class
    pca = PCA(n_components=n_comp, whiten=True)
    pca.fit(full_mat)
    print(f'Number of PC for {title}: ', pca.n_components_)
    return pca


def plot_pca(pca, size=(64, 64)):
    # plot eigenimages in a grid
    n = pca.n_components_
    fig = plt.figure(figsize=(8, 8))
    r = int(n ** .5)
    c = ceil(n / r)
    for i in range(n):
        ax = fig.add_subplot(r, c, i + 1, xticks=[], yticks=[])
        ax.imshow(pca.components_[i].reshape(size),
                  cmap='Greys_r')
    plt.axis('off')
    plt.show()