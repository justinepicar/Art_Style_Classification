import os

def create_new_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(path + ' created')
    else:
        print(path + ' path already exists')
    return None


def get_image_info(directory):
    ''' this function returns labels, filename, and
        the full filepath of each image
        param: directory
        return: list of labels, filename, full_path'''

    labels = []
    full_path = []
    filename = []

    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)  # images/name
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)  # images/name/pic.jpg
            if fpath is not None:
                filename.append(fname)
                fullpath.append(fpath)
                labels.append(folder_name)

    partialpath = get_directory(directory)
    classfolders = get_class_folder(directory)

    return [labels, filename, full_path]