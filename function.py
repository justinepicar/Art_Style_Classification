import os

def create_new_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(path + ' created')
    else:
        print(path + ' path already exists')
    return None

def get_class_folder(folder):
    ''' this function returns
        a list of all class folder names
        param: folder
        return: folder_list '''
    folder_list = []
    for f in os.listdir(folder):
        folder_list.append(f)

    return folder_list


def get_directory(directory):
    ''' this function creates the directory path
        and returns a partial path to the images
        param: directory
        return: partialpath'''

    partialpath = []
    classfolders = get_class_folder(directory)

    for folder in classfolders:
        path = os.path.join(directory, folder)
        if path is not None:
            partialpath.append(path)

    return partialpath


def get_image_info(directory):
    ''' this function returns labels, filename, and
        the full filepath of each image
        param: directory
        return: list of labels, filename, full_path'''

    # for name in os.listdir(directory):
    # if not name.endswith('.jpg'):
    #    continue
    # else:

    partialpath = get_directory(directory)
    classfolders = get_class_folder(directory)

    labels = []
    full_path = []
    filename = []

    i = 0

    for path in partialpath:
        for name in os.listdir(path):
            img = os.path.join(path, name)
            if img is not None:
                full_path.append(img)
                filename.append(name)
                labels.append(classfolders[i])
        i += 1

    return [labels, filename, full_path]