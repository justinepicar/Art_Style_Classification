import os

def create_new_folder(path):
    '''

    :param path:
    :return:
    '''
    if not os.path.exists(path):
        os.makedirs(path)
        print(path + ' created')
    else:
        print(path + ' path already exists')
    return None