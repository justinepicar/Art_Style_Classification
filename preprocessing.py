#filter any corrupt data
from PIL import Image
import os

def delete_corrupt_image(image_path):
    '''
    Delete images corrupted in the directory
    :param image_path: Directory
    :return: None
    '''
    num_skipped = 0

    for folder_name in os.listdir(image_path):
        folder_path = os.path.join(image_path, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                img = Image.open(fpath) #open and read images in file path
                img.verify() #verify this is an image
            except(IOError, SyntaxError) as e:
                print(f'Bad file: {fname}')
                num_skipped+=1
                # Delete corrupted image
                os.remove(fpath)

    print(f'Deleted {num_skipped} images')