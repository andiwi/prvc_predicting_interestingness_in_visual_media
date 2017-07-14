import os
import selectTrainingAndTestData
from file_handler import filecopy
from crop_imgs import crop_black_borders

def preprocessing(img_dirs):
    '''
    :param img_dirs: list of image paths
    :return:
    '''
    crop_black_borders(img_dirs)