import platform
from collections import OrderedDict

import cv2
import os
import glob


def read_imgs(directory):
    imgs = OrderedDict()
    for file in os.listdir(directory):
        if file.endswith(".jpg"):
            img = cv2.imread(os.path.join(directory, file))
            imgs[file] = img

    return imgs


def read_img_names(directory):
    imgNames = []
    for file in os.listdir(directory):
        if file.endswith(".jpg"):
            imgNames.append(file)

    return imgNames


def read_img(imgPath):
    img = cv2.imread(imgPath)
    return img


def read_img_dirs(dir):
    """
    read all jpg image paths in directory
    :param dir: the directory where .jpg files are searched for
    :type dir: str
    :return: the image paths
    :rtype: list
    """
    img_dirs = list()

    for root, dirs, files in os.walk(dir):
        for file_path in glob.glob(os.path.join(root, '*.jpg')):
            img_dirs.append(file_path)

    img_dirs.sort(key=_get_img_dirs_key)
    return img_dirs

def _get_img_dirs_key(img_dir):
    if platform.system() == 'Linux':
        videonumber = int(img_dir[img_dir.find('videos') + 13:img_dir.find('images') - 1])
        shotname = img_dir[img_dir.rfind('/') + 1:]

    else:  # windows
        videonumber = int(img_dir[img_dir.find('videos') + 13:img_dir.find('images') - 1])
        shotname = img_dir[img_dir.rfind('\\') + 1:]

    shotnum1 = int(shotname[:shotname.find('_')])
    shotnum2 = int(shotname[shotname.find('_')+1:shotname.find('-')])
    shotnum3 = int(shotname[shotname.find('-')+1:shotname.find('.jpg')])

    return (videonumber, shotnum1, shotnum2, shotnum3)


