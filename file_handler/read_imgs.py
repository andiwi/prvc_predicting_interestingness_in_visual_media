import cv2
import os
import glob


def read_imgs(directory):
    imgs = dict()
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

    return img_dirs
