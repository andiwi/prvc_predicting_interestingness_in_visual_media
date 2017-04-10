import cv2
import os

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
