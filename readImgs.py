import cv2
import os

def readImgs(directory):

    imgs = dict()
    for file in os.listdir(directory):
        if file.endswith(".jpg"):
            img = cv2.imread(os.path.join(directory, file))
            imgs[file] = img

    return imgs

def readImgNames(directory):

    imgNames = []
    for file in os.listdir(directory):
        if file.endswith(".jpg"):
            imgNames.append(file)

    return imgNames

def readImg(imgPath):
    img = cv2.imread(imgPath)
    return img
