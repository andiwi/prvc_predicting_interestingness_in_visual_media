import os

import cv2

from file_handler.read_imgs import read_img, read_img_names


def draw_rule_of_thirds_lines(directory):
    '''
    draws rule of third lines into image. 
    saves images in subdirectory ./ruleofthirds
    :param directory: path to images  (e.x.: 'D:\\PR aus Visual Computing\\Interestingness16data\\allvideos\\images\\interesting\\cropped')
    :return: 
    '''
    print("draw_rule_of_thirds_lines")

    imgNames = read_img_names(directory)

    for imgName in imgNames:
        img = read_img(os.path.join(directory, imgName))
        img = draw_grid(img)

        #save img
        if not os.path.exists(directory + '\\ruleofthirds\\'):
            os.makedirs(directory + '\\ruleofthirds\\')

        imgPath = directory + '\\ruleofthirds\\' + imgName
        success = cv2.imwrite(imgPath, img)

def draw_grid(img):
    height, width, channels = img.shape

    cv2.line(img, (0, height/3), (width, height/3), (255, 0, 0), 1)
    cv2.line(img, (0, (height/3)*2), (width, (height/3)*2), (255, 0, 0), 1)

    cv2.line(img, (width/3, 0), (width/3, height), (255, 0, 0), 1)
    cv2.line(img, ((width / 3)*2, 0), ((width / 3)*2, height), (255, 0, 0), 1)

    return img


