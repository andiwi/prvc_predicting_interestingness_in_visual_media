import cv2
import os
from read_imgs import read_img_names, read_img

def draw_phi_grid(directory):
    '''
    draws phi grid into image. 
    saves images in subdirectory ./phigrid
    :param directory: path to images (e.x.: 'D:\\PR aus Visual Computing\\Interestingness16data\\allvideos\\images\\interesting\\cropped')
    :return: 
    '''
    print("draw_phi_grid")

    imgNames = read_img_names(directory)

    for imgName in imgNames:
        img = read_img(os.path.join(directory, imgName))
        img = draw_grid(img)

        #save img
        if not os.path.exists(directory + '\\phigrid\\'):
            os.makedirs(directory + '\\phigrid\\')

        imgPath = directory + '\\phigrid\\' + imgName
        success = cv2.imwrite(imgPath, img)

def draw_grid(img):

    height, width, channels = img.shape
    ratio = 0.618

    int(round(width * ratio))

    cv2.line(img, (0, int(round(height*ratio))), (width, int(round(height*ratio))), (255, 0, 0), 1)
    cv2.line(img, (0, height - int(round(height*ratio))), (width, height-int(round(height*ratio))), (255, 0, 0), 1)

    cv2.line(img, (int(round(width * ratio)), 0), (int(round(width * ratio)), height), (255, 0, 0), 1)
    cv2.line(img, (width-int(round(width * ratio)), 0), (width-int(round(width * ratio)), height), (255, 0, 0), 1)

    return img


