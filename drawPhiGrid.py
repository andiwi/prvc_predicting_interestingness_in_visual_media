import cv2
import os
from readImgs import readImgNames, readImg

def drawPhiGrid():
    print("drawPhiGrid")

    directory = 'D:\\PR aus Visual Computing\\Interestingness16data\\allvideos\\images\\interesting\\phigrid'

    imgNames = readImgNames(directory)

    for imgName in imgNames:
        img = readImg(os.path.join(directory, imgName))
        img = drawGrid(img)

        #save img
        if not os.path.exists(directory + '\\modified\\'):
            os.makedirs(directory + '\\modified\\')

        imgPath = directory + '\\modified\\' + imgName
        success = cv2.imwrite(imgPath, img)

    print("Finished")

def drawGrid(img):

    height, width, channels = img.shape
    ratio = 0.618

    int(round(width * ratio))

    cv2.line(img, (0, int(round(height*ratio))), (width, int(round(height*ratio))), (255, 0, 0), 1)
    cv2.line(img, (0, height - int(round(height*ratio))), (width, height-int(round(height*ratio))), (255, 0, 0), 1)

    cv2.line(img, (int(round(width * ratio)), 0), (int(round(width * ratio)), height), (255, 0, 0), 1)
    cv2.line(img, (width-int(round(width * ratio)), 0), (width-int(round(width * ratio)), height), (255, 0, 0), 1)

    return img


