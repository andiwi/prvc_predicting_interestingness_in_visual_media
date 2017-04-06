import os
import cv2
from readImgs import readImgNames, readImg
from matplotlib import pyplot as plt
import numpy as np

#TODO need to mask images (black bar on side of images)

def calcHistograms():
    print('calcHistograms')

    directory = 'D:\\PR aus Visual Computing\\Interestingness16data\\allvideos\\images\\interesting\\histogram'

    imgNames = readImgNames(directory)

    for imgName in imgNames:
        img = readImg(os.path.join(directory, imgName))

        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        #plt.show()

        # save img
        if not os.path.exists(directory + '\\hist\\'):
            os.makedirs(directory + '\\hist\\')

        imgPath = directory + '\\hist\\' + imgName.split('.jpg')[0] + '_histogram.png'
        plt.savefig(imgPath)

    print("Finished")

def calcHistogramsNormalized():
    print('calcHistogramsNormalized')

    directory = 'D:\\PR aus Visual Computing\\Interestingness16data\\allvideos\\images\\interesting\\histogram'

    imgNames = readImgNames(directory)

    for imgName in imgNames:
        img = readImg(os.path.join(directory, imgName))

        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
        #plt.show()

        # save img
        if not os.path.exists(directory + '\\hist\\normalized\\'):
            os.makedirs(directory + '\\hist\\normalized\\')

        imgPath = directory + '\\hist\\normalized\\' + imgName.split('.jpg')[0] + '_histogram_normalized.png'
        plt.savefig(imgPath)

    print("Finished")

def calcHistogramsPlusImg():
    print('calcHistogramsPlusImg')

    directory = 'D:\\PR aus Visual Computing\\Interestingness16data\\allvideos\\images\\interesting\\histogram'

    imgNames = readImgNames(directory)

    fig = plt.figure()

    for imgName in imgNames:
        img = readImg(os.path.join(directory, imgName))

        a = fig.add_subplot(2, 1, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        a.set_title('Image')

        a = fig.add_subplot(2,1,2)

        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])

        a.set_title('Histogram')
        #plt.show()


        # save img
        if not os.path.exists(directory + '\\hist\\combined\\'):
            os.makedirs(directory + '\\hist\\combined\\')

        imgPath = directory + '\\hist\\combined\\' + imgName.split('.jpg')[0] + '_histogram.png'
        plt.savefig(imgPath)

        plt.clf()

    print("Finished")