import os

import cv2
from matplotlib import pyplot as plt

from file_handler.read_imgs import read_img_names, read_img


def calc_histograms(directory):
    '''
    calc histograms of images in directory. saves them into ./histograms/imgname_histogram.jpg
    :param directory: path to images  (e.x.: 'D:\\PR aus Visual Computing\\Interestingness16data\\allvideos\\images\\interesting\\cropped')
    :return: 
    '''
    print('calc_histograms')

    imgNames = read_img_names(directory)

    for imgName in imgNames:
        img = read_img(os.path.join(directory, imgName))

        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        #plt.show()

        # save img
        if not os.path.exists(directory + '\\histograms\\'):
            os.makedirs(directory + '\\histograms\\')

        imgPath = directory + '\\histograms\\' + imgName.split('.jpg')[0] + '_histogram.png'
        plt.savefig(imgPath)


def calc_histograms_normalized(directory):
    '''
    calc normalized histograms of images in directory. saves them into ./histograms/normalized/imgname_histogram_normalized.jpg
    :param directory: path to images  (e.x.: 'D:\\PR aus Visual Computing\\Interestingness16data\\allvideos\\images\\interesting\\cropped')
    :return: 
    '''
    print('calc_histograms_normalized')

    imgNames = read_img_names(directory)

    for imgName in imgNames:
        img = read_img(os.path.join(directory, imgName))

        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
        #plt.show()

        # save img
        if not os.path.exists(directory + '\\histograms\\normalized\\'):
            os.makedirs(directory + '\\histograms\\normalized\\')

        imgPath = directory + '\\histograms\\normalized\\' + imgName.split('.jpg')[0] + '_histogram_normalized.png'
        plt.savefig(imgPath)


def calc_histograms_plus_img(directory):
    '''
    calc histograms of images in directory. saves them beside image into ./histograms/combined/imgname_histogram.jpg
    :param directory: path to images  (e.x.: 'D:\\PR aus Visual Computing\\Interestingness16data\\allvideos\\images\\interesting\\cropped')
    :return: 
    '''
    print('calc_histograms_plus_img')

    imgNames = read_img_names(directory)

    fig = plt.figure()

    for imgName in imgNames:
        img = read_img(os.path.join(directory, imgName))

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
        if not os.path.exists(directory + '\\histograms\\combined\\'):
            os.makedirs(directory + '\\histograms\\combined\\')

        imgPath = directory + '\\histograms\\combined\\' + imgName.split('.jpg')[0] + '_histogram.png'
        plt.savefig(imgPath)

        plt.clf()


def calc_histograms_bw(directory):
    '''
    calc grayscale histograms of images in directory. saves them into ./histograms/imgname_histogram.jpg
    :param directory: path to images  (e.x.: 'D:\\PR aus Visual Computing\\Interestingness16data\\allvideos\\images\\interesting\\cropped')
    :return: 
    '''
    print('calc_histograms_bw')

    imgNames = read_img_names(directory)

    for imgName in imgNames:
        img = read_img(os.path.join(directory, imgName))
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        plt.hist(gray_image.ravel(), 256, [0, 256]);
        #plt.show()

        # save img
        if not os.path.exists(directory + '\\histograms\\grayscale\\'):
            os.makedirs(directory + '\\histograms\\grayscale\\')

        imgPath = directory + '\\histograms\\grayscale\\' + imgName.split('.jpg')[0] + '_histogram.png'
        plt.savefig(imgPath)
        plt.clf()
