import os
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from read_imgs import read_img_names, read_img

def crop_black_borders(directory):
    """
    crops black borders around image. saves it into folder ./cropped/
    :param directory: directory: path to image  (e.x.: 'D:\\PR aus Visual Computing\\Interestingness16data\\allvideos\\images\\interesting')
    :return: 
    """
    print('crop_black_borders')
    imgNames = read_img_names(directory)

    for imgName in imgNames:
        img = read_img(os.path.join(directory, imgName))
        #img = read_img('C:\\Users\\Andreas\\Desktop\\test.jpg')

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #sharpen image
        kernel_sharpen_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img_gray = cv2.filter2D(img_gray, -1, kernel_sharpen_1)

        edges = cv2.Canny(img_gray, 1, 1)

        #morphological closing
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        for i in range(100):
            closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)


        #check if border contains white pixels (contour)

        contours_pixel_cnt = cv2.countNonZero(closing)
        if contours_pixel_cnt > 0:
            height, width, channels = img.shape

            crop = False #is true if img should be cropped

            row_idx_top, row_idx_bottom = find_horizontal_img_border(closing)
            if row_idx_top != 0 and row_idx_bottom != height:
                crop = True
                #cv2.line(img, (0, row_idx_top), (width, row_idx_top), (0, 0, 255), 3)
                #cv2.line(img, (0, row_idx_bottom), (width, row_idx_bottom), (0, 0, 255), 3)

            col_idx_left, col_idx_right = find_vertical_img_border(closing)
            if col_idx_left != 0 and col_idx_right != width:
                crop = True
                #cv2.line(img, (col_idx_left, 0), (col_idx_left, height), (0, 0, 255), 3)
                #cv2.line(img, (col_idx_right, 0), (col_idx_right, height), (0, 0, 255), 3)

        #crop image if necessary
        if crop:
            img = img[row_idx_top:row_idx_bottom, col_idx_left:col_idx_right]

        # save img
        if not os.path.exists(directory + '\\cropped\\'):
            os.makedirs(directory + '\\cropped\\')

        imgPath = directory + '\\cropped\\' + imgName
        success = cv2.imwrite(imgPath, img)
    print('finished')

    '''
    #DEBUG Visualization
    plt.figure(1)
    plt.subplot(311)
    plt.imshow(edges, 'gray')

    plt.subplot(312)
    plt.imshow(closing, 'gray')
    plt.title('closing')

    plt.subplot(313)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    plt.title('image')

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    '''

def find_horizontal_img_border(img_binary):
    """
    :param img_binary: a binary image where real scene content is white and all unimportant content black (which should be cropped) 
    :return: row_idx_top, row_idx_bottom where real scene starts, ends
    """
    height, width = img_binary.shape
    # check for horizontal lines
    #from top to bottom
    for row_idx in range(0, height/3):
        # create mask for scanning image rowwise
        mask = np.zeros((height, width, 1), np.uint8)
        mask[row_idx:row_idx + 1, 0:width] = 1
        masked_img = cv2.bitwise_and(img_binary, img_binary, mask=mask)

        line_pixel_cnt = cv2.countNonZero(masked_img)

        if row_idx == 0 and line_pixel_cnt > 10:
            return 0, height #no borders

        if line_pixel_cnt > width * 0.7: # if 70 percent of all pixel in this row are white, then there begins the real image
            return row_idx, height - row_idx

    #from bottom to top
    for row_idx in range(0, height/3):
        # create mask for scanning image rowwise
        mask = np.zeros((height, width, 1), np.uint8)
        mask[height-row_idx:height-(row_idx + 1), 0:width] = 1
        masked_img = cv2.bitwise_and(img_binary, img_binary, mask=mask)

        line_pixel_cnt = cv2.countNonZero(masked_img)

        if row_idx == 0 and line_pixel_cnt > 10:
            return 0, height #no borders

        if line_pixel_cnt > width * 0.7: # if 70 percent of all pixel in this row are white, then there begins the real image
            return row_idx, height - row_idx

    return 0, height #cannot find border

def find_vertical_img_border(img_binary):
    """
    :param img_binary: a binary image where real scene content is white and all unimportant content black (which should be cropped) 
    :return: col_idx_left, col_idx_right where real scene starts, ends
    """
    height, width = img_binary.shape
    # check for vertical lines
    #from left to right
    for col_idx in range(0, width/3):
        # create mask for scanning image rowwise
        mask = np.zeros((height, width, 1), np.uint8)
        mask[0:height, col_idx:col_idx+1] = 1
        masked_img = cv2.bitwise_and(img_binary, img_binary, mask=mask)

        line_pixel_cnt = cv2.countNonZero(masked_img)

        if col_idx == 0 and line_pixel_cnt > 10:
            return 0, width #no borders

        if line_pixel_cnt > width * 0.7: # if 70 percent of all pixel in this column are white, then there begins the real image
            return col_idx, width - col_idx

    # from right to left
    for col_idx in range(0, width/3):
        # create mask for scanning image rowwise
        mask = np.zeros((height, width, 1), np.uint8)
        mask[0:height, width-col_idx:width-(col_idx + 1)] = 1
        masked_img = cv2.bitwise_and(img_binary, img_binary, mask=mask)

        line_pixel_cnt = cv2.countNonZero(masked_img)

        if col_idx == width and line_pixel_cnt > 10:
            return 0, width  # no borders

        if line_pixel_cnt > width * 0.7:  # if 70 percent of all pixel in this column are white, then there begins the real image
            return col_idx, width - col_idx

    return 0, width  # cannot find border


