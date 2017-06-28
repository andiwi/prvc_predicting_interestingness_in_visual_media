from read_imgs import read_img_names, read_img
import os
import numpy as np
import cv2
import math
from face_detection import detect_faces
from mpeg7_edge_histogram import calc_edge_histogram


def calc_features(directory, feature_calculator, *args):
    img_names = read_img_names(directory)

    features = []
    for img_name in img_names:
        img = read_img(os.path.join(directory, img_name))

        if args is None:
            feature = feature_calculator(img)
        else:
            feature = feature_calculator(img, *args)

        features.append(feature)

    return np.asarray(features)


def face_count_calculator(img, face_frontal_cascade=None, face_profile_cascade=None):
    directory_haarfeatures = os.getcwd() + '\\res\\haarcascades\\'
    if face_frontal_cascade is None:
        face_frontal_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_frontalface_default.xml')

    if face_profile_cascade is None:
        face_profile_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_profileface.xml')

    rect_faces, rect_faces_frontal, rect_faces_profile = detect_faces(img, face_frontal_cascade, face_profile_cascade)

    return len(rect_faces)


def img_tilted_calculator(img):
    '''
    uses edge histogram to detect if camera perspective is tilted
    :param img:
    :return: percentage of edges that are tilted
    '''
    hists, quant_hist, global_hist, quant_global_hist, semiglobal_hist, quant_semiglobal_hist = calc_edge_histogram(img)

    #not_tilted = sum(global_hist[0:2])
    tilted = sum(global_hist[2:4])
    #non_directional_edges = global_hist[4]

    return tilted / sum(global_hist)

def edge_hist_dir_calculator(img, only_strongest_dir=False, maintain_values=True):
    '''
    calculates edge histogram, sets
    :param img:
    :param only_strongest_dir if True -> only the strongest direction of a block has a value. Other directions are set to 0
    :param maintain_values if False -> values of edge histogram were replaced by a 1 for the max direction
    :return: edge histogram in a row as a feature vector for the image
    '''
    hists, quant_hist, global_hist, quant_global_hist, semiglobal_hist, quant_semiglobal_hist = calc_edge_histogram(img)

    if only_strongest_dir:
        #for every block find maximum value and set other values to 0
        def setToZeros(row):
            max_val = row.max()
            row = np.where(row != max_val, 0, max_val)
            return row

        hists = np.apply_along_axis(setToZeros, axis=1, arr=hists)

    if not maintain_values:
        #set remaining values to 1
        hists = np.where(hists > 0, 1, 0)

    hists_1D = hists.flatten()




    return hists_1D