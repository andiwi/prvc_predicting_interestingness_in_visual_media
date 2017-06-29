import os

import cv2
import numpy as np
import math

from face_detection import detect_faces
from file_handler.read_imgs import read_img_names, read_img
from mpeg7_edge_histogram import calc_edge_histogram
from feature_extraction.load_precalc_features import load_matlab_feature
from face_detection import  face_detection
from Features import Features

def calc_features(dir, feature_names):
    '''
    calculates all features given by feature_names
    :param (String) dir: directory containing subdirectories 'interesting' and 'uninteresting' which contain images
    :param (list) feature_names: list of feature names which should be calculated
    :return: (dict) {feature_name: [features_interesting, features_uninteresting]}
    '''
    features = {}
    dir_int = os.path.join(dir, 'interesting')
    dir_unint = os.path.join(dir, 'uninteresting')

    for name in feature_names:
        if name == Features.Face_count or name == Features.Rot_distance or name == Features.Face_bb:
            directory_haarfeatures = os.getcwd() + '\\res\\haarcascades\\'
            face_frontal_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_frontalface_default.xml')
            face_profile_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_profileface.xml')

            face_count_interesting, rot_distance_interesting, face_bb_interesting = face_detection(dir_int)
            face_count_uninteresting, rot_distance_uninteresting, face_bb_uninteresting = face_detection(dir_unint)

            features[Features.Face_count] = [face_count_interesting, face_count_uninteresting]
            features[Features.Rot_distance] = [rot_distance_interesting, rot_distance_uninteresting]
            features[Features.Face_bb] = [face_bb_interesting, face_bb_uninteresting]

        elif name == Features.Face_bb_full_img:
            directory_haarfeatures = os.getcwd() + '\\res\\haarcascades\\'
            face_frontal_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_frontalface_default.xml')
            face_profile_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_profileface.xml')

            interesting = _calc_features(dir_int, face_bb_full_img_calculator, face_frontal_cascade, face_profile_cascade)
            uninteresting = _calc_features(dir_unint, face_bb_full_img_calculator, face_frontal_cascade, face_profile_cascade)
            features[Features.Face_bb_full_img] = [interesting, uninteresting]

        elif name == Features.Face_bb_quarter_imgs:
            directory_haarfeatures = os.getcwd() + '\\res\\haarcascades\\'
            face_frontal_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_frontalface_default.xml')
            face_profile_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_profileface.xml')

            interesting = _calc_features(dir_int, face_bb_quarter_imgs_calculator, face_frontal_cascade, face_profile_cascade)
            uninteresting = _calc_features(dir_unint, face_bb_quarter_imgs_calculator, face_frontal_cascade, face_profile_cascade)
            features[Features.Face_bb_quarter_imgs] = [interesting, uninteresting]

        elif name == Features.Face_bb_eighth_imgs:
            directory_haarfeatures = os.getcwd() + '\\res\\haarcascades\\'
            face_frontal_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_frontalface_default.xml')
            face_profile_cascade = cv2.CascadeClassifier(directory_haarfeatures + 'haarcascade_profileface.xml')

            interesting = _calc_features(dir_int, face_bb_eighth_imgs_calculator, face_frontal_cascade, face_profile_cascade)
            uninteresting = _calc_features(dir_unint, face_bb_eighth_imgs_calculator, face_frontal_cascade, face_profile_cascade)
            features[Features.Face_bb_eighth_imgs] = [interesting, uninteresting]

        elif name == Features.Tilted_edges:
            interesting = _calc_features(dir_int, img_tilted_calculator)
            uninteresting = _calc_features(dir_unint, img_tilted_calculator)
            features[Features.Tilted_edges] = [interesting, uninteresting]

        elif name == Features.Edge_hist_v0:
            interesting = _calc_features(dir_int, edge_hist_dir_calculator, False, False)
            uninteresting = _calc_features(dir_unint, edge_hist_dir_calculator, False, False)
            features[Features.Edge_hist] = [interesting, uninteresting]

        elif name == Features.Edge_hist_v1:
            interesting = _calc_features(dir_int, edge_hist_dir_calculator, True, True)
            uninteresting = _calc_features(dir_unint, edge_hist_dir_calculator, True, True)
            features[Features.Edge_hist] = [interesting, uninteresting]

        elif name == Features.Edge_hist_v2:
            interesting = _calc_features(dir_int, edge_hist_dir_calculator, True, False)
            uninteresting = _calc_features(dir_unint, edge_hist_dir_calculator, True, False)
            features[Features.Edge_hist] = [interesting, uninteresting]

        #precalculated features
        elif name == Features.Hsv_hist:
            interesting = load_matlab_feature(dir_int, Features.Hsv_hist)
            uninteresting = load_matlab_feature(dir_unint, Features.Hsv_hist)
            features[Features.Hsv_hist] = [interesting, uninteresting]

        elif name == Features.DenseSIFT_L0:
            interesting = load_matlab_feature(dir_int, Features.DenseSIFT_L0)
            uninteresting = load_matlab_feature(dir_unint, Features.DenseSIFT_L0)
            features[Features.DenseSIFT_L0] = [interesting, uninteresting]

        elif name == Features.DenseSIFT_L1:
            interesting = load_matlab_feature(dir_int, Features.DenseSIFT_L1)
            uninteresting = load_matlab_feature(dir_unint, Features.DenseSIFT_L1)
            features[Features.DenseSIFT_L1] = [interesting, uninteresting]

        elif name == Features.DenseSIFT_L2:
            interesting = load_matlab_feature(dir_int, Features.DenseSIFT_L2)
            uninteresting = load_matlab_feature(dir_unint, Features.DenseSIFT_L2)
            features[Features.DenseSIFT_L2] = [interesting, uninteresting]

        elif name == Features.Hog_L0:
            interesting = load_matlab_feature(dir_int, Features.Hog_L0)
            uninteresting = load_matlab_feature(dir_unint, Features.Hog_L0)
            features[Features.Hog_L0] = [interesting, uninteresting]

        elif name == Features.Hog_L1:
            interesting = load_matlab_feature(dir_int, Features.Hog_L1)
            uninteresting = load_matlab_feature(dir_unint, Features.Hog_L1)
            features[Features.Hog_L1] = [interesting, uninteresting]

        elif name == Features.Hog_L2:
            interesting = load_matlab_feature(dir_int, Features.Hog_L2)
            uninteresting = load_matlab_feature(dir_unint, Features.Hog_L2)
            features[Features.Hog_L2] = [interesting, uninteresting]

        elif name == Features.Lbp_L0:
            interesting = load_matlab_feature(dir_int, Features.Lbp_L0)
            uninteresting = load_matlab_feature(dir_unint, Features.Lbp_L0)
            features[Features.Lbp_L0] = [interesting, uninteresting]

        elif name == Features.Lbp_L1:
            interesting = load_matlab_feature(dir_int, Features.Lbp_L1)
            uninteresting = load_matlab_feature(dir_unint, Features.Lbp_L1)
            features[Features.Lbp_L1] = [interesting, uninteresting]

        elif name == Features.Lbp_L2:
            interesting = load_matlab_feature(dir_int, Features.Lbp_L2)
            uninteresting = load_matlab_feature(dir_unint, Features.Lbp_L2)
            features[Features.Lbp_L2] = [interesting, uninteresting]

        elif name == Features.Gist:
            interesting = load_matlab_feature(dir_int, Features.Gist)
            uninteresting = load_matlab_feature(dir_unint, Features.Gist)
            features[Features.Gist] = [interesting, uninteresting]

        else:
            raise NotImplementedError

    return features

def _calc_features(directory, feature_calculator, *args):
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

def face_bb_full_img_calculator(img, face_frontal_cascade, face_profile_cascade):
    """
    detects faces in image. returns biggest bounding box

    :param img:
    :param face_frontal_cascade:
    :param face_profile_cascade:
    :return: the biggest bounding boxes
    """
    # DEBUG
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # DEBUG END

    rect_faces, rect_faces_frontal, rect_faces_profile = detect_faces(img, face_frontal_cascade, face_profile_cascade)
    # sort list in descending order
    rect_faces.sort(key=lambda rect: rect.area(), reverse=True)

    if len(rect_faces) > 0:
        bbox = [rect_faces[0].x,
                rect_faces[0].y,
                rect_faces[0].w,
                rect_faces[0].h]
    else:
        bbox = [0, 0, 0, 0]

    return np.asarray(bbox)

def face_bb_quarter_imgs_calculator(img, face_frontal_cascade, face_profile_cascade):
    """
    splits the image into 4 subimages. for each subimage it detects faces, chooses the face with biggest bounding box and stores bb (x,y,w,h) as feature

    How the image is splitted:
     _________
    |    |    |
    |__1_|__2_|
    |    |    |
    |__3_|_4__|

    :param img:
    :param face_frontal_cascade:
    :param face_profile_cascade:
    :return: the biggest bounding boxes (1 for each subimg)
    """

    #split image into 4 quarter images
    height, width, channels = img.shape

    subimgs = [img[0:int(height/2), 0:int(width/2)],
               img[0:int(height / 2), int(width / 2):width],
               img[int(height / 2):height, 0:int(width / 2)],
               img[int(height / 2):height, int(width / 2):width]
               ]

    bboxes = []

    # DEBUG
    cv2.imshow('image', img)
    cv2.waitKey(0)

    for subimg in subimgs:
        cv2.imshow('image', subimg)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    # DEBUG END

    for subimg in subimgs:
        rect_faces, rect_faces_frontal, rect_faces_profile = detect_faces(subimg, face_frontal_cascade, face_profile_cascade)
        # sort list in descending order
        rect_faces.sort(key=lambda rect: rect.area(), reverse=True)

        if len(rect_faces) > 0:

            bboxes.append(rect_faces[0].x)
            bboxes.append(rect_faces[0].y)
            bboxes.append(rect_faces[0].w)
            bboxes.append(rect_faces[0].h)
        else:
            bboxes.append(0)
            bboxes.append(0)
            bboxes.append(0)
            bboxes.append(0)

    return np.asarray(bboxes)

def face_bb_eighth_imgs_calculator(img, face_frontal_cascade, face_profile_cascade):
    """
    splits the image into 8 subimages. for each subimage it detects faces, chooses the face with biggest bounding box and stores bb (x,y,w,h) as feature

    How the image is splitted:

    :param img:
    :param face_frontal_cascade:
    :param face_profile_cascade:
    :return: the biggest bounding boxes (1 for each subimg)
    """

    #split image into 4 quarter images
    height, width, channels = img.shape

    subimgs = [img[0:int(height / 4), 0:int(width/4)],
               img[0:int(height / 4), int(width/4):int(width/2)],
               img[0:int(height / 4), int(width / 2):int(width / 4)*3],
               img[0:int(height / 4), int(width / 4)*3:width],

               img[int(height / 4):int(height / 2), 0:int(width / 4)],
               img[int(height / 4):int(height / 2), int(width / 4):int(width / 2)],
               img[int(height / 4):int(height / 2), int(width / 2):int(width / 4) * 3],
               img[int(height / 4):int(height / 2), int(width / 4) * 3:width],

               img[int(height / 2):int(height / 4)*3, 0:int(width / 4)],
               img[int(height / 2):int(height / 4)*3, int(width / 4):int(width / 2)],
               img[int(height / 2):int(height / 4)*3, int(width / 2):int(width / 4) * 3],
               img[int(height / 2):int(height / 4)*3, int(width / 4) * 3:width],

               img[int(height / 4)*3:height, 0:int(width / 4)],
               img[int(height / 4)*3:height, int(width / 4):int(width / 2)],
               img[int(height / 4)*3:height, int(width / 2):int(width / 4) * 3],
               img[int(height / 4)*3:height, int(width / 4) * 3:width]
               ]

    bboxes = []

    #DEBUG
    cv2.imshow('image', img)
    cv2.waitKey(0)

    for subimg in subimgs:
        cv2.imshow('image', subimg)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    # DEBUG END

    for subimg in subimgs:
        rect_faces, rect_faces_frontal, rect_faces_profile = detect_faces(subimg, face_frontal_cascade, face_profile_cascade)
        # sort list in descending order
        rect_faces.sort(key=lambda rect: rect.area(), reverse=True)

        if len(rect_faces) > 0:

            bboxes.append(rect_faces[0].x)
            bboxes.append(rect_faces[0].y)
            bboxes.append(rect_faces[0].w)
            bboxes.append(rect_faces[0].h)
        else:
            bboxes.append(0)
            bboxes.append(0)
            bboxes.append(0)
            bboxes.append(0)

    return np.asarray(bboxes)
