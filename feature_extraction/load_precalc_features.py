import os

import numpy as np
import scipy.io as sio

from file_handler.file_search import get_abs_path_of_file
from file_handler.read_imgs import read_img_names


def load_matlab_feature(path_to_imgs, feature_name):
    '''
    for all images in path_to_imgs directory: loads precalculated feature :feature_name
    :param path_to_imgs:
    :param feature_name:
    :return: numpy array of features of all imgs in path_to_imgs directory
    '''
    feature_root_dir = 'D:\\PR aus Visual Computing\\Interestingness17data\\features'

    img_names = read_img_names(path_to_imgs)

    features = []
    'D:\\PR aus Visual Computing\\Interestingness17data\\features\\'
    for img_name in img_names:

        if (feature_name == 'denseSIFT_L0'):
            feature = _denseSIFT_fe(feature_root_dir, img_name, level=0)
        elif (feature_name == 'denseSIFT_L1'):
            feature = _denseSIFT_fe(feature_root_dir, img_name, level=1)
        elif (feature_name == 'denseSIFT_L2'):
            feature = _denseSIFT_fe(feature_root_dir, img_name, level=2)
        elif (feature_name == 'HOG_L0'):
            feature = _hog_fe(feature_root_dir, img_name, level=0)
        elif (feature_name == 'HOG_L1'):
            feature = _hog_fe(feature_root_dir, img_name, level=1)
        elif (feature_name == 'HOG_L2'):
            feature = _hog_fe(feature_root_dir, img_name, level=2)
        elif (feature_name == 'LBP_L0'):
            feature = _lbp_fe(feature_root_dir, img_name, level=0)
        elif (feature_name == 'LBP_L1'):
            feature = _lbp_fe(feature_root_dir, img_name, level=1)
        elif (feature_name == 'LBP_L2'):
            feature = _lbp_fe(feature_root_dir, img_name, level=2)
        elif (feature_name == 'GIST'):
            feature = _gist_fe(feature_root_dir, img_name)
        elif (feature_name == 'ColorHist'):
            feature = _colorHist_fe(feature_root_dir, img_name)
        else:
            raise NotImplementedError
            # TODO do we need fc7 & prob layer of AlexNet and MFCC features?

        features.append(feature)
    return np.asarray(features)


def _colorHist_fe(feature_root_path, img_name):
    path = os.path.join(feature_root_path, 'Features_From_FudanUniversity', 'Image_Subtask', 'ColorHist')
    version = 1

    matfile_path = get_abs_path_of_file(path, img_name + '.mat')
    if (matfile_path is None):
        version = 2
        # remove .jpg. extension in filename
        img_name_splitted = img_name.split('.')
        matfile_path = get_abs_path_of_file(path, img_name_splitted[0] + '.mat')

        if (matfile_path is None):
            version = -1
            print 'STOP'

    img_features = sio.loadmat(matfile_path)
    if (version == 1):
        return img_features['hsv'][0]
    if (version == 2):
        return img_features['ColorHist'][0]
    if (version == -1):
        raise NotImplementedError


def _denseSIFT_fe(feature_root_path, img_name, level):
    path = os.path.join(feature_root_path, 'Features_From_FudanUniversity', 'Image_Subtask', 'denseSIFT')

    matfile_path = get_abs_path_of_file(path, img_name + '.mat')
    if (matfile_path is None):
        return None
    else:
        img_features = sio.loadmat(matfile_path)
        if level == 0:
            return img_features['hists']['L0'][0][0]
        elif level == 1:
            return img_features['hists']['L1'][0][0]
        elif level == 2:
            return img_features['hists']['L2'][0][0]
        else:
            return None


def _hog_fe(feature_root_path, img_name, level):
    path = os.path.join(feature_root_path, 'Features_From_FudanUniversity', 'Image_Subtask', 'hog2x2')

    matfile_path = get_abs_path_of_file(path, img_name + '.mat')
    if (matfile_path is None):
        return None
    else:
        img_features = sio.loadmat(matfile_path)
        if level == 0:
            return img_features['hists']['L0'][0][0]
        elif level == 1:
            return img_features['hists']['L1'][0][0]
        elif level == 2:
            return img_features['hists']['L2'][0][0]
        else:
            return None


def _lbp_fe(feature_root_path, img_name, level):
    path = os.path.join(feature_root_path, 'Features_From_FudanUniversity', 'Image_Subtask', 'lbp')

    matfile_path = get_abs_path_of_file(path, img_name + '.mat')
    if (matfile_path is None):
        return None
    else:
        img_features = sio.loadmat(matfile_path)
        if level == 0:
            return img_features['hists']['L0'][0][0]
        elif level == 1:
            return img_features['hists']['L1'][0][0]
        elif level == 2:
            return img_features['hists']['L2'][0][0]
        else:
            return None


def _gist_fe(feature_root_path, img_name):
    path = os.path.join(feature_root_path, 'Features_From_FudanUniversity', 'Image_Subtask', 'gist')

    matfile_path = get_abs_path_of_file(path, img_name + '.mat')
    if (matfile_path is None):
        return None
    else:
        img_features = sio.loadmat(matfile_path)
        return img_features['descrs']