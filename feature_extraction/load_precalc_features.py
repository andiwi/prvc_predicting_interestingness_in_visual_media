import os
import platform
from warnings import warn

import numpy as np
import scipy.io as sio

from file_handler.file_search import get_abs_path_of_file
from file_handler.read_imgs import read_img_names
from Features import Features


def load_precalc_feature(img_dir, feature_name):
    '''
    for all images in path_to_imgs directory: loads precalculated feature :feature_name
    :param path_to_imgs:
    :param feature_name:
    :return: numpy array of features of all imgs in path_to_imgs directory
    '''

    if platform.system() == 'Linux':
        feature_root_dir = os.path.join(img_dir[:img_dir.find('/videos')], 'features') #Linux system
        img_name = img_dir[img_dir.rfind('/') + 1:]
    else:
        feature_root_dir = os.path.join(img_dir[:img_dir.find('\\videos')], 'features') #windows system
        img_name = img_dir[img_dir.rfind('\\') + 1:]

    if feature_name == Features.DenseSIFT_L0:
        feature = _denseSIFT_fe(feature_root_dir, img_name, level=0)
    elif feature_name == Features.DenseSIFT_L1:
        feature = _denseSIFT_fe(feature_root_dir, img_name, level=1)
    elif feature_name == Features.DenseSIFT_L2:
        feature = _denseSIFT_fe(feature_root_dir, img_name, level=2)
    elif feature_name == Features.Hog_L0:
        feature = _hog_fe(feature_root_dir, img_name, level=0)
    elif feature_name == Features.Hog_L1:
        feature = _hog_fe(feature_root_dir, img_name, level=1)
    elif feature_name == Features.Hog_L2:
        feature = _hog_fe(feature_root_dir, img_name, level=2)
    elif feature_name == Features.Lbp_L0:
        feature = _lbp_fe(feature_root_dir, img_name, level=0)
    elif feature_name == Features.Lbp_L1:
        feature = _lbp_fe(feature_root_dir, img_name, level=1)
    elif feature_name == Features.Lbp_L2:
        feature = _lbp_fe(feature_root_dir, img_name, level=2)
    elif feature_name == Features.Gist:
        feature = _gist_fe(feature_root_dir, img_name)
    elif feature_name == Features.Hsv_hist:
        feature = _colorHist_fe(feature_root_dir, img_name)
    elif feature_name == Features.CNN_fc7:
        feature = _cnn_fc7(feature_root_dir, img_name)
    elif feature_name == Features.CNN_prob:
        feature = _cnn_prob(feature_root_dir, img_name)
    else:
        raise NotImplementedError

    return feature


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
            warn('File does not exist. {}'.format(matfile_path))

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
        warn('File does not exist. {}'.format(matfile_path))
        if level == 0:
            return np.zeros(300)
        elif level == 1:
            return np.zeros(1200)
        elif level == 2:
            return np.zeros(4800)
        else:
            raise NotImplementedError

    else:
        img_features = sio.loadmat(matfile_path)
        if level == 0:
            return img_features['hists']['L0'][0][0][:,0]
        elif level == 1:
            return img_features['hists']['L1'][0][0][:,0]
        elif level == 2:
            return img_features['hists']['L2'][0][0][:,0]
        else:
            raise NotImplementedError


def _hog_fe(feature_root_path, img_name, level):
    path = os.path.join(feature_root_path, 'Features_From_FudanUniversity', 'Image_Subtask', 'hog2x2')

    matfile_path = get_abs_path_of_file(path, img_name + '.mat')
    if (matfile_path is None):
        warn('File does not exist. {}'.format(matfile_path))
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
        warn('File does not exist. {}'.format(matfile_path))
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
        warn('File does not exist. {}'.format(matfile_path))
        return np.zeros(512)
    else:
        img_features = sio.loadmat(matfile_path)
        return img_features['descrs'][:,0]

def _cnn_fc7(feature_root_path, img_name):
    path = os.path.join(feature_root_path, 'Features_From_FudanUniversity', 'Image_Subtask', 'CNN', 'fc7')
    version = 1

    matfile_path = get_abs_path_of_file(path, img_name + '.mat')
    if matfile_path is None:
        version = 2
        #check if file withoud .jpg exist (testset)
        matfile_path = get_abs_path_of_file(path, img_name[:len(img_name)-4] + '.mat')

        if matfile_path is None:
            version = 3
            warn('File does not exist. {}'.format(matfile_path))

    if version == 1:
        img_features = sio.loadmat(matfile_path)
        return img_features['AlexNet_fc7'][0]
    elif version == 2:
        img_features = sio.loadmat(matfile_path)
        return img_features['fc7'][0]
    else:
        return np.zeros(4096)

def _cnn_prob(feature_root_path, img_name):
    path = os.path.join(feature_root_path, 'Features_From_FudanUniversity', 'Image_Subtask', 'CNN', 'prob')
    version = 1
    matfile_path = get_abs_path_of_file(path, img_name + '.mat')
    if matfile_path is None:
        # check if file withoud .jpg exist (testset)
        version = 2
        matfile_path = get_abs_path_of_file(path, img_name[:len(img_name) - 4] + '.mat')

        if matfile_path is None:
            version = 3
            warn('File does not exist. {}'.format(matfile_path))

    if version == 1:
        img_features = sio.loadmat(matfile_path)
        return img_features['AlexNet_prob'][0]
    elif version == 2:
        img_features = sio.loadmat(matfile_path)
        return img_features['prob'][0]
    else:
        return np.zeros(1000)

